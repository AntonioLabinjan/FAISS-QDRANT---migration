import json
import redis
import numpy as np
import os
import time
import datetime
import pandas as pd
from flask import Flask, Response, send_file
from multiprocessing import Process, Manager
import io
import traceback
import uuid
from collections import Counter

import torch
import torchaudio
import soundfile as sf
from transformers import CLIPProcessor, CLIPModel
import cv2

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6382))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_ORGANIZATION_ID = os.getenv("ORGANIZATION_ID", "fipu")
RESET_QDRANT_ON_START = os.getenv("RESET_QDRANT_ON_START", "0") == "1"

EMBEDDING_QUEUE_VISION = "queue_vision"
EMBEDDING_QUEUE_AUDIO = "queue_audio"

OUTPUT_CHANNEL = "system_events"
FUSION_CHANNEL = "fusion_events"

FACE_COLLECTION = "perceptryx_faces"
VOICE_COLLECTION = "perceptryx_voices"

FACE_DIM = 512
VOICE_DIM = 128

SAVE_INTERVAL = 5

THRESHOLD_VISION = float(os.getenv("THRESHOLD_VISION", 0.70))
THRESHOLD_AUDIO = float(os.getenv("THRESHOLD_AUDIO", 0.010))

FUSION_WINDOW = 5.0

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/antonio/Desktop/NMDU/triplet_run_2/best_triplet_model.pth",
)

def log(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def norm(v):
    v = np.array(v, dtype=np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def safe_vector(v, dim):
    v = np.array(v, dtype=np.float32).flatten()

    if v.shape[0] > dim:
        v = v[:dim]
    else:
        v = np.pad(v, (0, dim - v.shape[0]))

    return v.astype(np.float32)


def deterministic_uuid(value):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


def load_audio(path):
    try:
        wav, sr = sf.read(path)
    except Exception:
        wav, sr = torchaudio.load(path)

    wav = torch.tensor(wav).float()

    if wav.ndim == 2:
        wav = wav.mean(dim=1)

    wav = wav.unsqueeze(0)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav

class VoiceNetEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        def block(i, o):
            return torch.nn.Sequential(
                torch.nn.Conv2d(i, o, 3, padding=1),
                torch.nn.BatchNorm2d(o),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )

        self.features = torch.nn.Sequential(
            block(1, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 256),
        )

        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))

class QdrantBiometricStore:
    def __init__(self, url):
        self.client = QdrantClient(url=url)

    def ensure_collection(self, collection_name, dim, reset=False):
        exists = self.client.collection_exists(collection_name)

        if exists and reset:
            self.client.delete_collection(collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def count(self, collection_name):
        try:
            return self.client.count(
                collection_name=collection_name,
                exact=True,
            ).count
        except Exception:
            return 0

    def enroll_face(self, person_id, name, organization_id, embedding):
        embedding = safe_vector(norm(embedding), FACE_DIM)
        sample_id = f"face_{organization_id}_{person_id}_centroid"

        self.client.upsert(
            collection_name=FACE_COLLECTION,
            points=[
                PointStruct(
                    id=deterministic_uuid(sample_id),
                    vector=embedding.tolist(),
                    payload={
                        "sample_id": sample_id,
                        "person_id": person_id,
                        "name": name,
                        "id": person_id,
                        "organization_id": organization_id,
                        "modality": "vision",
                        "embedding_model": "clip-vit-base-patch32",
                        "sample_type": "centroid",
                        "active": True,
                    },
                )
            ],
            wait=True,
        )

    def enroll_voice(self, person_id, name, organization_id, embedding):
        embedding = safe_vector(norm(embedding), VOICE_DIM)
        sample_id = f"voice_{organization_id}_{person_id}_centroid"

        self.client.upsert(
            collection_name=VOICE_COLLECTION,
            points=[
                PointStruct(
                    id=deterministic_uuid(sample_id),
                    vector=embedding.tolist(),
                    payload={
                        "sample_id": sample_id,
                        "person_id": person_id,
                        "name": name,
                        "id": person_id,
                        "organization_id": organization_id,
                        "modality": "audio",
                        "embedding_model": "voicenet-v1",
                        "sample_type": "centroid",
                        "active": True,
                    },
                )
            ],
            wait=True,
        )

    def recognize(self, collection_name, embedding, dim, organization_id, threshold, limit=5):
        embedding = safe_vector(norm(embedding), dim)

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=organization_id),
                ),
                FieldCondition(
                    key="active",
                    match=MatchValue(value=True),
                ),
            ]
        )

        matches = self.client.query_points(
            collection_name=collection_name,
            query=embedding.tolist(),
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        ).points

        return vote(matches, threshold)


def vote(matches, threshold):
    if not matches:
        return "unknown", 0.0, []

    accepted = [m for m in matches if float(m.score) >= threshold]

    if not accepted:
        return "unknown", float(matches[0].score), matches

    person_votes = Counter(m.payload["person_id"] for m in accepted)
    winner_person_id, _ = person_votes.most_common(1)[0]

    winner_matches = [
        m for m in accepted
        if m.payload["person_id"] == winner_person_id
    ]

    best = max(winner_matches, key=lambda m: float(m.score))
    identity = best.payload.get("id") or best.payload.get("person_id") or "unknown"
    score = float(best.score)

    return identity, score, matches

def extract_clip_image_embedding(model, processor, image_rgb, device):
    inputs = processor(images=image_rgb, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model.get_image_features(pixel_values=pixel_values)

        if isinstance(outputs, torch.Tensor):
            emb = outputs
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            emb = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb = outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, (tuple, list)):
            emb = outputs[0]
        else:
            raise RuntimeError(f"Unsupported CLIP output type: {type(outputs)}")

    return emb.detach().cpu().numpy().flatten()


def load_voice_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing voice model file: {model_path}")

    model = VoiceNetEmbedding(embedding_dim=VOICE_DIM).to(device)

    state = torch.load(model_path, map_location=device)
    fixed = {k.replace("embedding_head", "head"): v for k, v in state.items()}

    model.load_state_dict(fixed, strict=True)
    model.eval()

    return model


def extract_voice_embedding(model, wav, mel_transform, db_transform, device):
    wav = wav.to(device)

    with torch.no_grad():
        spec = mel_transform(wav)
        spec = db_transform(spec).unsqueeze(0)

        spec = torch.nn.functional.interpolate(
            spec,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        spec = (spec - spec.mean()) / (spec.std() + 1e-7)

        emb = model(spec).cpu().numpy().flatten()
        emb = safe_vector(emb, VOICE_DIM)
        emb = norm(emb)

    return emb

def preload_faces(store):
    base = "./known_faces"

    store.ensure_collection(
        collection_name=FACE_COLLECTION,
        dim=FACE_DIM,
        reset=RESET_QDRANT_ON_START,
    )

    if not os.path.exists(base):
        log("vision", f"No {base} directory found")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    for person in os.listdir(base):
        pdir = os.path.join(base, person)

        if not os.path.isdir(pdir):
            continue

        embs = []

        for filename in os.listdir(pdir):
            path = os.path.join(pdir, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                emb = extract_clip_image_embedding(
                    model=model,
                    processor=processor,
                    image_rgb=img,
                    device=device,
                )

                emb = safe_vector(emb, FACE_DIM)
                embs.append(norm(emb))

            except Exception as e:
                log("vision", f"Failed face image {path}: {e}")
                traceback.print_exc()

        if embs:
            centroid = norm(np.mean(embs, axis=0))

            store.enroll_face(
                person_id=person,
                name=person,
                organization_id=DEFAULT_ORGANIZATION_ID,
                embedding=centroid,
            )

            log("vision", f"Enrolled face centroid: {person}")



def preload_voices(store):
    base = "./known_voices"

    store.ensure_collection(
        collection_name=VOICE_COLLECTION,
        dim=VOICE_DIM,
        reset=RESET_QDRANT_ON_START,
    )

    if not os.path.exists(base):
        log("audio", f"No {base} directory found")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_voice_model(MODEL_PATH, device)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
    ).to(device)

    db_transform = torchaudio.transforms.AmplitudeToDB().to(device)

    allowed = (".mp3", ".wav", ".flac", ".ogg", ".m4a")

    for person in os.listdir(base):
        pdir = os.path.join(base, person)

        if not os.path.isdir(pdir):
            continue

        embs = []

        for filename in os.listdir(pdir):
            if not filename.lower().endswith(allowed):
                continue

            path = os.path.join(pdir, filename)

            try:
                wav = load_audio(path)

                emb = extract_voice_embedding(
                    model=model,
                    wav=wav,
                    mel_transform=mel_transform,
                    db_transform=db_transform,
                    device=device,
                )

                embs.append(emb)

            except Exception as e:
                log("audio", f"Failed audio file {path}: {e}")
                traceback.print_exc()

        if embs:
            centroid = norm(np.mean(embs, axis=0))

            store.enroll_voice(
                person_id=person,
                name=person,
                organization_id=DEFAULT_ORGANIZATION_ID,
                embedding=centroid,
            )

            log("audio", f"Enrolled VoiceNet voice centroid: {person}")


def extract_core_identity(identity):
    if identity is None:
        return "unknown"

    raw = str(identity).strip()
    lowered = raw.lower()

    if lowered in ("unknown", "none", "null", ""):
        return "unknown"

    for prefix in ("face_", "voice_", "vision_", "audio_"):
        if lowered.startswith(prefix):
            return lowered[len(prefix):].strip()

    return lowered


def harmonic_mean(a, b):
    if a + b == 0:
        return 0.0

    return 2 * a * b / (a + b)


def run_fusion(r, node, new_entry, shared_metrics, now_ts, date_str, time_str):
    buffer_key = f"identity_buffer:{node}"
    lock_key = f"{buffer_key}:lock"

    lock = r.lock(lock_key, timeout=3, blocking_timeout=1)
    acquired = False

    try:
        acquired = lock.acquire(blocking=True)

        if not acquired:
            log("fusion", f"Could not acquire fusion lock for node={node}")
            return

        raw_buffer = r.get(buffer_key)
        buffer = json.loads(raw_buffer) if raw_buffer else []

        buffer.append(new_entry)

        buffer = [
            e for e in buffer
            if now_ts - float(e["timestamp"]) <= FUSION_WINDOW
        ]

        vision_entries = [e for e in buffer if e["modality"] == "vision"]
        audio_entries = [e for e in buffer if e["modality"] == "audio"]

        if not vision_entries or not audio_entries:
            r.set(buffer_key, json.dumps(buffer), ex=int(FUSION_WINDOW) + 5)

            modalities = [e["modality"] for e in buffer]
            log(
                "fusion",
                f"Waiting for pair | gate={node} | buffer_modalities={modalities}",
            )
            return

        v = max(vision_entries, key=lambda e: e["timestamp"])
        a = max(audio_entries, key=lambda e: e["timestamp"])

        dt = abs(float(v["timestamp"]) - float(a["timestamp"]))

        if dt > FUSION_WINDOW:
            r.set(buffer_key, json.dumps(buffer), ex=int(FUSION_WINDOW) + 5)
            log(
                "fusion",
                f"Pair too old | gate={node} | dt={dt:.2f}s | window={FUSION_WINDOW}s",
            )
            return

        v_core = extract_core_identity(v["identity"])
        a_core = extract_core_identity(a["identity"])

        if v_core == a_core and v_core != "unknown":
            fusion_identity = v_core
            fusion_status = "CONFIRMED"
            combined_score = harmonic_mean(v["score"], a["score"])

        elif v_core != "unknown" and a_core == "unknown":
            fusion_identity = v_core
            fusion_status = "VISION_ONLY"
            combined_score = v["score"]

        elif a_core != "unknown" and v_core == "unknown":
            fusion_identity = a_core
            fusion_status = "AUDIO_ONLY"
            combined_score = a["score"]

        else:
            fusion_identity = "unconfirmed"
            fusion_status = "CONFLICT" if v_core != a_core else "UNKNOWN"
            combined_score = harmonic_mean(v["score"], a["score"])

        fusion_event = {
            "type": "fusion",
            "node": node,
            "gate": node,
            "identity": fusion_identity,
            "id": fusion_identity,
            "status": fusion_status,
            "combined_score": round(combined_score, 4),
            "vision_identity": v["identity"],
            "vision_core": v_core,
            "vision_score": round(v["score"], 4),
            "audio_identity": a["identity"],
            "audio_core": a_core,
            "audio_score": round(a["score"], 4),
            "delta_seconds": round(dt, 3),
            "date": date_str,
            "time": time_str,
        }

        r.publish(FUSION_CHANNEL, json.dumps(fusion_event))

        log(
            "fusion",
            f"gate={node} | {fusion_status} | identity={fusion_identity} "
            f"| dt={dt:.3f}s | combined_score={combined_score:.4f} "
            f"| vision={v['identity']} -> {v_core} ({v['score']:.4f}) "
            f"| audio={a['identity']} -> {a_core} ({a['score']:.4f})",
        )

        fusion_log_key = "fusion_log"
        prev_log = list(shared_metrics.get(fusion_log_key, []))

        prev_log.append({
            "Date": date_str,
            "Time": time_str,
            "Name/ID": fusion_identity,
            "Status": fusion_status,
            "Combined_Score": round(combined_score, 4),
            "Vision_Identity": v["identity"],
            "Vision_Core": v_core,
            "Vision_Score": round(v["score"], 4),
            "Audio_Identity": a["identity"],
            "Audio_Core": a_core,
            "Audio_Score": round(a["score"], 4),
            "Delta_Seconds": round(dt, 3),
            "Gate": node,
        })

        shared_metrics[fusion_log_key] = prev_log

        shared_metrics[f"fusion:{fusion_status}"] = (
            shared_metrics.get(f"fusion:{fusion_status}", 0) + 1
        )

        r.delete(buffer_key)

    except Exception as e:
        log("fusion", f"Fusion error: {e}")
        traceback.print_exc()

    finally:
        if acquired:
            try:
                lock.release()
            except Exception:
                pass


def worker(queue, dim, tag, shared_metrics):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    store = QdrantBiometricStore(QDRANT_URL)

    if tag == "vision":
        collection_name = FACE_COLLECTION
        threshold = THRESHOLD_VISION
        preload_faces(store)
    else:
        collection_name = VOICE_COLLECTION
        threshold = THRESHOLD_AUDIO
        preload_voices(store)

    count = store.count(collection_name)
    log(tag, f"Worker ready — {count} known identities in Qdrant | threshold={threshold}")

    while True:
        _, raw = r.blpop(queue)

        try:
            obj = json.loads(raw)

            emb = np.array(obj["data"]["embedding"], dtype=np.float32)
            node = obj["meta"]["node_id"]
            gate = obj["meta"].get("gate_id", node)
            organization_id = obj["meta"].get("organization_id", DEFAULT_ORGANIZATION_ID)

            identity, score, matches = store.recognize(
                collection_name=collection_name,
                embedding=emb,
                dim=dim,
                organization_id=organization_id,
                threshold=threshold,
                limit=5,
            )

            now_ts = time.time()
            now_dt = datetime.datetime.now()
            date_str = now_dt.strftime("%Y-%m-%d")
            time_str = now_dt.strftime("%H:%M:%S")

            nodes_key = f"node:{node}"
            shared_metrics[nodes_key] = shared_metrics.get(nodes_key, 0) + 1
            shared_metrics[f"type:{tag}"] = shared_metrics.get(f"type:{tag}", 0) + 1

            bucket = "known" if identity != "unknown" else "unknown"
            shared_metrics[bucket] = shared_metrics.get(bucket, 0) + 1

            scores_key = "scores"
            prev_scores = list(shared_metrics.get(scores_key, []))
            prev_scores = (prev_scores + [score])[-200:]
            shared_metrics[scores_key] = prev_scores

            log_key = "attendance_log"
            prev_log = list(shared_metrics.get(log_key, []))

            prev_log.append({
                "Date": date_str,
                "Time": time_str,
                "Name/ID": identity,
                "Modality": tag,
                "Gate": gate,
                "Node": node,
                "Organization": organization_id,
                "Confidence_Score": round(score, 4),
                "Status": "PRESENT" if identity != "unknown" else "UNAUTHORIZED",
            })

            shared_metrics[log_key] = prev_log

            r.publish(OUTPUT_CHANNEL, json.dumps({
                "type": tag,
                "id": identity,
                "score": score,
                "node": node,
                "gate": gate,
                "organization_id": organization_id,
                "date": date_str,
                "time": time_str,
            }))

            if matches:
                debug_top = [
                    {
                        "id": m.payload.get("id"),
                        "score": round(float(m.score), 4),
                    }
                    for m in matches[:3]
                ]
            else:
                debug_top = []

            log(
                tag,
                f"{identity} | score={score:.4f} | node={node} | gate={gate} | top={debug_top}",
            )

            new_entry = {
                "modality": tag,
                "identity": identity,
                "score": score,
                "timestamp": now_ts,
            }

            run_fusion(
                r=r,
                node=gate,
                new_entry=new_entry,
                shared_metrics=shared_metrics,
                now_ts=now_ts,
                date_str=date_str,
                time_str=time_str,
            )

        except Exception as e:
            log(tag, f"Worker error: {e}")
            traceback.print_exc()


app = Flask(__name__)
shared_metrics = None


@app.route("/events")
def events():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    def stream():
        pubsub = r.pubsub()
        pubsub.subscribe(OUTPUT_CHANNEL)

        for msg in pubsub.listen():
            if msg["type"] == "message":
                yield f"data: {msg['data']}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/fusion-events")
def fusion_events():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    def stream():
        pubsub = r.pubsub()
        pubsub.subscribe(FUSION_CHANNEL)

        for msg in pubsub.listen():
            if msg["type"] == "message":
                yield f"data: {msg['data']}\n\n"

    return Response(stream(), mimetype="text/event-stream")


@app.route("/export")
def export_excel():
    log_data = list(shared_metrics.get("attendance_log", []))

    if not log_data:
        return "No data to export", 400

    df = pd.DataFrame(log_data)
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Attendance")

    output.seek(0)

    filename = f"Attendance_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/export-fusion")
def export_fusion_excel():
    log_data = list(shared_metrics.get("fusion_log", []))

    if not log_data:
        return "No fusion data to export", 400

    df = pd.DataFrame(log_data)
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Fusion")

    output.seek(0)

    filename = f"Fusion_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/metrics")
def metrics():
    nodes = {
        k[5:]: v
        for k, v in shared_metrics.items()
        if k.startswith("node:")
    }

    scores = list(shared_metrics.get("scores", []))

    fusion_stats = {
        k[7:]: v
        for k, v in shared_metrics.items()
        if k.startswith("fusion:")
    }

    return {
        "nodes": nodes,
        "known": shared_metrics.get("known", 0),
        "unknown": shared_metrics.get("unknown", 0),
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "fusion_stats": fusion_stats,
        "backend": "qdrant",
        "qdrant_url": QDRANT_URL,
        "face_collection": FACE_COLLECTION,
        "voice_collection": VOICE_COLLECTION,
        "vision_threshold": THRESHOLD_VISION,
        "audio_threshold": THRESHOLD_AUDIO,
    }


@app.route("/")
def home():
    return open("ui_school.html").read()


if __name__ == "__main__":
    manager = Manager()
    shared_metrics = manager.dict()

    processes = [
        Process(
            target=worker,
            args=(EMBEDDING_QUEUE_VISION, FACE_DIM, "vision", shared_metrics),
            name="VisionWorker",
            daemon=True,
        ),
        Process(
            target=worker,
            args=(EMBEDDING_QUEUE_AUDIO, VOICE_DIM, "audio", shared_metrics),
            name="AudioWorker",
            daemon=True,
        ),
    ]

    for p in processes:
        p.start()
        log("main", f"{p.name} started (pid={p.pid})")

    app.run(host="0.0.0.0", port=5000, threaded=True)