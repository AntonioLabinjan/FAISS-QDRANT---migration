import cv2
import numpy as np
import torch
import redis
import json
import time
import os
import traceback
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime


NODE_ID = os.getenv("NODE_ID", "vision_00")
GATE_ID = os.getenv("GATE_ID", "gate_00")
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID", "fipu")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6382))
VISION_QUEUE = os.getenv("VISION_QUEUE", "queue_vision")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

THRESHOLD_DISTANCE = float(os.getenv("THRESHOLD_DISTANCE", 0.85))
THRESHOLD_TIME = float(os.getenv("THRESHOLD_TIME", 60))

GRID_CELL = 80
VECTOR_DIM = 512

MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", 80))

face_state = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [VISION:{NODE_ID}] {msg}", flush=True)


def normalize(v):
    v = np.array(v, dtype=np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def safe_vector(v, dim):
    v = np.array(v, dtype=np.float32).flatten()

    if v.shape[0] > dim:
        v = v[:dim]
    else:
        v = np.pad(v, (0, dim - v.shape[0]))

    return v.astype(np.float32)


def face_key(x, y):
    return f"{x // GRID_CELL}_{y // GRID_CELL}"


def cosine_distance(a, b):
    a = normalize(a)
    b = normalize(b)
    return 1.0 - float(np.dot(a, b))


def should_classify(key, new_embedding):
    current_time = datetime.now()

    if key not in face_state:
        face_state[key] = {
            "embedding": new_embedding,
            "timestamp": current_time,
        }
        return True, 1.0

    prev = face_state[key]
    dist = cosine_distance(new_embedding, prev["embedding"])
    elapsed = (current_time - prev["timestamp"]).total_seconds()

    if dist > THRESHOLD_DISTANCE:
        face_state[key] = {
            "embedding": new_embedding,
            "timestamp": current_time,
        }
        return True, dist

    if elapsed > THRESHOLD_TIME:
        face_state[key] = {
            "embedding": new_embedding,
            "timestamp": current_time,
        }
        return True, dist

    return False, dist


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

    emb = emb.detach().cpu().numpy().flatten()
    emb = safe_vector(emb, VECTOR_DIM)

    return normalize(emb)


def build_payload(embedding, frame_id):
    now = time.time()

    return {
        "meta": {
            "node_id": NODE_ID,
            "gate_id": GATE_ID,
            "organization_id": ORGANIZATION_ID,
            "node_type": "vision",
            "embedding_type": "clip-vit-base-patch32",
            "vector_dim": VECTOR_DIM,
            "timestamp": now,
            "frame_id": frame_id,
        },
        "data": {
            "embedding": normalize(embedding).tolist(),
        },
    }


def load_haar_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    log(f"Loaded Haar face detector: {cascade_path}")

    return detector


def detect_faces(detector, frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )

    return faces


def main():
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True,
    )

    r.ping()

    log(f"Redis connected | {REDIS_HOST}:{REDIS_PORT}")
    log(f"Queue: {VISION_QUEUE}")
    log(f"Gate: {GATE_ID}")
    log(f"Organization: {ORGANIZATION_ID}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    log("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    log("CLIP ready")

    face_detector = load_haar_detector()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    log(f"Camera opened | index={CAMERA_INDEX}")

    frame_id = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                log("Camera frame failed")
                break

            frame_id += 1

            faces = detect_faces(face_detector, frame)
            log(f"Frame {frame_id} | faces={len(faces)}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for i, (x, y, fw, fh) in enumerate(faces):
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + fw)
                y2 = int(y + fh)

                if x2 <= x1 or y2 <= y1:
                    continue

                key = face_key(x1, y1)

                current_time = datetime.now()

                if key in face_state:
                    elapsed = (current_time - face_state[key]["timestamp"]).total_seconds()

                    if elapsed < 2.0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        continue

                face_img = rgb[y1:y2, x1:x2]

                try:
                    emb = extract_clip_image_embedding(
                        model=model,
                        processor=processor,
                        image_rgb=face_img,
                        device=device,
                    )

                except Exception as e:
                    log(f"Embedding extraction failed: {e}")
                    traceback.print_exc()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    continue

                trigger, dist = should_classify(key, emb)

                log(f"Face {i} | key={key} | dist={dist:.4f}")

                if trigger:
                    payload = build_payload(emb, frame_id)
                    r.lpush(VISION_QUEUE, json.dumps(payload))

                    log(
                        f"TRIGGER -> pushed | queue={VISION_QUEUE} | "
                        f"dim={len(payload['data']['embedding'])}"
                    )

                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.imshow("Vision Node", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log("Shutdown requested")
                break

    except Exception as e:
        log(f"CRASH: {e}")
        traceback.print_exc()

    finally:
        log("Cleaning up")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()