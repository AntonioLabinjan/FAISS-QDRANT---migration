import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import pyaudio
import json
import redis
import time
import traceback
import os


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/antonio/Desktop/NMDU/triplet_run_2/best_triplet_model.pth",
)

MIC_NODE = int(os.getenv("MIC_NODE", 0))

MIC_INDEX_BY_NODE = {
    0: 0,
    1: 5,
}

MIC_INDEX = int(os.getenv("MIC_INDEX", MIC_INDEX_BY_NODE.get(MIC_NODE, 0)))

NODE_ID = os.getenv("NODE_ID", f"audio_{MIC_NODE:02d}")
GATE_ID = os.getenv("GATE_ID", "gate_00")
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID", "fipu")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6382))
AUDIO_QUEUE = os.getenv("AUDIO_QUEUE", "queue_audio")

MODEL_SAMPLE_RATE = 16000
CAPTURE_SAMPLE_RATE = int(os.getenv("CAPTURE_SAMPLE_RATE", 48000))

CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", 1.0))
CHUNK = int(CAPTURE_SAMPLE_RATE * CHUNK_SECONDS)

DECOUPLING_TIME = float(os.getenv("DECOUPLING_TIME", 1.5))
PEAK_THRESHOLD = float(os.getenv("PEAK_THRESHOLD", 0.03))

VECTOR_DIM = 128


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [AUDIO:{NODE_ID}] {msg}", flush=True)


def normalize(v):
    v = np.array(v, dtype=np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


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


def list_input_devices(pa):
    log("Available PyAudio input devices:")

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)

        if info["maxInputChannels"] > 0:
            log(
                f"{i}: {info['name']} | "
                f"channels={info['maxInputChannels']} | "
                f"default_rate={int(info['defaultSampleRate'])}"
            )


def get_input_device_info(pa, index):
    info = pa.get_device_info_by_index(index)

    if info["maxInputChannels"] <= 0:
        raise RuntimeError(f"Selected device index {index} is not an input device.")

    return info


class AudioProcessor:
    def __init__(self, model_path, device):
        self.device = device
        self.last_send_time = 0

        self.model = VoiceNetEmbedding(embedding_dim=VECTOR_DIM).to(device)
        self._load_model(model_path)

        if CAPTURE_SAMPLE_RATE != MODEL_SAMPLE_RATE:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=CAPTURE_SAMPLE_RATE,
                new_freq=MODEL_SAMPLE_RATE,
            ).to(device)
        else:
            self.resampler = None

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=MODEL_SAMPLE_RATE,
            n_mels=128,
            n_fft=1024,
            hop_length=512,
        ).to(device)

        self.db_transform = torchaudio.transforms.AmplitudeToDB().to(device)

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {path}")

        state = torch.load(path, map_location=self.device)
        fixed = {k.replace("embedding_head", "head"): v for k, v in state.items()}

        self.model.load_state_dict(fixed, strict=True)
        self.model.eval()

        log("Model loaded and patched.")

    def should_process(self, audio_np):
        peak = float(np.max(np.abs(audio_np)))
        now = time.time()

        if peak > PEAK_THRESHOLD and (now - self.last_send_time) > DECOUPLING_TIME:
            return True, peak

        return False, peak

    def process_and_embed(self, audio_np, peak):
        now = time.time()

        wav = torch.from_numpy(audio_np).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            if self.resampler is not None:
                wav = self.resampler(wav)

            spec = self.mel_transform(wav)
            spec = self.db_transform(spec).unsqueeze(0)

            spec = F.interpolate(
                spec,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            )

            spec = (spec - spec.mean()) / (spec.std() + 1e-7)

            emb = self.model(spec).cpu().numpy().flatten()
            emb = normalize(emb)

        self.last_send_time = now

        return {
            "meta": {
                "node_id": NODE_ID,
                "gate_id": GATE_ID,
                "organization_id": ORGANIZATION_ID,
                "node_type": "audio",
                "embedding_type": "voicenet-v1",
                "vector_dim": VECTOR_DIM,
                "timestamp": now,
                "peak": round(peak, 4),
                "mic_node": MIC_NODE,
                "mic_index": MIC_INDEX,
            },
            "data": {
                "embedding": emb.tolist(),
            },
        }


def main():
    stream = None
    pa = None

    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=True,
        )

        r.ping()

        log(f"Redis connected | {REDIS_HOST}:{REDIS_PORT}")
        log(f"Queue: {AUDIO_QUEUE}")
        log(f"Gate: {GATE_ID}")
        log(f"Organization: {ORGANIZATION_ID}")

    except Exception as e:
        log(f"Redis connection failed: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = AudioProcessor(MODEL_PATH, device)

        pa = pyaudio.PyAudio()

        list_input_devices(pa)

        selected_device = get_input_device_info(pa, MIC_INDEX)

        log(f"Selected MIC_NODE: {MIC_NODE}")
        log(f"Selected MIC_INDEX: {MIC_INDEX}")
        log(f"Selected microphone: {selected_device['name']}")
        log(f"Capture sample rate: {CAPTURE_SAMPLE_RATE}")
        log(f"Model sample rate: {MODEL_SAMPLE_RATE}")

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=CAPTURE_SAMPLE_RATE,
            input=True,
            input_device_index=MIC_INDEX,
            frames_per_buffer=CHUNK,
        )

        log(f"System active | Node={NODE_ID} | Gate={GATE_ID} | Device={device}")

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            triggered, peak = processor.should_process(audio_np)

            if triggered:
                log(f"Triggered | peak={peak:.4f}")

                payload = processor.process_and_embed(audio_np, peak)
                r.lpush(AUDIO_QUEUE, json.dumps(payload))

                log(
                    f"Embedding pushed | queue={AUDIO_QUEUE} | "
                    f"dim={len(payload['data']['embedding'])}"
                )

    except KeyboardInterrupt:
        log("Shutting down...")

    except Exception as e:
        log(f"CRASH: {e}")
        traceback.print_exc()

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()

        if pa is not None:
            pa.terminate()


if __name__ == "__main__":
    main()