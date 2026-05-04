import uuid
from collections import Counter
from typing import List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


QDRANT_URL = "http://localhost:6333"

FACE_COLLECTION = "perceptryx_faces"
VOICE_COLLECTION = "perceptryx_voices"

FACE_DIM = 512
VOICE_DIM = 128

FACE_THRESHOLD = 0.70
VOICE_THRESHOLD = 0.70


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def noisy_sample(base: np.ndarray, noise: float = 0.03) -> np.ndarray:
    return normalize(base + np.random.normal(0, noise, size=base.shape))


def deterministic_uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, value))


class QdrantBiometricStore:
    def __init__(self, url: str):
        self.client = QdrantClient(url=url)

    def reset_collections(self) -> None:
        for collection_name, dim in [
            (FACE_COLLECTION, FACE_DIM),
            (VOICE_COLLECTION, VOICE_DIM),
        ]:
            if self.client.collection_exists(collection_name):
                self.client.delete_collection(collection_name)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def enroll_face(
        self,
        sample_id: str,
        person_id: str,
        name: str,
        organization_id: str,
        embedding: np.ndarray,
        angle: str,
    ) -> None:
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
                        "organization_id": organization_id,
                        "modality": "face",
                        "embedding_model": "clip-vit-base-patch32",
                        "angle": angle,
                        "active": True,
                    },
                )
            ],
            wait=True,
        )

    def enroll_voice(
        self,
        sample_id: str,
        person_id: str,
        name: str,
        organization_id: str,
        embedding: np.ndarray,
        phrase: str,
    ) -> None:
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
                        "organization_id": organization_id,
                        "modality": "voice",
                        "embedding_model": "voicenet-v1",
                        "phrase": phrase,
                        "active": True,
                    },
                )
            ],
            wait=True,
        )

    def search_face(
        self,
        embedding: np.ndarray,
        organization_id: str,
        limit: int = 5,
    ):
        return self._search(
            collection_name=FACE_COLLECTION,
            embedding=embedding,
            organization_id=organization_id,
            limit=limit,
        )

    def search_voice(
        self,
        embedding: np.ndarray,
        organization_id: str,
        limit: int = 5,
    ):
        return self._search(
            collection_name=VOICE_COLLECTION,
            embedding=embedding,
            organization_id=organization_id,
            limit=limit,
        )

    def _search(
        self,
        collection_name: str,
        embedding: np.ndarray,
        organization_id: str,
        limit: int,
    ):
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

        return self.client.query_points(
            collection_name=collection_name,
            query=embedding.tolist(),
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        ).points


def vote(matches, threshold: float) -> Dict[str, Any]:
    accepted = [m for m in matches if m.score >= threshold]

    if not accepted:
        return {
            "recognized": False,
            "reason": "no_match_above_threshold",
            "best_score": matches[0].score if matches else None,
        }

    person_votes = Counter(m.payload["person_id"] for m in accepted)
    winner_person_id, vote_count = person_votes.most_common(1)[0]

    winner_matches = [
        m for m in accepted
        if m.payload["person_id"] == winner_person_id
    ]

    best = max(winner_matches, key=lambda m: m.score)

    return {
        "recognized": True,
        "person_id": winner_person_id,
        "name": best.payload["name"],
        "best_score": best.score,
        "votes": vote_count,
        "sample_id": best.payload["sample_id"],
    }


def print_matches(title: str, matches) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    for index, match in enumerate(matches, start=1):
        payload = match.payload
        print(
            f"{index}. "
            f"name={payload['name']}, "
            f"person_id={payload['person_id']}, "
            f"sample_id={payload['sample_id']}, "
            f"score={match.score:.4f}"
        )


def seed_demo_data(store: QdrantBiometricStore):
    np.random.seed(42)

    people = [
        {
            "person_id": "p_001",
            "name": "Antonio Labinjan",
            "organization_id": "fipu",
        },
        {
            "person_id": "p_002",
            "name": "Luka Horvat",
            "organization_id": "fipu",
        },
        {
            "person_id": "p_003",
            "name": "Ivan Ivić",
            "organization_id": "neuromorphyx",
        },
    ]

    face_bases = {}
    voice_bases = {}

    for person in people:
        person_id = person["person_id"]
        face_bases[person_id] = normalize(np.random.normal(size=FACE_DIM))
        voice_bases[person_id] = normalize(np.random.normal(size=VOICE_DIM))

    for person in people:
        person_id = person["person_id"]
        name = person["name"]
        organization_id = person["organization_id"]

        for i, angle in enumerate(["front", "slight_left", "slight_right"], start=1):
            sample_id = f"face_{person_id}_{i:03d}"
            store.enroll_face(
                sample_id=sample_id,
                person_id=person_id,
                name=name,
                organization_id=organization_id,
                embedding=noisy_sample(face_bases[person_id]),
                angle=angle,
            )

        for i, phrase in enumerate(["free_speech", "different_sentence", "mild_noise"], start=1):
            sample_id = f"voice_{person_id}_{i:03d}"
            store.enroll_voice(
                sample_id=sample_id,
                person_id=person_id,
                name=name,
                organization_id=organization_id,
                embedding=noisy_sample(voice_bases[person_id]),
                phrase=phrase,
            )

    return face_bases, voice_bases


def main() -> None:
    store = QdrantBiometricStore(QDRANT_URL)

    print("Resetting Qdrant collections...")
    store.reset_collections()

    print("Seeding demo biometric samples...")
    face_bases, voice_bases = seed_demo_data(store)

    antonio_face_query = noisy_sample(face_bases["p_001"], noise=0.025)
    antonio_voice_query = noisy_sample(voice_bases["p_001"], noise=0.025)

    face_matches = store.search_face(
        embedding=antonio_face_query,
        organization_id="fipu",
        limit=5,
    )

    voice_matches = store.search_voice(
        embedding=antonio_voice_query,
        organization_id="fipu",
        limit=5,
    )

    print_matches("Face search result", face_matches)
    print_matches("Voice search result", voice_matches)

    face_decision = vote(face_matches, FACE_THRESHOLD)
    voice_decision = vote(voice_matches, VOICE_THRESHOLD)

    print("\nFace decision")
    print(face_decision)

    print("\nVoice decision")
    print(voice_decision)

    fusion_match = (
        face_decision["recognized"]
        and voice_decision["recognized"]
        and face_decision["person_id"] == voice_decision["person_id"]
    )

    print("\nFusion decision")
    if fusion_match:
        print(
            {
                "fusion_match": True,
                "person_id": face_decision["person_id"],
                "name": face_decision["name"],
                "face_score": round(face_decision["best_score"], 4),
                "voice_score": round(voice_decision["best_score"], 4),
            }
        )
    else:
        print({"fusion_match": False})


if __name__ == "__main__":
    main()