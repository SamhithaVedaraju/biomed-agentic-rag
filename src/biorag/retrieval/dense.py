from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Instruction prefixes each embedding family was trained with. Leaving them off costs
# several points of retrieval quality.
_PREFIXES: dict[str, tuple[str, str]] = {
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
}


def _prefixes(model_name: str) -> tuple[str, str]:
    name = model_name.lower()
    for family, prefixes in _PREFIXES.items():
        if family in name:
            return prefixes
    return "", ""


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix, self.passage_prefix = _prefixes(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            [self.passage_prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)


class VectorStore:
    """Qdrant in embedded (on-disk) mode, so no server is needed. Point it at a Qdrant server URL
    instead for a shared deployment."""

    def __init__(self, path: Path, collection: str):
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray, batch: int = 512) -> None:
        for start in range(0, len(chunk_ids), batch):
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=start + i, vector=vec.tolist(), payload={"chunk_id": cid})
                    for i, (cid, vec) in enumerate(
                        zip(chunk_ids[start : start + batch], vectors[start : start + batch],
                            strict=True)
                    )
                ],
            )

    def search(self, vector: np.ndarray, k: int) -> list[str]:
        hits = self.client.query_points(self.collection, query=vector.tolist(), limit=k).points
        return [h.payload["chunk_id"] for h in hits]
