"""Hybrid retriever: BM25 + dense vectors, fused with RRF, then re-ranked by a cross-encoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sentence_transformers import CrossEncoder

from biorag.config import Settings
from biorag.data.documents import Chunk
from biorag.retrieval.dense import Embedder, VectorStore
from biorag.retrieval.fusion import reciprocal_rank_fusion
from biorag.retrieval.sparse import BM25Index


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]: ...


def chunks_path(index_dir: Path) -> Path:
    return index_dir / "chunks.jsonl"


def bm25_path(index_dir: Path) -> Path:
    return index_dir / "bm25.pkl"


def qdrant_path(index_dir: Path) -> Path:
    return index_dir / "qdrant"


class HybridRetriever:
    def __init__(self, settings: Settings, mode: str = "hybrid", rerank: bool = True):
        if mode not in {"hybrid", "sparse", "dense"}:
            raise ValueError(f"unknown retrieval mode: {mode}")
        self.s = settings
        self.mode = mode
        index_dir = settings.index_dir
        with chunks_path(index_dir).open(encoding="utf-8") as f:
            self.chunks = {c.chunk_id: c for c in (Chunk.model_validate(json.loads(x)) for x in f)}
        self.bm25 = BM25Index.load(bm25_path(index_dir)) if mode != "dense" else None
        if mode != "sparse":
            self.embedder = Embedder(settings.embedding_model)
            self.store = VectorStore(qdrant_path(index_dir), settings.collection)
        self.reranker = CrossEncoder(settings.reranker_model) if rerank else None

    def candidates(self, query: str) -> list[str]:
        rankings = []
        if self.bm25 is not None:
            rankings.append(self.bm25.search(query, self.s.sparse_k))
        if self.mode != "sparse":
            rankings.append(self.store.search(self.embedder.embed_query(query), self.s.dense_k))
        fused = reciprocal_rank_fusion(rankings)
        return [cid for cid, _ in fused[: self.s.fused_k]]

    def retrieve(self, query: str, k: int | None = None) -> list[ScoredChunk]:
        k = k or self.s.rerank_k
        ids = self.candidates(query)
        chunks = [self.chunks[i] for i in ids]
        if self.reranker is None or not chunks:
            return [ScoredChunk(c, 1.0 / (r + 1)) for r, c in enumerate(chunks[:k])]
        scores = self.reranker.predict([(query, c.render()) for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [ScoredChunk(c, float(s)) for c, s in ranked[:k]]
