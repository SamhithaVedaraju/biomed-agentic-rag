"""Retrieval-only benchmark (no LLM calls, so it is free to run).

Compares sparse, dense and hybrid retrieval, with and without cross-encoder re-ranking. Run it
once per embedding model (each needs its own index) to compare embedders.
"""

from __future__ import annotations

import time

from biorag.data.pubmedqa import QAExample
from biorag.eval.metrics import dedupe_docs, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from biorag.retrieval.retriever import HybridRetriever


def bench(retriever: HybridRetriever, examples: list[QAExample], k: int = 10) -> dict[str, float]:
    recalls, rrs, ndcgs, latencies = [], [], [], []
    for ex in examples:
        start = time.perf_counter()
        hits = retriever.retrieve(ex.question, k=k * 3)  # chunks -> fewer unique docs
        latencies.append(time.perf_counter() - start)
        ranked = dedupe_docs([h.chunk.doc_id for h in hits])[:k]
        recalls.append(recall_at_k(ranked, ex.gold, 5))
        rrs.append(reciprocal_rank(ranked, ex.gold))
        ndcgs.append(ndcg_at_k(ranked, ex.gold, 10))
    latencies.sort()
    return {
        "recall@5": mean(recalls),
        "mrr@10": mean(rrs),
        "ndcg@10": mean(ndcgs),
        "p50_latency_ms": 1000 * latencies[len(latencies) // 2],
        "p95_latency_ms": 1000 * latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0,
    }
