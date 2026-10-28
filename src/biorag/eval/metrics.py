"""Retrieval and answer metrics. Retrieval is scored at the document (PMID) level."""

from __future__ import annotations

import math
from collections.abc import Sequence


def dedupe_docs(chunk_doc_ids: Sequence[str]) -> list[str]:
    """Collapse a ranked chunk list to a ranked document list (first occurrence wins)."""
    return list(dict.fromkeys(chunk_doc_ids))


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1) if doc in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
