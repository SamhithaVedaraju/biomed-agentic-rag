from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were "
    "which with this these those than do does did not".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str]):
        self.ids = ids
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        top = scores.argsort()[::-1][:k]
        return [self.ids[i] for i in top if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> BM25Index:
        with path.open("rb") as f:
            return pickle.load(f)
