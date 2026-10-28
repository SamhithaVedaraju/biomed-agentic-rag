from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel


class Document(BaseModel):
    doc_id: str  # PubMed ID
    title: str = ""
    text: str
    source: str = "pubmed"


class Chunk(BaseModel):
    chunk_id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    text: str

    def render(self) -> str:
        """Text shown to the embedding model, BM25 and the LLM."""
        return f"{self.title}\n{self.text}" if self.title else self.text


def write_jsonl(path: Path, docs: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.model_dump_json() + "\n")
            n += 1
    return n


def read_documents(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Document.model_validate(json.loads(line))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def chunk_document(doc: Document, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split on sentence boundaries into chunks of at most ~`size` characters.

    Consecutive chunks share up to `overlap` characters of trailing sentences so a fact that
    straddles a boundary is still retrievable from one chunk.
    """
    sentences = [s for s in _SENTENCE_END.split(doc.text.strip()) if s]
    chunks: list[list[str]] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + len(sentence) + 1 > size:
            chunks.append(current)
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(current)

    return [
        Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, title=doc.title, text=" ".join(c))
        for i, c in enumerate(chunks)
    ]
