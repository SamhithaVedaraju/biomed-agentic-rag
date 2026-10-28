"""PubMedQA (Jin et al., 2019) loader: questions for evaluation, abstracts for the corpus.

The labeled split has 1,000 expert-annotated yes/no/maybe questions, each tied to the abstract
it was written from. We put those abstracts in the corpus together with distractor abstracts
from the unlabeled split, so retrieval has to find the right paper among thousands.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from datasets import load_dataset
from pydantic import BaseModel

from biorag.data.documents import Document

DATASET = "qiaojin/PubMedQA"

Decision = Literal["yes", "no", "maybe"]


class QAExample(BaseModel):
    qid: str
    question: str
    gold_doc_id: str
    gold_doc_ids: list[str] = []  # BioASQ lists several relevant papers per question
    decision: Decision
    long_answer: str

    @property
    def gold(self) -> set[str]:
        return set(self.gold_doc_ids) or {self.gold_doc_id}


def _to_document(row: dict) -> Document:
    ctx = row["context"]
    parts = [
        f"{label.title()}: {sentence}" if label else sentence
        for label, sentence in zip(ctx["labels"], ctx["contexts"], strict=False)
    ]
    return Document(doc_id=str(row["pubid"]), text=" ".join(parts), source="pubmedqa")


def load_questions(limit: int | None = None) -> list[QAExample]:
    ds = load_dataset(DATASET, "pqa_labeled", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [
        QAExample(
            qid=str(r["pubid"]),
            question=r["question"],
            gold_doc_id=str(r["pubid"]),
            decision=r["final_decision"],
            long_answer=r["long_answer"],
        )
        for r in ds
    ]


def iter_corpus(n_distractors: int = 9000) -> Iterator[Document]:
    """All labeled abstracts, then `n_distractors` abstracts from the unlabeled split."""
    seen: set[str] = set()
    for row in load_dataset(DATASET, "pqa_labeled", split="train"):
        doc = _to_document(row)
        seen.add(doc.doc_id)
        yield doc
    if n_distractors <= 0:
        return
    unlabeled = load_dataset(DATASET, "pqa_unlabeled", split="train", streaming=True)
    added = 0
    for row in unlabeled:
        doc = _to_document(row)
        if doc.doc_id in seen:
            continue
        yield doc
        added += 1
        if added >= n_distractors:
            break
