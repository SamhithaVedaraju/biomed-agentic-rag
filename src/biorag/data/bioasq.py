"""BioASQ Task B loader.

BioASQ data needs a (free) account, so it is not downloaded automatically. Download a Task B
training file from the participants area at https://participants-area.bioasq.org/ and pass
its path to the CLI. Only yes/no questions are used for answer accuracy; every question
type is usable for retrieval metrics because each lists its relevant PubMed documents.
"""

from __future__ import annotations

import json
from pathlib import Path

from biorag.data.pubmedqa import QAExample


def pmid_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def load_bioasq(path: Path, yesno_only: bool = True, limit: int | None = None) -> list[dict]:
    """Return raw questions with `gold_doc_ids` added."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for q in data["questions"]:
        if yesno_only and q.get("type") != "yesno":
            continue
        q["gold_doc_ids"] = [pmid_from_url(u) for u in q.get("documents", [])]
        out.append(q)
        if limit and len(out) >= limit:
            break
    return out


def to_examples(questions: list[dict]) -> list[QAExample]:
    """Yes/no questions as QAExamples (first gold document kept for the single-gold field)."""
    examples = []
    for q in questions:
        exact = q.get("exact_answer")
        if q.get("type") != "yesno" or exact not in ("yes", "no") or not q["gold_doc_ids"]:
            continue
        ideal = q.get("ideal_answer") or [""]
        examples.append(
            QAExample(
                qid=q["id"],
                question=q["body"],
                gold_doc_id=q["gold_doc_ids"][0],
                gold_doc_ids=q["gold_doc_ids"],
                decision=exact,
                long_answer=ideal[0] if isinstance(ideal, list) else ideal,
            )
        )
    return examples


def gold_pmids(questions: list[dict]) -> list[str]:
    return sorted({pmid for q in questions for pmid in q["gold_doc_ids"]})
