"""End-to-end evaluation: closed-book LLM vs. naive RAG vs. agentic RAG.

For every question we record
  - accuracy: predicted yes/no/maybe == gold label
  - retrieval: recall@k / MRR / nDCG@10 of the gold paper(s) in the evidence the system used
  - grounding: a judge model checks every claim. RAG systems are checked against the passages
    they cite; the closed-book baseline (which has nothing to cite) is checked against the gold
    abstract. A claim that fails is counted as a hallucination.

The judge is a separate, cheaper model, and grounding is re-checked in a fresh call here
rather than reusing the pipeline's own verifier output.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from biorag.agents import prompts
from biorag.agents.graph import Pipeline, PipelineConfig, closed_book
from biorag.agents.schemas import Answer, Verification
from biorag.data.documents import Chunk
from biorag.data.pubmedqa import QAExample
from biorag.eval.metrics import dedupe_docs, mean, ndcg_at_k, recall_at_k, reciprocal_rank
from biorag.llm import LLM, RefusalError
from biorag.retrieval.retriever import Retriever, ScoredChunk

SYSTEMS = ("closed_book", "naive_rag", "agentic_rag")


@dataclass
class Record:
    qid: str
    system: str
    gold: str
    predicted: str | None
    correct: bool
    n_claims: int = 0
    n_unsupported: int = 0
    recall_at_5: float | None = None
    mrr: float | None = None
    ndcg_at_10: float | None = None
    latency_s: float = 0.0
    error: str | None = None
    answer: dict | None = None
    evidence_pmids: list[str] = field(default_factory=list)


def check_claims(judge: LLM, answer: Answer, passages_for: Callable[[list[int]], list[str]]) -> int:
    """Return how many claims the judge marks unsupported."""
    if not answer.claims:
        return 0
    blocks = []
    for i, claim in enumerate(answer.claims):
        passages = passages_for(claim.citations)
        body = "\n".join(f"  - {p}" for p in passages) or "  (no valid citation)"
        blocks.append(f"Claim {i}: {claim.text}\nCited passages:\n{body}")
    result = judge.structured(prompts.VERIFIER, "\n\n".join(blocks), Verification)
    judged = {v.claim_index: v.supported for v in result.verdicts}
    # A claim the judge skipped counts as unsupported.
    return sum(1 for i in range(len(answer.claims)) if not judged.get(i, False))


def _retrieval_metrics(rec: Record, evidence: list[ScoredChunk], gold: set[str]) -> None:
    ranked = dedupe_docs([e.chunk.doc_id for e in evidence])
    rec.evidence_pmids = ranked
    rec.recall_at_5 = recall_at_k(ranked, gold, 5)
    rec.mrr = reciprocal_rank(ranked, gold)
    rec.ndcg_at_10 = ndcg_at_k(ranked, gold, 10)


def evaluate_example(
    ex: QAExample,
    system: str,
    llm: LLM,
    judge: LLM,
    retriever: Retriever,
    gold_chunks: dict[str, list[Chunk]],
) -> Record:
    rec = Record(qid=ex.qid, system=system, gold=ex.decision, predicted=None, correct=False)
    start = time.perf_counter()
    try:
        if system == "closed_book":
            answer = closed_book(llm, ex.question)
            gold_text = [c.render() for d in ex.gold for c in gold_chunks.get(d, [])]
            passages_for: Callable[[list[int]], list[str]] = lambda _cites: gold_text  # noqa: E731
        else:
            config = PipelineConfig(plan=system == "agentic_rag", verify=system == "agentic_rag")
            state = Pipeline(llm, judge, retriever, config).run(ex.question)
            answer, evidence = state["answer"], state["evidence"]
            _retrieval_metrics(rec, evidence, ex.gold)

            def passages_for(cites: list[int]) -> list[str]:
                return [evidence[c - 1].chunk.render() for c in cites if 1 <= c <= len(evidence)]

        rec.latency_s = time.perf_counter() - start
        rec.predicted = answer.decision
        rec.correct = answer.decision == ex.decision
        rec.answer = answer.model_dump()
        rec.n_claims = len(answer.claims)
        rec.n_unsupported = check_claims(judge, answer, passages_for)
    except RefusalError as e:
        rec.error = f"refusal: {e}"
    except Exception as e:  # keep the run going; failures are reported in the summary
        rec.error = f"{type(e).__name__}: {e}"
    return rec


def summarize(records: list[Record]) -> dict[str, dict[str, float]]:
    summary = {}
    for system in SYSTEMS:
        rs = [r for r in records if r.system == system]
        if not rs:
            continue
        ok = [r for r in rs if r.error is None]
        claims = sum(r.n_claims for r in ok)
        row = {
            "n": len(rs),
            "errors": len(rs) - len(ok),
            # Errors count as wrong answers so a fragile system cannot look better by failing.
            "accuracy": mean([float(r.correct) for r in rs]),
            "answers_with_hallucination": mean([float(r.n_unsupported > 0) for r in ok]),
            # Answers with no claims have nothing to verify; reported so they can't pass as
            # "hallucination-free".
            "answers_without_claims": mean([float(r.n_claims == 0) for r in ok]),
            "unsupported_per_100_claims": 100 * sum(r.n_unsupported for r in ok) / claims
            if claims
            else 0.0,
            "mean_latency_s": mean([r.latency_s for r in ok]),
        }
        if system != "closed_book":
            row["recall@5"] = mean([r.recall_at_5 for r in ok if r.recall_at_5 is not None])
            row["mrr"] = mean([r.mrr for r in ok if r.mrr is not None])
            row["ndcg@10"] = mean([r.ndcg_at_10 for r in ok if r.ndcg_at_10 is not None])
        summary[system] = row
    return summary


def to_markdown(summary: dict[str, dict[str, float]]) -> str:
    cols = ["n", "errors", "accuracy", "answers_with_hallucination", "answers_without_claims",
            "unsupported_per_100_claims", "recall@5", "mrr", "ndcg@10", "mean_latency_s"]
    lines = ["| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for system, row in summary.items():
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results(out_dir: Path, records: list[Record], meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    summary = summarize(records)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(summary) + "\n")
    return summary
