from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

load_dotenv()

from biorag.config import get_settings  # noqa: E402

app = typer.Typer(add_completion=False, help="Agentic RAG over biomedical literature.")


def _ncbi() -> tuple[str, str | None]:
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise typer.BadParameter("Set NCBI_EMAIL in .env (NCBI requires a contact email).")
    return email, os.environ.get("NCBI_API_KEY") or None


def _prevent_sleep() -> None:
    """Keep Windows from idle-sleeping while this process runs (long local-model evals).
    Released automatically when the process exits; a closed lid still sleeps."""
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def _examples(dataset: str, bioasq_path: Path | None, limit: int | None):
    if dataset == "pubmedqa":
        from biorag.data.pubmedqa import load_questions

        return load_questions(limit)
    if bioasq_path is None:
        raise typer.BadParameter("--bioasq-path is required for --dataset bioasq")
    from biorag.data.bioasq import load_bioasq, to_examples

    return to_examples(load_bioasq(bioasq_path, yesno_only=True, limit=limit))


@app.command()
def prepare_pubmedqa(distractors: int = 9000) -> None:
    """Build the corpus from PubMedQA: 1,000 labeled abstracts + N distractor abstracts."""
    from biorag.data.documents import write_jsonl
    from biorag.data.pubmedqa import iter_corpus

    s = get_settings()
    n = write_jsonl(s.corpus_path, iter_corpus(distractors))
    typer.echo(f"Wrote {n} documents to {s.corpus_path}")


@app.command()
def prepare_bioasq(
    path: Path,
    limit: Annotated[int | None, typer.Option(help="Max questions (all types)")] = None,
    pubmedqa_distractors: int = 5000,
) -> None:
    """Build the corpus for a BioASQ Task B file: its gold abstracts from PubMed + distractors."""
    from biorag.data.bioasq import gold_pmids, load_bioasq
    from biorag.data.documents import write_jsonl
    from biorag.data.pubmed import fetch_abstracts
    from biorag.data.pubmedqa import iter_corpus

    s = get_settings()
    email, api_key = _ncbi()
    pmids = gold_pmids(load_bioasq(path, yesno_only=False, limit=limit))
    typer.echo(f"Fetching {len(pmids)} gold abstracts from PubMed...")
    gold = list(fetch_abstracts(None, 0, email, api_key, pmids=pmids))
    seen = {d.doc_id for d in gold}
    distractors = (d for d in iter_corpus(pubmedqa_distractors) if d.doc_id not in seen)
    n = write_jsonl(s.corpus_path, [*gold, *distractors])
    typer.echo(f"Wrote {n} documents ({len(gold)} gold) to {s.corpus_path}")


@app.command()
def fetch_pubmed(term: str, retmax: int = 1000) -> None:
    """Build a corpus from a PubMed search, e.g. 'metformin AND cancer'."""
    from biorag.data.documents import write_jsonl
    from biorag.data.pubmed import fetch_abstracts

    s = get_settings()
    email, api_key = _ncbi()
    n = write_jsonl(s.corpus_path, fetch_abstracts(term, retmax, email, api_key))
    typer.echo(f"Wrote {n} documents to {s.corpus_path}")


@app.command()
def build_index() -> None:
    """Chunk the corpus and build the BM25 + vector indexes."""
    from biorag.retrieval.indexing import build_index as build

    n = build(get_settings())
    typer.echo(f"Indexed {n} chunks")


def _pipeline(plan: bool = True, verify: bool = True):
    from biorag.agents.graph import Pipeline, PipelineConfig
    from biorag.llm import make_llm
    from biorag.retrieval.retriever import HybridRetriever

    s = get_settings()
    llm, judge = make_llm(s, "generator"), make_llm(s, "judge")
    retriever = HybridRetriever(s)
    config = PipelineConfig(plan=plan, verify=verify, max_subquestions=s.max_subquestions)
    return Pipeline(llm, judge, retriever, config), llm, judge, retriever


@app.command()
def ask(question: str) -> None:
    """Answer one question with the full agentic pipeline."""
    pipeline, *_ = _pipeline()
    state = pipeline.run(question)
    answer = state["answer"]
    typer.echo(f"\nDecision: {answer.decision}\n\n{answer.summary}\n")
    failed = {v.claim_index for v in state.get("verdicts", []) if not v.supported}
    for i, claim in enumerate(answer.claims):
        flag = "  [unverified]" if i in failed else ""
        typer.echo(f"- {claim.text} {claim.citations}{flag}")
    typer.echo("\nEvidence:")
    for i, e in enumerate(state["evidence"], 1):
        typer.echo(f"[{i}] PMID {e.chunk.doc_id}  {e.chunk.render()[:110]}...")


@app.command("eval")
def run_eval(
    dataset: Annotated[str, typer.Option(help="pubmedqa or bioasq")] = "pubmedqa",
    bioasq_path: Path | None = None,
    limit: int = 50,
    systems: Annotated[str, typer.Option(help="Comma-separated")] = (
        "closed_book,naive_rag,agentic_rag"
    ),
    out: Path = Path("artifacts/results"),
) -> None:
    """Compare closed-book, naive RAG and agentic RAG (with the claude backend this costs money)."""
    from biorag.eval.run import evaluate_example, to_markdown, write_results

    _prevent_sleep()
    _, llm, judge, retriever = _pipeline()
    gold_chunks = defaultdict(list)
    for chunk in retriever.chunks.values():
        gold_chunks[chunk.doc_id].append(chunk)

    examples = _examples(dataset, bioasq_path, limit)
    chosen = [x.strip() for x in systems.split(",")]
    records = []
    with typer.progressbar(length=len(examples) * len(chosen), label="Evaluating") as bar:
        for ex in examples:
            for system in chosen:
                records.append(evaluate_example(ex, system, llm, judge, retriever, gold_chunks))
                bar.update(1)

    s = get_settings()
    run_dir = out / f"{dataset}-{datetime.now():%Y%m%d-%H%M%S}"
    meta = {
        "dataset": dataset, "n_questions": len(examples), "backend": s.llm_backend,
        "generator": llm.model, "judge": judge.model,
        "embedding": s.embedding_model, "reranker": s.reranker_model,
        "tokens": {"generator": vars(llm.usage), "judge": vars(judge.usage)},
    }
    summary = write_results(run_dir, records, meta)
    typer.echo("\n" + to_markdown(summary))
    typer.echo(f"\nSaved to {run_dir}")


@app.command()
def bench_retrieval(
    dataset: str = "pubmedqa",
    bioasq_path: Path | None = None,
    limit: int = 500,
    out: Path = Path("artifacts/results"),
) -> None:
    """Free retrieval benchmark: sparse vs dense vs hybrid, with and without re-ranking."""
    from biorag.eval.retrieval_bench import bench
    from biorag.retrieval.retriever import HybridRetriever

    s = get_settings()
    examples = _examples(dataset, bioasq_path, limit)
    results = {}
    for mode in ("sparse", "dense", "hybrid"):
        for rerank in (False, True):
            name = f"{mode}{'+rerank' if rerank else ''}"
            typer.echo(f"Running {name}...")
            results[name] = bench(HybridRetriever(s, mode=mode, rerank=rerank), examples)
    report = json.dumps({"dataset": dataset, "embedding": s.embedding_model,
                         "reranker": s.reranker_model, "n": len(examples),
                         "results": results}, indent=2)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"retrieval-{dataset}-{s.embedding_model.split('/')[-1]}.json"
    path.write_text(report)
    typer.echo(report + f"\nSaved to {path}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("biorag.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
