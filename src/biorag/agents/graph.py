"""LangGraph pipeline: plan -> retrieve -> answer -> verify (-> answer again if claims fail).

                 +---------+     +----------+     +--------+     +--------+
    question --> | planner | --> | retrieve | --> | answer | --> | verify | --> result
                 +---------+     +----------+     +--------+     +--------+
                                                      ^              |
                                                      +-- revise ----+

The planner and verifier can be switched off, which gives the "naive RAG" baseline used in
the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from biorag.agents import prompts
from biorag.agents.schemas import Answer, ClaimVerdict, SubQuestions, Verification
from biorag.llm import LLM
from biorag.retrieval.retriever import Retriever, ScoredChunk


class State(TypedDict, total=False):
    question: str
    subquestions: list[str]
    evidence: list[ScoredChunk]
    answer: Answer
    verdicts: list[ClaimVerdict]
    revisions: int


@dataclass
class PipelineConfig:
    plan: bool = True
    verify: bool = True
    max_subquestions: int = 3
    per_query_k: int = 5
    max_evidence: int = 12
    max_revisions: int = 1


def format_evidence(evidence: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (PMID {e.chunk.doc_id}) {e.chunk.render()}" for i, e in enumerate(evidence, 1)
    )


def unsupported(state: State) -> list[ClaimVerdict]:
    return [v for v in state.get("verdicts", []) if not v.supported]


@dataclass
class Pipeline:
    llm: LLM
    judge: LLM
    retriever: Retriever
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        self.graph = self._build()

    def run(self, question: str) -> State:
        return self.graph.invoke({"question": question, "revisions": 0})

    # --- nodes -------------------------------------------------------------------------

    def plan(self, state: State) -> State:
        result = self.llm.structured(
            prompts.PLANNER.format(max_n=self.config.max_subquestions),
            state["question"],
            SubQuestions,
        )
        return {"subquestions": result.subquestions[: self.config.max_subquestions]}

    def retrieve(self, state: State) -> State:
        queries = [state["question"], *state.get("subquestions", [])]
        best: dict[str, ScoredChunk] = {}
        for query in dict.fromkeys(queries):  # de-duplicate, keep order
            for hit in self.retriever.retrieve(query, self.config.per_query_k):
                prev = best.get(hit.chunk.chunk_id)
                if prev is None or hit.score > prev.score:
                    best[hit.chunk.chunk_id] = hit
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return {"evidence": ranked[: self.config.max_evidence]}

    def answer(self, state: State) -> State:
        prompt = f"Question: {state['question']}\n\nEvidence:\n{format_evidence(state['evidence'])}"
        problems = unsupported(state)
        if problems and "answer" in state:
            claims = state["answer"].claims
            listed = "\n".join(
                f"- \"{claims[v.claim_index].text}\": {v.reason}"
                for v in problems
                if 0 <= v.claim_index < len(claims)
            )
            prompt += "\n\n" + prompts.REVISION.format(problems=listed)
        answer = self.llm.structured(prompts.ANSWERER, prompt, Answer, effort="medium")
        return {"answer": answer, "revisions": state.get("revisions", 0) + (1 if problems else 0)}

    def verify(self, state: State) -> State:
        evidence = state["evidence"]
        claims = state["answer"].claims
        if not claims:
            return {"verdicts": []}
        lines = []
        for i, claim in enumerate(claims):
            cited = [evidence[c - 1] for c in claim.citations if 1 <= c <= len(evidence)]
            passages = "\n".join(f"  - {e.chunk.render()}" for e in cited)
            passages = passages or "  (no valid citation)"
            lines.append(f"Claim {i}: {claim.text}\nCited passages:\n{passages}")
        result = self.judge.structured(prompts.VERIFIER, "\n\n".join(lines), Verification)
        return {"verdicts": result.verdicts}

    def route_after_verify(self, state: State) -> str:
        if unsupported(state) and state.get("revisions", 0) < self.config.max_revisions:
            return "answer"
        return END

    # --- wiring ------------------------------------------------------------------------

    def _build(self):
        g = StateGraph(State)
        g.add_node("retrieve", self.retrieve)
        g.add_node("answer", self.answer)
        if self.config.plan:
            g.add_node("plan", self.plan)
            g.add_edge(START, "plan")
            g.add_edge("plan", "retrieve")
        else:
            g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "answer")
        if self.config.verify:
            g.add_node("verify", self.verify)
            g.add_edge("answer", "verify")
            g.add_conditional_edges("verify", self.route_after_verify, ["answer", END])
        else:
            g.add_edge("answer", END)
        return g.compile()


def closed_book(llm: LLM, question: str) -> Answer:
    """No-retrieval baseline."""
    return llm.structured(prompts.CLOSED_BOOK, question, Answer, effort="medium")
