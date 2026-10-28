from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from biorag.agents.graph import Pipeline, PipelineConfig
from biorag.agents.schemas import Answer, ClaimVerdict
from biorag.config import get_settings
from biorag.llm import make_llm
from biorag.retrieval.retriever import HybridRetriever

load_dotenv()
_state: dict[str, Pipeline] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    _state["pipeline"] = Pipeline(
        make_llm(s, "generator"),
        make_llm(s, "judge"),
        HybridRetriever(s),
        PipelineConfig(max_subquestions=s.max_subquestions),
    )
    yield


app = FastAPI(title="biorag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class EvidenceOut(BaseModel):
    pmid: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: Answer
    subquestions: list[str]
    verdicts: list[ClaimVerdict]
    evidence: list[EvidenceOut]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    state = _state["pipeline"].run(req.question)
    return AskResponse(
        answer=state["answer"],
        subquestions=state.get("subquestions", []),
        verdicts=state.get("verdicts", []),
        evidence=[
            EvidenceOut(pmid=e.chunk.doc_id, text=e.chunk.render(), score=e.score)
            for e in state["evidence"]
        ],
    )
