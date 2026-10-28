"""Pipeline control-flow tests with a scripted LLM and an in-memory retriever (no API calls)."""

from biorag.agents.graph import Pipeline, PipelineConfig
from biorag.agents.schemas import Answer, Claim, ClaimVerdict, SubQuestions, Verification
from biorag.data.documents import Chunk
from biorag.llm import _strict_schema
from biorag.retrieval.retriever import ScoredChunk


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def structured(self, system, prompt, schema, effort="low"):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        assert isinstance(response, schema)
        return response


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=None):
        self.queries.append(query)
        chunk = Chunk(chunk_id="111#0", doc_id="111", title="", text="Drug X reduced mortality.")
        return [ScoredChunk(chunk, 0.9)]


def answer(*claims):
    claim_list = [Claim(text=c, citations=[1]) for c in claims]
    return Answer(decision="yes", summary="s", claims=claim_list)


def test_full_pipeline_revises_once_when_claims_fail():
    llm = ScriptedLLM([
        SubQuestions(subquestions=["drug X mortality"]),
        answer("Drug X reduced mortality.", "Drug X cures cancer."),
        answer("Drug X reduced mortality."),
    ])
    judge = ScriptedLLM([
        Verification(verdicts=[
            ClaimVerdict(claim_index=0, supported=True, reason="stated"),
            ClaimVerdict(claim_index=1, supported=False, reason="not in evidence"),
        ]),
        Verification(verdicts=[ClaimVerdict(claim_index=0, supported=True, reason="stated")]),
    ])
    retriever = FakeRetriever()
    state = Pipeline(llm, judge, retriever).run("Does drug X reduce mortality?")

    assert retriever.queries == ["Does drug X reduce mortality?", "drug X mortality"]
    assert len(state["evidence"]) == 1  # same chunk from both queries is merged
    assert state["revisions"] == 1
    assert [c.text for c in state["answer"].claims] == ["Drug X reduced mortality."]
    assert "Drug X cures cancer." in llm.prompts[-1]  # the revision prompt names the bad claim


def test_naive_pipeline_skips_planner_and_verifier():
    llm = ScriptedLLM([answer("Drug X reduced mortality.")])
    judge = ScriptedLLM([])
    state = Pipeline(llm, judge, FakeRetriever(), PipelineConfig(plan=False, verify=False)).run("q")
    assert state["answer"].decision == "yes"
    assert "verdicts" not in state


def test_strict_schema_closes_every_object():
    schema = _strict_schema(Answer)
    objects = [schema, *schema.get("$defs", {}).values()]
    assert all(o.get("additionalProperties") is False for o in objects if o.get("type") == "object")
