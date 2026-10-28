from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubQuestions(BaseModel):
    subquestions: list[str] = Field(
        description="Focused search queries that together cover the question. One is fine."
    )


class Claim(BaseModel):
    text: str = Field(description="One factual statement supporting the answer.")
    citations: list[int] = Field(description="Evidence numbers that directly support the claim.")


class Answer(BaseModel):
    decision: Literal["yes", "no", "maybe"]
    summary: str = Field(description="Two to four sentence answer for a clinician.")
    claims: list[Claim]


class ClaimVerdict(BaseModel):
    claim_index: int
    supported: bool
    reason: str


class Verification(BaseModel):
    verdicts: list[ClaimVerdict]
