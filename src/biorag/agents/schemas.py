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
    # Field order is generation order: claims first, so the decision follows from the evidence
    # instead of being committed to up front (small models otherwise default to "maybe").
    claims: list[Claim] = Field(min_length=1)
    summary: str = Field(description="Two to four sentence answer for a clinician.")
    decision: Literal["yes", "no", "maybe"]


class ClaimVerdict(BaseModel):
    claim_index: int
    supported: bool
    reason: str


class Verification(BaseModel):
    verdicts: list[ClaimVerdict]
