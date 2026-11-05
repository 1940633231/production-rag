"""Judge protocol and normalized judge output."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from rag_eval.contracts import EvalSample, Prediction, StrictModel


class JudgeResult(StrictModel):
    metric: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str = ""
    parse_status: str = "ok"
    model: str
    prompt_version: str


@runtime_checkable
class LLMJudge(Protocol):
    name: str

    def judge(self, sample: EvalSample, prediction: Prediction) -> JudgeResult:
        """Return a calibrated score or an explicit unknown/parse failure."""
