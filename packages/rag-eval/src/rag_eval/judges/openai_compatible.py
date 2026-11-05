"""Minimal OpenAI-compatible judge boundary.

The core package intentionally has no HTTP dependency. Applications can inject
a callable that performs their own request handling.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_eval.contracts import EvalSample, Prediction
from rag_eval.judges.base import JudgeResult


class OpenAICompatibleJudge:
    def __init__(
        self,
        *,
        model: str,
        prompt_version: str,
        invoke: Callable[[str], str],
        name: str = "openai-compatible",
    ):
        if not model or not prompt_version:
            raise ValueError("model and prompt_version are required")
        self.name = name
        self.model = model
        self.prompt_version = prompt_version
        self._invoke = invoke

    def judge(self, sample: EvalSample, prediction: Prediction) -> JudgeResult:
        prompt = (
            "Return a JSON object with a score between 0 and 1 and a rationale. "
            f"Question: {sample.query}\nAnswer: {prediction.answer or ''}"
        )
        try:
            response = self._invoke(prompt)
            return JudgeResult(
                metric="answer_correctness",
                score=None,
                rationale=response,
                parse_status="unparsed",
                model=self.model,
                prompt_version=self.prompt_version,
            )
        except Exception as exc:
            return JudgeResult(
                metric="answer_correctness",
                score=None,
                rationale=str(exc),
                parse_status="call_failed",
                model=self.model,
                prompt_version=self.prompt_version,
            )
