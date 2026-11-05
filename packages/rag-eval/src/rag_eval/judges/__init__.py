"""Optional judge interfaces kept outside deterministic metric calculation."""

from rag_eval.judges.base import JudgeResult, LLMJudge
from rag_eval.judges.openai_compatible import OpenAICompatibleJudge

__all__ = ["JudgeResult", "LLMJudge", "OpenAICompatibleJudge"]
