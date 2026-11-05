"""Baseline comparison and evaluation gates."""

from rag_eval.regression.baseline import (
    Baseline,
    ComparisonResult,
    append_trend_csv,
    baseline_from_report,
    compare_reports,
)
from rag_eval.regression.gate import GateResult, evaluate_gate

__all__ = [
    "Baseline",
    "ComparisonResult",
    "GateResult",
    "append_trend_csv",
    "baseline_from_report",
    "compare_reports",
    "evaluate_gate",
]
