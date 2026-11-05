"""Public package surface for rag-eval."""

from rag_eval.contracts import (
    Citation,
    DatasetManifest,
    EvalSample,
    Evidence,
    Gold,
    GoldDocument,
    MetricSummary,
    Prediction,
    RunManifest,
    Span,
)

__all__ = [
    "Citation",
    "DatasetManifest",
    "EvalSample",
    "Evidence",
    "Gold",
    "GoldDocument",
    "MetricSummary",
    "Prediction",
    "RunManifest",
    "Span",
]

__version__ = "0.1.0"
