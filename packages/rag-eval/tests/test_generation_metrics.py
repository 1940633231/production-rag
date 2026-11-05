from __future__ import annotations

from rag_eval.contracts import EvalSample, Prediction
from rag_eval.metrics.generation import evaluate_generation


def test_generation_expected_value_citation_and_refusal() -> None:
    samples = [
        EvalSample.model_validate(
            {
                "id": "answer",
                "task": "generation",
                "query": "capital?",
                "gold": {
                    "documents": [{"document_id": "doc-1", "relevance": 1}],
                    "answers": ["Paris"],
                    "behavior": "answer",
                },
            }
        ),
        EvalSample.model_validate(
            {
                "id": "refuse",
                "task": "generation",
                "query": "unknown?",
                "gold": {"behavior": "refuse"},
            }
        ),
    ]
    predictions = [
        Prediction.model_validate(
            {
                "sample_id": "answer",
                "answer": "The capital is Paris.",
                "documents": [{"document_id": "doc-1"}],
                "citations": [{"document_id": "doc-1"}],
            }
        ),
        Prediction.model_validate(
            {
                "sample_id": "refuse",
                "answer": None,
                "metadata": {"behavior": "refuse"},
            }
        ),
    ]

    summary = evaluate_generation(samples, predictions)

    assert summary.values["expected_value_hit"] == 1.0
    assert summary.values["citation_valid"] == 1.0
    assert summary.values["citation_coverage"] == 1.0
    assert summary.values["refusal_correct"] == 1.0
    assert summary.values["coverage"] == 1.0
    assert summary.values["error_rate"] == 0.0


def test_failed_prediction_is_counted_in_error_rate() -> None:
    sample = EvalSample.model_validate(
        {"id": "q1", "task": "generation", "query": "query", "gold": {}}
    )
    prediction = Prediction(sample_id="q1", status="error", error="boom")

    summary = evaluate_generation([sample], [prediction])

    assert summary.values["coverage"] == 0.0
    assert summary.values["error_rate"] == 1.0
    assert summary.counts["error_samples"] == 1
