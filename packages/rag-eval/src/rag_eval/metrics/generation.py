"""Deterministic generation, refusal, citation, and coverage metrics."""

from __future__ import annotations

import re
from collections.abc import Iterable

from rag_eval.contracts import EvalSample, MetricSummary, Prediction
from rag_eval.metrics import safe_mean


def _prediction_map(predictions: Iterable[Prediction]) -> dict[str, Prediction]:
    result: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.sample_id in result:
            raise ValueError(f"duplicate prediction sample_id: {prediction.sample_id}")
        result[prediction.sample_id] = prediction
    return result


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def answer_contains_expected(answer: str | None, expected: list[str]) -> bool:
    if answer is None or not expected:
        return False
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    return any(
        normalize_answer(item) in normalized_answer
        for item in expected
        if normalize_answer(item)
    )


def _gold_document_ids(sample: EvalSample) -> set[str]:
    ids = {
        document.document_id
        for document in sample.gold.documents
        if document.relevance > 0
    }
    ids.update(
        chunk.document_id for chunk in sample.gold.chunks if chunk.relevance > 0
    )
    return ids


def _prediction_document_ids(prediction: Prediction) -> set[str]:
    ids = {document.document_id for document in prediction.documents}
    ids.update(chunk.document_id for chunk in prediction.chunks)
    return ids


def is_refusal(prediction: Prediction | None) -> bool:
    if prediction is None or prediction.status != "ok":
        return False
    behavior = prediction.metadata.get("behavior")
    if behavior == "refuse":
        return True
    return prediction.answer is None or not normalize_answer(prediction.answer)


def citation_is_valid(sample: EvalSample, prediction: Prediction | None) -> bool:
    if prediction is None or prediction.status != "ok":
        return False
    if not prediction.citations:
        return not bool(_gold_document_ids(sample))
    retrieved_documents = _prediction_document_ids(prediction)
    retrieved_chunks = {chunk.chunk_id for chunk in prediction.chunks}
    for citation in prediction.citations:
        if citation.document_id not in retrieved_documents:
            return False
        if citation.chunk_id is not None and citation.chunk_id not in retrieved_chunks:
            return False
    return True


def citation_coverage(sample: EvalSample, prediction: Prediction | None) -> float:
    expected = _gold_document_ids(sample)
    if not expected:
        return 0.0
    if prediction is None or prediction.status != "ok":
        return 0.0
    retrieved_documents = _prediction_document_ids(prediction)
    cited_documents = {
        citation.document_id
        for citation in prediction.citations
        if citation.document_id in retrieved_documents
    }
    return len(expected & cited_documents) / len(expected)


def evaluate_generation(
    samples: list[EvalSample],
    predictions: Iterable[Prediction],
) -> MetricSummary:
    if not samples:
        return MetricSummary()
    prediction_by_id = _prediction_map(predictions)
    expected_values: list[float] = []
    citation_validity: list[float] = []
    citation_coverages: list[float] = []
    refusal_correctness: list[float] = []
    completed = 0
    errors = 0
    skipped = 0

    for sample in samples:
        prediction = prediction_by_id.get(sample.id)
        if prediction is not None:
            if prediction.status == "ok":
                completed += 1
            elif prediction.status == "skipped":
                skipped += 1
            else:
                errors += 1

        expected = sample.gold.answers + sample.gold.values
        if expected:
            expected_values.append(
                1.0
                if answer_contains_expected(prediction.answer if prediction else None, expected)
                else 0.0
            )

        if sample.gold.behavior == "answer" and _gold_document_ids(sample):
            citation_validity.append(
                1.0 if citation_is_valid(sample, prediction) else 0.0
            )
            citation_coverages.append(citation_coverage(sample, prediction))

        if sample.gold.behavior == "refuse":
            refusal_correctness.append(1.0 if is_refusal(prediction) else 0.0)

    total = len(samples)
    values = {
        "expected_value_hit": safe_mean(expected_values),
        "citation_valid": safe_mean(citation_validity),
        "citation_coverage": safe_mean(citation_coverages),
        "refusal_correct": safe_mean(refusal_correctness),
        "coverage": completed / total,
        "error_rate": errors / total,
    }
    denominators = {
        "expected_value_hit": len(expected_values),
        "citation_valid": len(citation_validity),
        "citation_coverage": len(citation_coverages),
        "refusal_correct": len(refusal_correctness),
        "coverage": total,
        "error_rate": total,
    }
    counts = {
        "expected_samples": total,
        "completed_samples": completed,
        "error_samples": errors,
        "skipped_samples": skipped,
    }
    return MetricSummary(values=values, denominators=denominators, counts=counts)
