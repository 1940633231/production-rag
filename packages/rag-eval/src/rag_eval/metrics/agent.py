"""Agent workflow metrics derived from prediction metadata and evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable

from rag_eval.contracts import EvalSample, MetricSummary, Prediction
from rag_eval.metrics.generation import (
    _gold_document_ids,
    _prediction_document_ids,
    citation_coverage,
)
from rag_eval.metrics.retrieval import evidence_hit_at_k


def _prediction_map(predictions: Iterable[Prediction]) -> dict[str, Prediction]:
    result: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.sample_id in result:
            raise ValueError(f"duplicate prediction sample_id: {prediction.sample_id}")
        result[prediction.sample_id] = prediction
    return result


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def evaluate_agent(
    samples: list[EvalSample],
    predictions: Iterable[Prediction],
    evidence_k: int = 10,
) -> MetricSummary:
    if not samples:
        return MetricSummary()
    prediction_by_id = _prediction_map(predictions)
    retry_count = 0
    rewrite_sources = 0
    retry_samples = 0
    human_review_samples = 0
    evidence_hits: list[float] = []
    citation_coverages: list[float] = []
    foreign_evidence = 0
    evidence_total = 0
    foreign_citations = 0
    citation_total = 0
    latencies: list[float] = []
    costs: list[float] = []
    token_estimate = 0

    for sample in samples:
        prediction = prediction_by_id.get(sample.id)
        evidence_hits.append(evidence_hit_at_k(sample, prediction, evidence_k))
        citation_coverages.append(citation_coverage(sample, prediction))
        if prediction is None:
            continue
        metadata = prediction.metadata
        sample_retries = int(metadata.get("retry_count", 0) or 0)
        retry_count += sample_retries
        if sample_retries > 0:
            retry_samples += 1
        rewrite_sources += int(metadata.get("rewrite_sources", 0) or 0)
        if metadata.get("behavior") == "human_review":
            human_review_samples += 1
        if prediction.latency_ms is not None:
            latencies.append(prediction.latency_ms)
        cost = prediction.usage.get("cost", metadata.get("cost"))
        if isinstance(cost, (int, float)):
            costs.append(float(cost))
        tokens = prediction.usage.get(
            "total_tokens",
            prediction.usage.get("token_count", metadata.get("token_count", 0)),
        )
        if isinstance(tokens, (int, float)):
            token_estimate += int(tokens)

        relevant = _gold_document_ids(sample)
        for item in prediction.evidence:
            evidence_total += 1
            if relevant and item.document_id not in relevant:
                foreign_evidence += 1
        retrieved = _prediction_document_ids(prediction)
        for citation in prediction.citations:
            citation_total += 1
            is_foreign = (
                (relevant and citation.document_id not in relevant)
                or citation.document_id not in retrieved
            )
            if is_foreign:
                foreign_citations += 1

    total = len(samples)
    values = {
        "retry_count": float(retry_count),
        "retry_rate": retry_samples / total,
        "rewrite_sources": float(rewrite_sources),
        "evidence_hit_final": sum(evidence_hits) / total,
        "citation_coverage": sum(citation_coverages) / total,
        "foreign_evidence_rate": (
            foreign_evidence / evidence_total if evidence_total else 0.0
        ),
        "foreign_citation_rate": (
            foreign_citations / citation_total if citation_total else 0.0
        ),
        "human_review_rate": human_review_samples / total,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "token_estimate": float(token_estimate),
        "cost_estimate": float(sum(costs)),
    }
    denominators = {
        "retry_count": total,
        "retry_rate": total,
        "rewrite_sources": total,
        "evidence_hit_final": total,
        "citation_coverage": total,
        "foreign_evidence_rate": evidence_total,
        "foreign_citation_rate": citation_total,
        "human_review_rate": total,
        "latency_p50_ms": len(latencies),
        "latency_p95_ms": len(latencies),
        "token_estimate": total,
        "cost_estimate": total,
    }
    counts = {
        "retry_samples": retry_samples,
        "human_review_samples": human_review_samples,
        "evidence_items": evidence_total,
        "citation_items": citation_total,
    }
    return MetricSummary(values=values, denominators=denominators, counts=counts)
