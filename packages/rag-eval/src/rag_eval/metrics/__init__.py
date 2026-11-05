"""Metric implementations and shared aggregation helpers."""

from rag_eval.contracts import MetricSummary


def merge_metric_summaries(*summaries: MetricSummary) -> MetricSummary:
    values: dict[str, float] = {}
    denominators: dict[str, int] = {}
    counts: dict[str, int] = {}
    for summary in summaries:
        values.update(summary.values)
        denominators.update(summary.denominators)
        counts.update(summary.counts)
    return MetricSummary(values=values, denominators=denominators, counts=counts)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / len(values)))
