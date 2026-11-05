"""Baseline schema, report comparison, and trend export."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from rag_eval.contracts import (
    EvaluationReport,
    MetricSummary,
    RunSummary,
    StrictModel,
)


class BaselineCompatibility(StrictModel):
    adapter: str
    model: str | None = None
    prompt_version: str | None = None
    seed: int
    config_hash: str


class Baseline(StrictModel):
    schema_version: str = "1"
    baseline_id: str
    created_at: datetime
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    compatibility: BaselineCompatibility
    metrics: MetricSummary
    summary: RunSummary


class MetricDelta(StrictModel):
    metric: str
    baseline: float
    current: float
    delta: float
    max_drop: float
    passed: bool


class ComparisonResult(StrictModel):
    dataset_compatible: bool
    adapter_compatible: bool
    model_compatible: bool
    prompt_compatible: bool
    config_compatible: bool
    run_complete: bool
    metric_deltas: list[MetricDelta] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    passed: bool


def baseline_from_report(report: EvaluationReport | dict[str, Any]) -> Baseline:
    if not isinstance(report, EvaluationReport):
        report = EvaluationReport.model_validate(report)
    manifest = report.run_manifest
    return Baseline(
        baseline_id=manifest.run_id,
        created_at=manifest.ended_at,
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        dataset_hash=manifest.dataset_hash,
        compatibility=BaselineCompatibility(
            adapter=manifest.adapter,
            model=manifest.model,
            prompt_version=manifest.prompt_version,
            seed=manifest.seed,
            config_hash=manifest.config_hash,
        ),
        metrics=report.metrics,
        summary=report.summary,
    )


def parse_baseline(payload: Baseline | EvaluationReport | dict[str, Any]) -> Baseline:
    if isinstance(payload, Baseline):
        return payload
    if isinstance(payload, EvaluationReport):
        return baseline_from_report(payload)
    if "run_manifest" in payload:
        return baseline_from_report(payload)
    return Baseline.model_validate(payload)


def _advisory_metric(name: str) -> bool:
    return name.startswith("latency_") or name.startswith("cost_")


def compare_reports(
    current: EvaluationReport | dict[str, Any],
    baseline: Baseline | EvaluationReport | dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
    min_coverage: float = 1.0,
    allow_missing_metrics: bool = False,
) -> ComparisonResult:
    current_report = (
        current
        if isinstance(current, EvaluationReport)
        else EvaluationReport.model_validate(current)
    )
    baseline_value = parse_baseline(baseline)
    thresholds = thresholds or {}
    issues: list[str] = []
    current_manifest = current_report.run_manifest
    baseline_manifest = baseline_value

    dataset_compatible = (
        current_manifest.dataset_id == baseline_manifest.dataset_id
        and current_manifest.dataset_version == baseline_manifest.dataset_version
        and current_manifest.dataset_hash == baseline_manifest.dataset_hash
    )
    if not dataset_compatible:
        issues.append("dataset id, version, or hash does not match baseline")

    adapter_compatible = (
        current_manifest.adapter == baseline_manifest.compatibility.adapter
    )
    if not adapter_compatible:
        issues.append("adapter does not match baseline")
    model_compatible = current_manifest.model == baseline_manifest.compatibility.model
    if not model_compatible:
        issues.append("model does not match baseline")
    prompt_compatible = (
        current_manifest.prompt_version
        == baseline_manifest.compatibility.prompt_version
    )
    if not prompt_compatible:
        issues.append("prompt version does not match baseline")
    config_compatible = (
        current_manifest.config_hash == baseline_manifest.compatibility.config_hash
    )
    if not config_compatible:
        issues.append("configuration hash does not match baseline")
    seed_compatible = current_manifest.seed == baseline_manifest.compatibility.seed
    if not seed_compatible:
        issues.append("seed does not match baseline")

    summary = current_report.summary
    run_complete = (
        summary.expected_samples > 0
        and summary.coverage >= min_coverage
        and summary.within_thresholds
    )
    if not run_complete:
        issues.append(
            f"current run is incomplete (coverage {summary.coverage:.4f}, "
            f"minimum {min_coverage:.4f})"
        )

    baseline_metrics = baseline_manifest.metrics.values
    if not baseline_metrics:
        issues.append("baseline contains no metrics")

    required_metrics = {
        name for name in baseline_metrics if not _advisory_metric(name)
    }
    required_metrics.update(thresholds)
    if allow_missing_metrics:
        required_metrics = set(thresholds)

    deltas: list[MetricDelta] = []
    for metric_name in sorted(required_metrics):
        max_drop = thresholds.get(metric_name, 0.0)
        if metric_name not in baseline_metrics:
            issues.append(f"baseline metric is missing: {metric_name}")
            continue
        if metric_name not in current_report.metrics.values:
            issues.append(f"current metric is missing: {metric_name}")
            continue
        baseline_score = baseline_metrics[metric_name]
        current_score = current_report.metrics.values[metric_name]
        delta = current_score - baseline_score
        passed = (-delta) <= max_drop + 1e-12
        deltas.append(
            MetricDelta(
                metric=metric_name,
                baseline=baseline_score,
                current=current_score,
                delta=delta,
                max_drop=max_drop,
                passed=passed,
            )
        )
        if not passed:
            issues.append(
                f"metric {metric_name} dropped {-delta:.6f}, "
                f"exceeding max drop {max_drop:.6f}"
            )

    return ComparisonResult(
        dataset_compatible=dataset_compatible,
        adapter_compatible=adapter_compatible,
        model_compatible=model_compatible,
        prompt_compatible=prompt_compatible,
        config_compatible=config_compatible and seed_compatible,
        run_complete=run_complete,
        metric_deltas=deltas,
        issues=issues,
        passed=not issues,
    )


def append_trend_csv(path: str | Path, report: EvaluationReport) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "ended_at",
        "dataset_id",
        "dataset_version",
        "dataset_hash",
        "adapter",
        "model",
        "prompt_version",
        "seed",
        "metric",
        "value",
        "denominator",
    ]
    existing = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        manifest = report.run_manifest
        for metric in sorted(report.metrics.values):
            writer.writerow(
                {
                    "run_id": manifest.run_id,
                    "ended_at": manifest.ended_at.isoformat(),
                    "dataset_id": manifest.dataset_id,
                    "dataset_version": manifest.dataset_version,
                    "dataset_hash": manifest.dataset_hash,
                    "adapter": manifest.adapter,
                    "model": manifest.model or "",
                    "prompt_version": manifest.prompt_version or "",
                    "seed": manifest.seed,
                    "metric": metric,
                    "value": report.metrics.values[metric],
                    "denominator": report.metrics.denominators.get(metric, 0),
                }
            )
    return output
