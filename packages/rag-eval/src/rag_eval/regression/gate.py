"""Configuration-driven regression gate."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from rag_eval.config import EvaluationConfig
from rag_eval.contracts import EvaluationReport, StrictModel
from rag_eval.regression.baseline import (
    ComparisonResult,
    compare_reports,
    parse_baseline,
)
from rag_eval.report.json_report import load_report_payload


class GateResult(StrictModel):
    passed: bool
    dataset_hash_matches: bool
    config_compatible: bool
    complete: bool
    thresholds_defined: bool
    comparison: ComparisonResult
    issues: list[str] = Field(default_factory=list)


def evaluate_gate(
    config: EvaluationConfig,
    baseline_path: str | Path,
    report_path: str | Path,
) -> GateResult:
    baseline_payload = load_report_payload(baseline_path)
    current_payload = load_report_payload(report_path)
    current = EvaluationReport.model_validate(current_payload)
    baseline = parse_baseline(baseline_payload)
    comparison = compare_reports(
        current,
        baseline_payload,
        thresholds=config.thresholds,
        min_coverage=config.min_coverage,
        allow_missing_metrics=config.allow_missing_metrics,
    )
    issues = list(comparison.issues)
    gate_config_compatible = (
        config.config_hash() == baseline.compatibility.config_hash
        and config.model == baseline.compatibility.model
        and config.prompt_version == baseline.compatibility.prompt_version
        and config.seed == baseline.compatibility.seed
    )
    if not gate_config_compatible:
        issues.append("gate configuration does not match baseline execution configuration")
    if current.summary.coverage < config.min_coverage:
        issues.append(
            f"gate coverage {current.summary.coverage:.4f} is below "
            f"min_coverage {config.min_coverage:.4f}"
        )
    if current.summary.error_samples > config.max_errors:
        issues.append(
            f"gate error samples {current.summary.error_samples} exceed "
            f"max_errors {config.max_errors}"
        )
    thresholds_defined = bool(config.thresholds)
    return GateResult(
        passed=not issues,
        dataset_hash_matches=comparison.dataset_compatible,
        config_compatible=comparison.config_compatible and gate_config_compatible,
        complete=comparison.run_complete,
        thresholds_defined=thresholds_defined,
        comparison=comparison,
        issues=issues,
    )
