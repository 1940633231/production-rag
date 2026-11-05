from __future__ import annotations

from pathlib import Path

from conftest import retrieval_sample, write_dataset
from rag_eval.config import EvaluationConfig
from rag_eval.contracts import EvaluationReport
from rag_eval.regression.baseline import (
    append_trend_csv,
    baseline_from_report,
    compare_reports,
)
from rag_eval.regression.gate import evaluate_gate
from rag_eval.registry import FakeAdapter
from rag_eval.report.json_report import write_json_report
from rag_eval.runner import run_evaluation


def _run(tmp_path: Path) -> EvaluationReport:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    config = EvaluationConfig(
        dataset=str(dataset_dir),
        adapter="rag_eval.registry:FakeAdapter",
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "reports"),
        k_values=[5],
    )
    return run_evaluation(config, adapter=FakeAdapter()).report


def test_metric_drop_exceeding_threshold_fails(tmp_path: Path) -> None:
    baseline_report = _run(tmp_path)
    current = baseline_report.model_copy(deep=True)
    current.metrics.values["recall@5"] = 0.5

    comparison = compare_reports(
        current,
        baseline_from_report(baseline_report),
        thresholds={"recall@5": 0.1},
    )

    assert not comparison.passed
    assert any("recall@5" in issue for issue in comparison.issues)


def test_missing_current_metric_fails_closed(tmp_path: Path) -> None:
    baseline_report = _run(tmp_path)
    current = baseline_report.model_copy(deep=True)
    del current.metrics.values["recall@5"]

    comparison = compare_reports(
        current,
        baseline_from_report(baseline_report),
        thresholds={"recall@5": 0.0},
    )

    assert not comparison.passed
    assert any("current metric is missing" in issue for issue in comparison.issues)


def test_dataset_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    baseline_report = _run(tmp_path)
    current = baseline_report.model_copy(deep=True)
    current.run_manifest.dataset_hash = "different"

    comparison = compare_reports(
        current,
        baseline_from_report(baseline_report),
        thresholds={"recall@5": 0.0},
    )

    assert not comparison.dataset_compatible
    assert not comparison.passed


def test_gate_exit_contract_and_trend_csv(tmp_path: Path) -> None:
    baseline_report = _run(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "current.json"
    write_json_report(baseline_path, baseline_report)
    write_json_report(report_path, baseline_report)
    config = EvaluationConfig(
        dataset=str(tmp_path / "dataset"),
        adapter="rag_eval.registry:FakeAdapter",
        output_dir=str(tmp_path / "reports"),
        k_values=[5],
        thresholds={"recall@5": 0.0},
    )

    result = evaluate_gate(config, baseline_path, report_path)
    trend_path = append_trend_csv(tmp_path / "trend.csv", baseline_report)

    assert result.passed
    assert trend_path.is_file()
    assert "recall@5" in trend_path.read_text(encoding="utf-8")


def test_gate_rejects_mismatched_execution_config(tmp_path: Path) -> None:
    baseline_report = _run(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "current.json"
    write_json_report(baseline_path, baseline_report)
    write_json_report(report_path, baseline_report)
    config = EvaluationConfig(
        dataset=str(tmp_path / "dataset"),
        adapter="rag_eval.registry:FakeAdapter",
        output_dir=str(tmp_path / "reports"),
        k_values=[1, 5],
        thresholds={"recall@5": 0.0},
    )

    result = evaluate_gate(config, baseline_path, report_path)

    assert not result.passed
    assert not result.config_compatible
