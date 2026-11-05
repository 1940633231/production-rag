"""Command line interface for rag-eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rag_eval.config import load_eval_config
from rag_eval.contracts import DatasetValidationError, EvaluationReport
from rag_eval.regression.baseline import (
    append_trend_csv,
    baseline_from_report,
    compare_reports,
)
from rag_eval.regression.gate import evaluate_gate
from rag_eval.report.json_report import load_report_payload
from rag_eval.runner import run_evaluation


def _json_dump(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a dataset directory")
    validate.add_argument("dataset_dir")
    validate.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="run an evaluation configuration")
    run.add_argument("--config", required=True)
    run.add_argument("--json", action="store_true")
    run.add_argument("--save-baseline")

    compare = subparsers.add_parser("compare", help="compare two reports")
    compare.add_argument("current")
    compare.add_argument("baseline")
    compare.add_argument("--max-drop", type=float, default=0.0)
    compare.add_argument("--json", action="store_true")

    gate = subparsers.add_parser("gate", help="apply configured regression thresholds")
    gate.add_argument("--config", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--report", required=True)
    gate.add_argument("--json", action="store_true")

    trend = subparsers.add_parser("trend", help="append a report to a trend CSV")
    trend.add_argument("--report", required=True)
    trend.add_argument("--csv", required=True)
    return parser


def _handle_validate(args: argparse.Namespace) -> int:
    from rag_eval.contracts import validate_dataset_dir

    dataset, errors = validate_dataset_dir(args.dataset_dir)
    if args.json:
        payload: dict[str, Any] = {"valid": not errors, "errors": errors}
        if dataset is not None:
            payload["dataset"] = {
                "dataset_id": dataset.manifest.dataset_id,
                "version": dataset.manifest.version,
                "task": dataset.manifest.task,
                "samples": len(dataset.samples),
                "content_hash": dataset.content_hash,
            }
        print(_json_dump(payload))
    elif errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
    else:
        assert dataset is not None
        print(
            f"VALID {dataset.manifest.dataset_id}@{dataset.manifest.version} "
            f"task={dataset.manifest.task} samples={len(dataset.samples)} "
            f"hash={dataset.content_hash}"
        )
    return 1 if errors else 0


def _handle_run(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    outcome = run_evaluation(config)
    if args.save_baseline:
        baseline_path = Path(args.save_baseline)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            baseline_from_report(outcome.report).model_dump_json(indent=2),
            encoding="utf-8",
        )
    if args.json:
        print(
            _json_dump(
                {
                    "failed": outcome.failed,
                    "failure_reasons": outcome.failure_reasons,
                    "run_directory": str(outcome.run_directory),
                    "report": outcome.report.model_dump(mode="json"),
                }
            )
        )
    else:
        summary = outcome.report.summary
        print(f"run_id: {outcome.report.run_manifest.run_id}")
        print(f"report: {outcome.run_directory / 'report.json'}")
        print(f"markdown: {outcome.run_directory / 'report.md'}")
        print(
            f"coverage: {summary.coverage:.4f} "
            f"errors: {summary.error_samples} skipped: {summary.skipped_samples}"
        )
        for name in sorted(outcome.report.metrics.values):
            print(f"{name}: {outcome.report.metrics.values[name]:.6f}")
        for reason in outcome.failure_reasons:
            print(f"FAIL: {reason}", file=sys.stderr)
    return 1 if outcome.failed else 0


def _handle_compare(args: argparse.Namespace) -> int:
    current = EvaluationReport.model_validate(load_report_payload(args.current))
    baseline = load_report_payload(args.baseline)
    thresholds = {
        metric: args.max_drop for metric in baseline_metric_names(baseline)
    } if args.max_drop > 0 else {}
    comparison = compare_reports(current, baseline, thresholds=thresholds)
    if args.json:
        print(_json_dump(comparison))
    else:
        for delta in comparison.metric_deltas:
            print(
                f"{delta.metric}: {delta.current:.6f} "
                f"(baseline {delta.baseline:.6f}, delta {delta.delta:+.6f})"
            )
        for issue in comparison.issues:
            print(f"FAIL: {issue}", file=sys.stderr)
        print("PASS" if comparison.passed else "FAIL")
    return 0 if comparison.passed else 1


def baseline_metric_names(payload: dict[str, Any]) -> list[str]:
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        values = payload["metrics"].get("values", payload["metrics"])
        if isinstance(values, dict):
            return [
                name
                for name in values
                if not name.startswith("latency_") and not name.startswith("cost_")
            ]
    return []


def _handle_gate(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    result = evaluate_gate(config, args.baseline, args.report)
    if args.json:
        print(_json_dump(result))
    else:
        for issue in result.issues:
            print(f"FAIL: {issue}", file=sys.stderr)
        print("PASS" if result.passed else "FAIL")
    return 0 if result.passed else 1


def _handle_trend(args: argparse.Namespace) -> int:
    report = EvaluationReport.model_validate(load_report_payload(args.report))
    path = append_trend_csv(args.csv, report)
    print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _handle_validate(args)
        if args.command == "run":
            return _handle_run(args)
        if args.command == "compare":
            return _handle_compare(args)
        if args.command == "gate":
            return _handle_gate(args)
        if args.command == "trend":
            return _handle_trend(args)
        parser.error(f"unknown command: {args.command}")
    except (DatasetValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
