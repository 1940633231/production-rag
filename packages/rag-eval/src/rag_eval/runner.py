"""Evaluation runner orchestration."""

from __future__ import annotations

import subprocess
import threading
from queue import Empty, Queue
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_eval.cache import PredictionCache, prediction_cache_key
from rag_eval.config import EvaluationConfig
from rag_eval.contracts import (
    EvaluationReport,
    LoadedDataset,
    MetricSummary,
    Prediction,
    RunManifest,
    RunSummary,
    canonical_json,
    load_dataset,
    sha256_value,
)
from rag_eval.metrics import merge_metric_summaries
from rag_eval.metrics.agent import evaluate_agent
from rag_eval.metrics.generation import evaluate_generation
from rag_eval.metrics.retrieval import evaluate_retrieval
from rag_eval.registry import (
    AdapterValidationError,
    EvaluationAdapter,
    FakeAdapter,
    RunContext,
    load_adapter,
    validate_adapter_instance,
)
from rag_eval.report.json_report import write_json_report
from rag_eval.report.markdown_report import write_markdown_report


@dataclass
class RunOutcome:
    report: EvaluationReport
    predictions: list[Prediction]
    run_directory: Path
    failure_reasons: list[str]

    @property
    def failed(self) -> bool:
        return not self.report.summary.within_thresholds


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _code_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
        revision = completed.stdout.strip()
        if completed.returncode == 0 and revision:
            return revision
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _run_id(config_hash: str, dataset_hash: str, adapter: str, seed: int) -> str:
    return sha256_value(
        {
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "adapter": adapter,
            "seed": seed,
        }
    )[:24]


def _error_prediction(sample_id: str, status: str, error: str) -> Prediction:
    return Prediction(sample_id=sample_id, status=status, error=error)


def _execute_one(
    adapter: EvaluationAdapter,
    sample,
    context: RunContext,
    timeout_seconds: float,
) -> Prediction:
    result_queue: Queue[tuple[str, Any]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put(("ok", adapter.run(sample, context)), block=False)
        except Exception as exc:
            result_queue.put(
                ("error", f"{type(exc).__name__}: {exc}"),
                block=False,
            )

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        status, result = result_queue.get(timeout=timeout_seconds)
    except Empty:
        return _error_prediction(
            sample.id, "timeout", f"adapter exceeded {timeout_seconds:g}s timeout"
        )
    if status == "error":
        return _error_prediction(sample.id, "error", str(result))
    if isinstance(result, Prediction):
        prediction = result
    else:
        try:
            prediction = Prediction.model_validate(result)
        except Exception as exc:
            return _error_prediction(
                sample.id, "error", f"adapter returned invalid prediction: {exc}"
            )
    if prediction.sample_id != sample.id:
        return _error_prediction(
            sample.id,
            "error",
            f"adapter returned sample_id {prediction.sample_id!r}, expected {sample.id!r}",
        )
    return prediction


def _write_predictions(path: Path, predictions: list[Prediction]) -> None:
    lines = [prediction.model_dump_json() for prediction in predictions]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _summary(
    samples: int,
    predictions: list[Prediction],
    config: EvaluationConfig,
) -> tuple[RunSummary, list[str]]:
    errors = sum(
        prediction.status in {"error", "timeout"} for prediction in predictions
    )
    skipped = sum(prediction.status == "skipped" for prediction in predictions)
    completed = sum(prediction.status == "ok" for prediction in predictions)
    coverage = completed / samples if samples else 0.0
    error_rate = errors / samples if samples else 0.0
    reasons: list[str] = []
    if errors > config.max_errors:
        reasons.append(
            f"error samples {errors} exceed max_errors {config.max_errors}"
        )
    if coverage < config.min_coverage:
        reasons.append(
            f"coverage {coverage:.4f} is below min_coverage {config.min_coverage:.4f}"
        )
    return (
        RunSummary(
            expected_samples=samples,
            completed_samples=completed,
            error_samples=errors,
            skipped_samples=skipped,
            coverage=coverage,
            error_rate=error_rate,
            min_coverage=config.min_coverage,
            max_errors=config.max_errors,
            within_thresholds=not reasons,
        ),
        reasons,
    )


def _calculate_metrics(
    dataset: LoadedDataset,
    predictions: list[Prediction],
    config: EvaluationConfig,
) -> MetricSummary:
    summaries = [evaluate_generation(dataset.samples, predictions)]
    if dataset.manifest.task in {"retrieval", "agent"}:
        summaries.append(
            evaluate_retrieval(
                dataset.samples,
                predictions,
                config.k_values,
                qrels=dataset.qrels or None,
            )
        )
    if dataset.manifest.task == "agent":
        summaries.append(
            evaluate_agent(
                dataset.samples,
                predictions,
                evidence_k=max(config.k_values),
            )
        )
    return merge_metric_summaries(*summaries)


def run_evaluation(
    config: EvaluationConfig,
    *,
    adapter: EvaluationAdapter | None = None,
    dataset: LoadedDataset | None = None,
) -> RunOutcome:
    started_at = _utc_now()
    dataset = dataset or load_dataset(config.dataset)
    if adapter is None:
        adapter = load_adapter(config.adapter)
    else:
        validate_adapter_instance(adapter, source=config.adapter)
    if dataset.manifest.task not in adapter.capabilities:
        raise AdapterValidationError(
            adapter.name,
            [f"does not support dataset task {dataset.manifest.task!r}"],
        )

    config_hash = config.config_hash()
    run_id = _run_id(config_hash, dataset.content_hash, adapter.name, config.seed)
    context = RunContext(
        run_id=run_id,
        seed=config.seed,
        metadata={
            "dataset_id": dataset.manifest.dataset_id,
            "dataset_version": dataset.manifest.version,
            **config.metadata,
        },
    )
    warmup = getattr(adapter, "warmup", None)
    warmup_pending = callable(warmup)
    cache = PredictionCache(config.cache_dir)
    predictions: list[Prediction] = []
    for sample in dataset.samples:
        cache_key = prediction_cache_key(
            dataset_hash=dataset.content_hash,
            sample=sample.model_dump(mode="json"),
            adapter=adapter.name,
            model=config.model,
            prompt_version=config.prompt_version,
            seed=config.seed,
            config_hash=config_hash,
        )
        cached = cache.get(cache_key)
        if cached is not None and cached.sample_id == sample.id:
            predictions.append(cached)
            continue
        if warmup_pending:
            warmup(context)
            warmup_pending = False
        prediction = _execute_one(
            adapter, sample, context, config.timeout_seconds
        )
        if prediction.status == "ok":
            cache.put(cache_key, prediction)
        predictions.append(prediction)

    ended_at = _utc_now()
    summary, failure_reasons = _summary(len(dataset.samples), predictions, config)
    metrics = _calculate_metrics(dataset, predictions, config)
    manifest = RunManifest(
        run_id=run_id,
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.version,
        dataset_hash=dataset.content_hash,
        adapter=adapter.name,
        model=config.model,
        prompt_version=config.prompt_version,
        seed=config.seed,
        code_revision=_code_revision(),
        config_hash=config_hash,
        started_at=started_at,
        ended_at=ended_at,
    )
    report = EvaluationReport(
        run_manifest=manifest,
        summary=summary,
        metrics=metrics,
    )
    run_directory = Path(config.output_dir) / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    _write_predictions(run_directory / "predictions.jsonl", predictions)
    write_json_report(run_directory / "report.json", report)
    write_markdown_report(run_directory / "report.md", report)
    (run_directory / "run_manifest.json").write_text(
        canonical_json(manifest.model_dump(mode="json")),
        encoding="utf-8",
    )
    return RunOutcome(
        report=report,
        predictions=predictions,
        run_directory=run_directory,
        failure_reasons=failure_reasons,
    )


__all__ = ["FakeAdapter", "RunOutcome", "run_evaluation"]
