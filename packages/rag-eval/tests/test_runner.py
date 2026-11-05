from __future__ import annotations

import time
from pathlib import Path

from conftest import retrieval_sample, write_dataset
from rag_eval.config import EvaluationConfig
from rag_eval.contracts import Prediction
from rag_eval.registry import FakeAdapter
from rag_eval.runner import run_evaluation


def _config(tmp_path: Path, dataset_dir: Path, **overrides) -> EvaluationConfig:
    values = {
        "dataset": str(dataset_dir),
        "adapter": "rag_eval.registry:FakeAdapter",
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "reports"),
        "min_coverage": 1.0,
        "max_errors": 0,
        "k_values": [5],
    }
    values.update(overrides)
    return EvaluationConfig(**values)


def test_fake_adapter_runs_and_writes_reports(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    outcome = run_evaluation(_config(tmp_path, dataset_dir), adapter=FakeAdapter())

    assert not outcome.failed
    assert outcome.report.summary.completed_samples == 1
    assert outcome.report.summary.coverage == 1.0
    assert outcome.report.metrics.denominators["recall@5"] == 1
    assert (outcome.run_directory / "predictions.jsonl").is_file()
    assert (outcome.run_directory / "report.json").is_file()
    assert (outcome.run_directory / "report.md").is_file()
    assert (outcome.run_directory / "run_manifest.json").is_file()


class ExplodingFake(FakeAdapter):
    def run(self, sample, context):
        raise AssertionError("cache should have prevented adapter execution")


class WarmupFake(FakeAdapter):
    name = "warmup-fake"

    def __init__(self):
        self.warmed = False

    def warmup(self, context):
        self.warmed = True

    def run(self, sample, context):
        assert self.warmed
        return super().run(sample, context)


class CachedOnlyFake(FakeAdapter):
    def warmup(self, context):
        raise AssertionError("warmup should not run when all predictions are cached")

    def run(self, sample, context):
        raise AssertionError("run should not execute when all predictions are cached")


def test_runner_calls_optional_warmup_before_timed_samples(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    adapter = WarmupFake()

    outcome = run_evaluation(_config(tmp_path, dataset_dir), adapter=adapter)

    assert adapter.warmed
    assert not outcome.failed


def test_runner_skips_warmup_when_cache_is_complete(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    config = _config(tmp_path, dataset_dir)
    run_evaluation(config, adapter=FakeAdapter())

    cached = run_evaluation(config, adapter=CachedOnlyFake())

    assert not cached.failed
    assert cached.report.summary.completed_samples == 1


def test_second_run_uses_prediction_cache(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    config = _config(tmp_path, dataset_dir)
    first = run_evaluation(config, adapter=FakeAdapter())
    second = run_evaluation(config, adapter=ExplodingFake())

    assert first.predictions[0].model_dump() == second.predictions[0].model_dump()


class FailingAdapter(FakeAdapter):
    name = "failing"

    def run(self, sample, context):
        return Prediction(sample_id=sample.id, status="error", error="expected failure")


def test_failure_is_persisted_and_fails_completion_threshold(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    outcome = run_evaluation(
        _config(tmp_path, dataset_dir),
        adapter=FailingAdapter(),
    )

    assert outcome.failed
    assert outcome.report.summary.error_samples == 1
    assert outcome.report.summary.coverage == 0.0
    predictions_text = (
        outcome.run_directory / "predictions.jsonl"
    ).read_text(encoding="utf-8")
    assert '"status":"error"' in predictions_text


class SlowAdapter(FakeAdapter):
    name = "slow"

    def run(self, sample, context):
        time.sleep(0.2)
        return super().run(sample, context)


def test_timeout_is_recorded_as_failed_sample(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    outcome = run_evaluation(
        _config(tmp_path, dataset_dir, timeout_seconds=0.01),
        adapter=SlowAdapter(),
    )

    assert outcome.failed
    assert outcome.predictions[0].status == "timeout"
    assert outcome.report.summary.error_samples == 1
