from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import retrieval_sample, write_dataset
from rag_eval.config import EvaluationConfig
from rag_eval.contracts import (
    DatasetValidationError,
    load_dataset,
    validate_dataset_dir,
)


def test_valid_dataset_loads_and_hash_is_stable(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path, [retrieval_sample()])
    first = load_dataset(dataset_dir)

    samples_path = dataset_dir / "samples.jsonl"
    samples_path.write_text(
        "\n" + samples_path.read_text(encoding="utf-8").replace("\n", "\n\n"),
        encoding="utf-8",
    )
    second = load_dataset(dataset_dir)

    assert first.manifest.dataset_id == "test-dataset"
    assert len(first.samples) == 1
    assert first.content_hash == second.content_hash


def test_dataset_hash_ignores_sample_line_order(tmp_path: Path) -> None:
    first_dir = write_dataset(
        tmp_path / "first",
        [retrieval_sample("b"), retrieval_sample("a")],
    )
    second_dir = write_dataset(
        tmp_path / "second",
        [retrieval_sample("a"), retrieval_sample("b")],
    )

    assert load_dataset(first_dir).content_hash == load_dataset(second_dir).content_hash


def test_duplicate_sample_ids_fail(tmp_path: Path) -> None:
    dataset_dir = write_dataset(
        tmp_path, [retrieval_sample("same"), retrieval_sample("same")]
    )

    dataset, errors = validate_dataset_dir(dataset_dir)

    assert dataset is None
    assert any("duplicate sample id" in error for error in errors)
    with pytest.raises(DatasetValidationError):
        load_dataset(dataset_dir)


def test_missing_document_id_fails(tmp_path: Path) -> None:
    sample = retrieval_sample()
    del sample["gold"]["documents"][0]["document_id"]
    dataset_dir = write_dataset(tmp_path, [sample])

    dataset, errors = validate_dataset_dir(dataset_dir)

    assert dataset is None
    assert any("document_id" in error for error in errors)


def test_dataset_task_mismatch_fails(tmp_path: Path) -> None:
    dataset_dir = write_dataset(
        tmp_path,
        [retrieval_sample()],
        task="generation",
    )

    dataset, errors = validate_dataset_dir(dataset_dir)

    assert dataset is None
    assert any("does not match dataset task" in error for error in errors)


def test_evaluation_config_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig(dataset="data", adapter="module:object", min_coverage=1.1)
    with pytest.raises(ValidationError):
        EvaluationConfig(dataset="data", adapter="module:object", k_values=[0])
