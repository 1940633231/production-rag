"""Shared data contracts and dataset loading for rag-eval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskType = Literal["retrieval", "generation", "agent"]
BehaviorType = Literal["answer", "refuse", "human_review"]
PredictionStatus = Literal["ok", "error", "timeout", "skipped"]


class StrictModel(BaseModel):
    """Base model that keeps dataset and report typos visible."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Span(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> Span:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end


class GoldDocument(StrictModel):
    document_id: str = Field(min_length=1)
    relevance: float = Field(default=1.0, ge=0.0)
    spans: list[Span] = Field(default_factory=list)

    @field_validator("document_id")
    @classmethod
    def strip_document_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("document_id must not be blank")
        return value


class GoldChunk(StrictModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    relevance: float = Field(default=1.0, ge=0.0)
    spans: list[Span] = Field(default_factory=list)

    @field_validator("chunk_id", "document_id")
    @classmethod
    def strip_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chunk_id and document_id must not be blank")
        return value


class Gold(StrictModel):
    documents: list[GoldDocument] = Field(default_factory=list)
    chunks: list[GoldChunk] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    behavior: BehaviorType = "answer"


class EvalSample(StrictModel):
    id: str = Field(min_length=1)
    task: TaskType = "retrieval"
    query: str = Field(min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)
    gold: Gold = Field(default_factory=Gold)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def strip_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sample id must not be blank")
        return value

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class RetrievedDocument(StrictModel):
    document_id: str = Field(min_length=1)
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(StrictModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)
    span: Span | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(StrictModel):
    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    span: Span | None = None
    quote: str | None = None


class Evidence(StrictModel):
    document_id: str = Field(min_length=1)
    chunk_id: str | None = None
    span: Span | None = None
    text: str | None = None


class Prediction(StrictModel):
    sample_id: str = Field(min_length=1)
    status: PredictionStatus = "ok"
    documents: list[RetrievedDocument] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0.0)
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id")
    @classmethod
    def strip_sample_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sample_id must not be blank")
        return value


class DatasetManifest(StrictModel):
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task: TaskType
    samples_file: str = Field(min_length=1)
    qrels_file: str | None = None
    description: str = ""
    source: str = "local"
    created_at: datetime

    @field_validator("dataset_id", "version", "samples_file")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class MetricSummary(StrictModel):
    values: dict[str, float] = Field(default_factory=dict)
    denominators: dict[str, int] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("values")
    @classmethod
    def validate_finite_values(cls, values: dict[str, float]) -> dict[str, float]:
        for name, value in values.items():
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"metric {name!r} must be finite")
        return values


class RunManifest(StrictModel):
    run_id: str = Field(min_length=1)
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    adapter: str
    model: str | None = None
    prompt_version: str | None = None
    seed: int
    code_revision: str
    config_hash: str
    started_at: datetime
    ended_at: datetime

    @model_validator(mode="after")
    def validate_times(self) -> RunManifest:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        return self


class RunSummary(StrictModel):
    expected_samples: int = Field(ge=0)
    completed_samples: int = Field(ge=0)
    error_samples: int = Field(ge=0)
    skipped_samples: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    min_coverage: float = Field(ge=0.0, le=1.0)
    max_errors: int = Field(ge=0)
    within_thresholds: bool


class EvaluationReport(StrictModel):
    schema_version: Literal["1"] = "1"
    run_manifest: RunManifest
    summary: RunSummary
    metrics: MetricSummary


@dataclass(frozen=True)
class LoadedDataset:
    manifest: DatasetManifest
    samples: list[EvalSample]
    content_hash: str
    directory: Path
    qrels: dict[str, dict[str, float]]


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def compute_dataset_hash(
    manifest: DatasetManifest,
    samples: list[EvalSample],
    qrels: dict[str, dict[str, float]] | None = None,
) -> str:
    identity = {
        "dataset_id": manifest.dataset_id,
        "version": manifest.version,
        "task": manifest.task,
        "samples": [
            sample.model_dump(mode="json")
            for sample in sorted(samples, key=lambda item: item.id)
        ],
        "qrels": qrels or {},
    }
    return sha256_value(identity)


def _load_json(path: Path, label: str, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        errors.append(f"{label} not found: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"{label} is not valid JSON: {path}: {exc}")
    except OSError as exc:
        errors.append(f"cannot read {label}: {path}: {exc}")
    return None


def _find_manifest(directory: Path) -> Path:
    for name in ("manifest.json", "dataset_manifest.json", "dataset.json"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return directory / "manifest.json"


def _load_samples(path: Path, errors: list[str]) -> list[EvalSample]:
    samples: list[EvalSample] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        errors.append(f"samples file not found: {path}")
        return samples
    except OSError as exc:
        errors.append(f"cannot read samples file: {path}: {exc}")
        return samples

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            samples.append(EvalSample.model_validate(payload))
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
        except Exception as exc:
            errors.append(f"{path}:{line_number}: invalid sample: {exc}")
    return samples


def _load_qrels(path: Path | None, errors: list[str]) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = _load_json(path, "qrels file", errors)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        errors.append(f"qrels file must contain a JSON object: {path}")
        return {}

    qrels: dict[str, dict[str, float]] = {}
    for query_id, documents in payload.items():
        if not isinstance(query_id, str) or not query_id:
            errors.append("qrels query ids must be non-empty strings")
            continue
        if not isinstance(documents, dict):
            errors.append(f"qrels for query {query_id!r} must be an object")
            continue
        qrels[query_id] = {}
        for document_id, relevance in documents.items():
            if not isinstance(document_id, str) or not document_id:
                errors.append(f"qrels for query {query_id!r} has a blank document id")
                continue
            if not isinstance(relevance, (int, float)) or relevance < 0:
                errors.append(
                    f"qrels relevance for {query_id!r}/{document_id!r} must be non-negative"
                )
                continue
            qrels[query_id][document_id] = float(relevance)
    return qrels


def validate_dataset_dir(directory: str | Path) -> tuple[LoadedDataset | None, list[str]]:
    root = Path(directory)
    errors: list[str] = []
    if not root.is_dir():
        return None, [f"dataset directory not found: {root}"]

    manifest_path = _find_manifest(root)
    manifest_payload = _load_json(manifest_path, "dataset manifest", errors)
    if manifest_payload is None:
        return None, errors
    try:
        manifest = DatasetManifest.model_validate(manifest_payload)
    except Exception as exc:
        errors.append(f"invalid dataset manifest: {exc}")
        return None, errors

    samples_path = root / manifest.samples_file
    samples = _load_samples(samples_path, errors)
    sample_ids: set[str] = set()
    for sample in samples:
        if sample.id in sample_ids:
            errors.append(f"duplicate sample id: {sample.id}")
        sample_ids.add(sample.id)
        if sample.task != manifest.task:
            errors.append(
                f"sample {sample.id!r} task {sample.task!r} does not match "
                f"dataset task {manifest.task!r}"
            )

    qrels_path = root / manifest.qrels_file if manifest.qrels_file else None
    qrels = _load_qrels(qrels_path, errors)
    for query_id in qrels:
        if query_id not in sample_ids:
            errors.append(f"qrels query id has no sample: {query_id}")

    if not samples and not any("samples file" in error for error in errors):
        errors.append("dataset must contain at least one sample")
    if errors:
        return None, errors

    return (
        LoadedDataset(
            manifest=manifest,
            samples=samples,
            content_hash=compute_dataset_hash(manifest, samples, qrels),
            directory=root,
            qrels=qrels,
        ),
        [],
    )


def load_dataset(directory: str | Path) -> LoadedDataset:
    dataset, errors = validate_dataset_dir(directory)
    if errors or dataset is None:
        raise DatasetValidationError(errors or ["unknown dataset validation error"])
    return dataset
