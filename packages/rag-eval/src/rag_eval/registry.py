"""Adapter discovery and the built-in deterministic fake adapter."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rag_eval.contracts import (
    Citation,
    EvalSample,
    Evidence,
    Prediction,
    RetrievedChunk,
    RetrievedDocument,
)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class EvaluationAdapter(Protocol):
    name: str
    capabilities: set[str]

    def validate(self) -> list[str]:
        """Return human-readable validation errors."""

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        """Evaluate one sample without calculating platform metrics."""


class AdapterValidationError(ValueError):
    def __init__(self, adapter: str, errors: list[str]):
        self.adapter = adapter
        self.errors = errors
        super().__init__(f"adapter {adapter!r} is invalid: " + "; ".join(errors))


class FakeAdapter:
    """Adapter that returns gold evidence and answers deterministically."""

    name = "fake"
    capabilities = {"retrieval", "generation", "agent", "citation", "latency"}

    def validate(self) -> list[str]:
        return []

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        documents = [
            RetrievedDocument(
                document_id=document.document_id,
                score=document.relevance,
                rank=index,
                metadata={"source": "fake-gold"},
            )
            for index, document in enumerate(sample.gold.documents, start=1)
        ]
        chunks = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                score=chunk.relevance,
                rank=index,
                span=chunk.spans[0] if chunk.spans else None,
                metadata={"source": "fake-gold"},
            )
            for index, chunk in enumerate(sample.gold.chunks, start=1)
        ]
        answer = None
        if sample.gold.behavior == "answer" and sample.gold.answers:
            answer = sample.gold.answers[0]
        elif sample.gold.behavior == "answer" and sample.gold.values:
            answer = sample.gold.values[0]

        citations: list[Citation] = []
        evidence: list[Evidence] = []
        for chunk in sample.gold.chunks:
            span = chunk.spans[0] if chunk.spans else None
            citations.append(
                Citation(document_id=chunk.document_id, chunk_id=chunk.chunk_id, span=span)
            )
            evidence.append(
                Evidence(document_id=chunk.document_id, chunk_id=chunk.chunk_id, span=span)
            )
        if not citations:
            citations = [
                Citation(document_id=document.document_id)
                for document in sample.gold.documents
            ]
            evidence = [
                Evidence(document_id=document.document_id)
                for document in sample.gold.documents
            ]

        return Prediction(
            sample_id=sample.id,
            status="ok",
            documents=documents,
            chunks=chunks,
            answer=answer,
            citations=citations,
            evidence=evidence,
            latency_ms=0.0,
            usage={"source": "fake", "context": context.run_id},
            metadata={
                "behavior": sample.gold.behavior,
                "retry_count": 0,
                "rewrite_sources": 0,
            },
        )


def _resolve_object(spec: str) -> Any:
    module_name, separator, object_path = spec.partition(":")
    if not separator or not module_name or not object_path:
        raise ValueError("adapter must use 'module:object' syntax")
    working_directory = str(Path.cwd())
    if working_directory not in sys.path:
        # Console entry points start with the executable directory, not cwd.
        sys.path.insert(0, working_directory)
    module = importlib.import_module(module_name)
    value: Any = module
    for part in object_path.split("."):
        if not part:
            raise ValueError(f"invalid adapter object path: {spec!r}")
        value = getattr(value, part)
    if isinstance(value, type):
        value = value()
    return value


def load_adapter(spec: str) -> EvaluationAdapter:
    try:
        adapter = _resolve_object(spec)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise AdapterValidationError(spec, [str(exc)]) from exc

    validate_adapter_instance(adapter, source=spec)
    return adapter


def validate_adapter_instance(adapter: Any, *, source: str | None = None) -> None:
    label = source or getattr(adapter, "name", type(adapter).__name__)
    errors: list[str] = []
    name = getattr(adapter, "name", None)
    capabilities = getattr(adapter, "capabilities", None)
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")
    if not isinstance(capabilities, (set, frozenset)) or not all(
        isinstance(item, str) and item for item in capabilities
    ):
        errors.append("capabilities must be a set of non-empty strings")
    if not callable(getattr(adapter, "validate", None)):
        errors.append("validate() is required")
    if not callable(getattr(adapter, "run", None)):
        errors.append("run(sample, context) is required")
    if errors:
        raise AdapterValidationError(label, errors)

    validation_errors = adapter.validate()
    if validation_errors:
        raise AdapterValidationError(label, list(validation_errors))
