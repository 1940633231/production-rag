from __future__ import annotations

import json
from pathlib import Path

from eval_adapter import (
    HttpEvaluationAdapter,
    LocalRetrievalAdapter,
    ProductionRAGServiceAdapter,
)
from rag_eval.contracts import EvalSample, load_dataset
from rag_eval.registry import RunContext
from scripts.migrate_rag_eval_datasets import migrate_production_smoke


def _sample() -> EvalSample:
    return EvalSample.model_validate(
        {
            "id": "q1",
            "task": "retrieval",
            "query": "iron supply",
            "gold": {
                "documents": [
                    {"document_id": "doc-1", "relevance": 1, "spans": []}
                ]
            },
        }
    )


def _context() -> RunContext:
    return RunContext(run_id="run-1", seed=42)


def test_local_adapter_maps_metadata_hits_without_external_services(
    tmp_path: Path,
) -> None:
    metadata = {
        "1": {
            "chunk_id": "doc-1-chunk-1",
            "document_id": "doc-1",
            "content": "iron supply demand market",
            "start_offset": 0,
            "end_offset": 27,
            "metadata": {"document_id": "doc-1"},
        },
        "2": {
            "chunk_id": "doc-2-chunk-1",
            "document_id": "doc-2",
            "content": "weather report",
            "start_offset": 0,
            "end_offset": 14,
            "metadata": {"document_id": "doc-2"},
        },
        "3": {
            "chunk_id": "doc-3-chunk-1",
            "document_id": "doc-3",
            "content": "finance handbook",
            "start_offset": 0,
            "end_offset": 16,
            "metadata": {"document_id": "doc-3"},
        },
        "4": {
            "chunk_id": "doc-4-chunk-1",
            "document_id": "doc-4",
            "content": "artificial intelligence history",
            "start_offset": 0,
            "end_offset": 29,
            "metadata": {"document_id": "doc-4"},
        },
        "5": {
            "chunk_id": "doc-5-chunk-1",
            "document_id": "doc-5",
            "content": "annual leave policy",
            "start_offset": 0,
            "end_offset": 19,
            "metadata": {"document_id": "doc-5"},
        },
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    adapter = LocalRetrievalAdapter(
        metadata_path=metadata_path,
        top_k=2,
    )

    assert adapter.validate() == []
    prediction = adapter.run(_sample(), _context())

    assert prediction.sample_id == "q1"
    assert prediction.documents[0].document_id == "doc-1"
    assert prediction.chunks[0].span.start == 0
    assert prediction.chunks[0].span.end == 27
    assert prediction.usage["backend"] == "local_bm25"


def test_local_adapter_reports_missing_metadata(tmp_path: Path) -> None:
    adapter = LocalRetrievalAdapter(metadata_path=tmp_path / "missing.json")

    errors = adapter.validate()

    assert errors
    assert "not found" in errors[0]


def test_http_adapter_maps_chat_response() -> None:
    def transport(endpoint, payload, headers, timeout_seconds):
        assert endpoint == "http://localhost:8000/api/chat"
        assert payload["query"] == "iron supply"
        assert headers["Content-Type"] == "application/json"
        assert timeout_seconds == 5
        return {
            "answer": "Supply is high [1].",
            "chunks": [
                {
                    "chunk_id": "doc-1-chunk-1",
                    "content": "iron supply",
                    "start_offset": 0,
                    "end_offset": 11,
                    "score": 1.5,
                    "metadata": {"document_id": "doc-1"},
                }
            ],
            "citations": [{"chunk_id": "doc-1-chunk-1"}],
            "stats": {},
        }

    adapter = HttpEvaluationAdapter(
        endpoint="http://localhost:8000",
        timeout_seconds=5,
        transport=transport,
    )

    assert adapter.validate() == []
    prediction = adapter.run(_sample(), _context())

    assert prediction.answer == "Supply is high [1]."
    assert prediction.documents[0].document_id == "doc-1"
    assert prediction.citations[0].document_id == "doc-1"


def test_service_adapter_validates_config_without_building_service(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("retrieval: {}\n", encoding="utf-8")
    adapter = ProductionRAGServiceAdapter(config_path=config_path)

    assert adapter.validate() == []
    assert adapter._service is None


def test_production_smoke_migration_adds_document_ids(tmp_path: Path) -> None:
    output = migrate_production_smoke(tmp_path / "migrated")
    dataset = load_dataset(output)

    assert len(dataset.samples) == 4
    assert all(sample.gold.documents for sample in dataset.samples)
    assert all(
        document.document_id
        for sample in dataset.samples
        for document in sample.gold.documents
    )
