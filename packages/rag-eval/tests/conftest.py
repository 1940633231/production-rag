from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_dataset(
    root: Path,
    samples: list[dict[str, Any]],
    *,
    dataset_id: str = "test-dataset",
    version: str = "1.0.0",
    task: str = "retrieval",
) -> Path:
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": dataset_id,
        "version": version,
        "task": task,
        "samples_file": "samples.jsonl",
        "qrels_file": None,
        "description": "test dataset",
        "source": "local",
        "created_at": "2026-11-05T00:00:00+08:00",
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (dataset_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    return dataset_dir


def retrieval_sample(
    sample_id: str = "q1",
    *,
    document_id: str = "doc-1",
    start: int = 10,
    end: int = 20,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "task": "retrieval",
        "query": "What is the answer?",
        "history": [],
        "gold": {
            "documents": [
                {
                    "document_id": document_id,
                    "relevance": 1,
                    "spans": [{"start": start, "end": end}],
                }
            ],
            "chunks": [
                {
                    "chunk_id": f"{document_id}-chunk-1",
                    "document_id": document_id,
                    "relevance": 1,
                    "spans": [{"start": start, "end": end}],
                }
            ],
            "answers": ["Paris"],
            "values": ["Paris"],
            "behavior": "answer",
        },
        "metadata": {"tag": "test"},
    }
