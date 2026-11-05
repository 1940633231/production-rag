"""Build rag-eval datasets without modifying legacy evaluation assets.

Usage:
  .venv\\Scripts\\python.exe scripts\\migrate_rag_eval_datasets.py production-smoke
  .venv\\Scripts\\python.exe scripts\\migrate_rag_eval_datasets.py cmteb-50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_DATASET = PROJECT_ROOT / "data/evaluation/retrieval_dataset.json"
SMOKE_OUTPUT = PROJECT_ROOT / "data/evaluation/rag_eval/production_smoke"
CMTEB_OUTPUT = PROJECT_ROOT / "data/evaluation/rag_eval/cmteb_50"

LEGACY_DOCUMENT_IDS = [
    "test",
    "financial_report_q3",
    "ai_history",
    "company_handbook",
]

SMOKE_ANSWERS = [
    ["供强需弱", "低位窄幅震荡"],
    ["1.5 亿元", "3500 万元"],
    ["早期阶段", "黄金时代", "专家系统", "现代 AI"],
    ["10 天", "1 天"],
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _manifest(
    dataset_id: str,
    task: str,
    description: str,
    qrels_file: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "version": "1.0.0",
        "task": task,
        "samples_file": "samples.jsonl",
        "qrels_file": qrels_file,
        "description": description,
        "source": "production-rag",
        "created_at": "2026-11-05T00:00:00+08:00",
    }


def migrate_production_smoke(output_dir: Path = SMOKE_OUTPUT) -> Path:
    legacy = json.loads(LEGACY_DATASET.read_text(encoding="utf-8"))
    if len(legacy) != len(LEGACY_DOCUMENT_IDS):
        raise ValueError(
            "legacy production dataset no longer matches the documented 4-sample mapping"
        )

    samples: list[dict[str, Any]] = []
    for index, (item, document_id) in enumerate(
        zip(legacy, LEGACY_DOCUMENT_IDS), start=1
    ):
        samples.append(
            {
                "id": f"production-smoke-{index:03d}",
                "task": "retrieval",
                "query": item["query"],
                "history": [],
                "gold": {
                    "documents": [
                        {
                            "document_id": document_id,
                            "relevance": 1,
                            "spans": item.get("relevant_spans", []),
                        }
                    ],
                    "chunks": [],
                    "answers": SMOKE_ANSWERS[index - 1],
                    "values": SMOKE_ANSWERS[index - 1],
                    "behavior": "answer",
                },
                "metadata": {
                    "source_dataset": "data/evaluation/retrieval_dataset.json",
                    "migration": "explicit-document-id",
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "manifest.json",
        _manifest(
            "production-rag-smoke",
            "retrieval",
            "Four production-rag smoke queries migrated with explicit document_id.",
        ),
    )
    _write_jsonl(output_dir / "samples.jsonl", samples)
    return output_dir


def migrate_cmteb_50(
    output_dir: Path = CMTEB_OUTPUT,
    *,
    max_corpus: int = 1000,
    max_queries: int = 50,
    fill_factor: int = 5,
    seed: int = 42,
) -> Path:
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import eval_cmteb

    corpus, queries, relevant_map = eval_cmteb.load_data(
        max_corpus=max_corpus,
        max_queries=max_queries,
        fill_factor=fill_factor,
        seed=seed,
    )
    corpus_ids = set(corpus["id"].astype(str))
    samples: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, float]] = {}
    missing = 0

    for row in queries.itertuples():
        query_id = str(row.id)
        relevant_ids = sorted(
            document_id
            for document_id in relevant_map.get(query_id, set())
            if document_id in corpus_ids
        )
        if not relevant_ids:
            missing += 1
        qrels[query_id] = {document_id: 1.0 for document_id in relevant_ids}
        samples.append(
            {
                "id": query_id,
                "task": "retrieval",
                "query": str(row.text),
                "history": [],
                "gold": {
                    "documents": [
                        {
                            "document_id": document_id,
                            "relevance": 1,
                            "spans": [],
                        }
                        for document_id in relevant_ids
                    ],
                    "chunks": [],
                    "answers": [],
                    "values": [],
                    "behavior": "answer",
                },
                "metadata": {
                    "source_dataset": "C-MTEB/T2Retrieval",
                    "subset": {
                        "seed": seed,
                        "max_queries": max_queries,
                        "max_corpus": max_corpus,
                        "fill_factor": fill_factor,
                    },
                },
            }
        )
    if missing:
        raise ValueError(
            f"{missing} sampled queries have no relevant document in the corpus subset"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "manifest.json",
        _manifest(
            "cmteb-t2retrieval-50",
            "retrieval",
            (
                "C-MTEB T2Retrieval subset generated from the existing benchmark "
                f"pipeline (seed={seed}, max_corpus={max_corpus}, "
                f"fill_factor={fill_factor})."
            ),
            qrels_file="qrels.json",
        ),
    )
    _write_jsonl(output_dir / "samples.jsonl", samples)
    _write_json(output_dir / "qrels.json", qrels)
    _write_json(
        output_dir / "subset.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(),
            "seed": seed,
            "max_queries": max_queries,
            "max_corpus": max_corpus,
            "fill_factor": fill_factor,
            "corpus_size": len(corpus_ids),
            "query_count": len(samples),
        },
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=["production-smoke", "cmteb-50"],
    )
    parser.add_argument("--output")
    parser.add_argument("--max-corpus", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--fill-factor", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dataset == "production-smoke":
        output = migrate_production_smoke(
            Path(args.output) if args.output else SMOKE_OUTPUT
        )
    else:
        output = migrate_cmteb_50(
            Path(args.output) if args.output else CMTEB_OUTPUT,
            max_corpus=args.max_corpus,
            max_queries=args.max_queries,
            fill_factor=args.fill_factor,
            seed=args.seed,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
