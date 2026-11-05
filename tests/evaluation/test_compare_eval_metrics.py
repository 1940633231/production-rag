from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_eval_metrics import compare


def test_compare_legacy_and_platform_metrics(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    platform = tmp_path / "platform.json"
    legacy.write_text(
        json.dumps({"metrics": {"recall@5": 0.8, "ndcg@10": 0.9}}),
        encoding="utf-8",
    )
    platform.write_text(
        json.dumps(
            {"metrics": {"values": {"recall@5": 0.79, "ndcg@10": 0.905}}}
        ),
        encoding="utf-8",
    )

    rows = compare(legacy, platform, tolerance=0.02)

    assert {row["metric"] for row in rows} == {"recall@5", "ndcg@10"}


def test_compare_rejects_platform_metric_drift(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    platform = tmp_path / "platform.json"
    legacy.write_text(
        json.dumps({"metrics": {"recall@5": 0.8}}),
        encoding="utf-8",
    )
    platform.write_text(
        json.dumps({"metrics": {"values": {"recall@5": 0.5}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recall@5"):
        compare(legacy, platform, tolerance=0.02)
