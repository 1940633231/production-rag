"""JSON report serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_eval.contracts import EvaluationReport


def write_json_report(path: str | Path, report: EvaluationReport) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output


def load_report_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report JSON must contain an object")
    return payload
