"""Compare common metrics from legacy and rag-eval baseline JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _platform_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if "metrics" not in payload:
        raise ValueError("platform baseline has no metrics")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("platform metrics must be an object")
    values = metrics.get("values", metrics)
    if not isinstance(values, dict):
        raise ValueError("platform metric values must be an object")
    return {
        str(name): float(value)
        for name, value in values.items()
        if isinstance(value, (int, float))
    }


def _legacy_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("legacy baseline has no metrics")
    return {
        str(name): float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float))
    }


def compare(
    legacy_path: Path,
    platform_path: Path,
    tolerance: float,
) -> list[dict[str, float | str]]:
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    platform_payload = json.loads(platform_path.read_text(encoding="utf-8"))
    legacy = _legacy_metrics(legacy_payload)
    platform = _platform_metrics(platform_payload)
    rows: list[dict[str, float | str]] = []
    missing: list[str] = []
    for name, legacy_value in sorted(legacy.items()):
        if name not in platform:
            missing.append(name)
            continue
        platform_value = platform[name]
        delta = platform_value - legacy_value
        rows.append(
            {
                "metric": name,
                "legacy": legacy_value,
                "platform": platform_value,
                "delta": delta,
            }
        )
        if abs(delta) > tolerance:
            raise ValueError(
                f"{name} differs by {delta:+.6f} "
                f"(legacy={legacy_value:.6f}, platform={platform_value:.6f})"
            )
    if missing:
        raise ValueError("platform baseline is missing legacy metrics: " + ", ".join(missing))
    if not rows:
        raise ValueError("no common metrics were compared")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy")
    parser.add_argument("platform")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    try:
        rows = compare(
            Path(args.legacy),
            Path(args.platform),
            args.tolerance,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("metric,legacy,platform,delta")
    for row in rows:
        print(
            "{metric},{legacy:.6f},{platform:.6f},{delta:+.6f}".format(**row)
        )
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
