"""Deterministic prediction cache backed by JSON files."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from rag_eval.contracts import Prediction, canonical_json, sha256_value


def prediction_cache_key(
    *,
    dataset_hash: str,
    sample: dict[str, Any],
    adapter: str,
    model: str | None,
    prompt_version: str | None,
    seed: int,
    config_hash: str,
) -> str:
    return sha256_value(
        {
            "cache_schema": "1",
            "dataset_hash": dataset_hash,
            "sample": sample,
            "adapter": adapter,
            "model": model,
            "prompt_version": prompt_version,
            "seed": seed,
            "config_hash": config_hash,
        }
    )


class PredictionCache:
    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory) if directory else None
        self._lock = threading.Lock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Prediction | None:
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_schema") != "1" or payload.get("key") != key:
                return None
            return Prediction.model_validate(payload["prediction"])
        except (
            AttributeError,
            FileNotFoundError,
            json.JSONDecodeError,
            KeyError,
            OSError,
            ValueError,
        ):
            return None

    def put(self, key: str, prediction: Prediction) -> None:
        if self.directory is None:
            return
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".json.tmp")
        payload = {
            "cache_schema": "1",
            "key": key,
            "prediction": prediction.model_dump(mode="json"),
        }
        with self._lock:
            temporary.write_text(canonical_json(payload), encoding="utf-8")
            os.replace(temporary, path)
