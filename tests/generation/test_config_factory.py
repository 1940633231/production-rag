"""Config 环境变量解析与统一 Generator 工厂测试。"""
from pathlib import Path

import pytest
import yaml

from app.core.config import Config
from app.generation.generator import create_generator_from_config
from app.generation.llm import LLMGenerator


def _config(tmp_path: Path, llm_cfg=None) -> Config:
    data = {
        "generation": {
            "backend": "llm",
            "temperature": 0.1,
            "max_tokens": 256,
            "timeout": 10,
            "retry_times": 0,
            "retry_backoff": 0,
            "max_concurrency": 1,
            "llm": llm_cfg or {},
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return Config(str(path))


def test_llm_env_overrides_take_priority(tmp_path, monkeypatch):
    cfg = _config(tmp_path, {
        "base_url_env": "LLM_BASE_URL",
        "model_env": "LLM_MODEL,FALLBACK_MODEL",
    })
    monkeypatch.setenv("LLM_BASE_URL", "https://primary.example/v1/")
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model-env")

    assert cfg.generation_llm_base_url == "https://primary.example/v1"
    assert cfg.generation_llm_model == "primary-model"
    assert cfg.generation_effective_model == "primary-model"


def test_llm_env_uses_fallback_candidate(tmp_path, monkeypatch):
    cfg = _config(tmp_path, {
        "model_env": "LLM_MODEL,FALLBACK_MODEL",
    })
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model")

    assert cfg.generation_llm_model == "fallback-model"


def test_factory_builds_llm_generator_from_config(tmp_path, monkeypatch):
    cfg = _config(tmp_path, {
        "base_url_env": "LLM_BASE_URL",
        "api_key_env": "PRIMARY_LLM_KEY,LLM_API_KEY",
        "model_env": "LLM_MODEL",
        "require_api_key": True,
    })
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("LLM_MODEL", "provider-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")

    generator = create_generator_from_config(cfg)
    assert isinstance(generator, LLMGenerator)
    assert generator.model == "provider-model"
    assert generator._chat_url == "https://provider.example/v1/chat/completions"
    assert generator.api_key == "sk-test"


def test_factory_allows_local_endpoint_without_key(tmp_path, monkeypatch):
    cfg = _config(tmp_path, {
        "base_url_env": "LOCAL_LLM_BASE_URL",
        "api_key_env": "MISSING_LLM_KEY",
        "model_env": "LOCAL_LLM_MODEL",
        "require_api_key": False,
    })
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "local-model")
    monkeypatch.delenv("MISSING_LLM_KEY", raising=False)

    generator = create_generator_from_config(cfg)
    assert isinstance(generator, LLMGenerator)
    assert "Authorization" not in generator._headers


def test_factory_requires_base_url_and_model_from_env(tmp_path, monkeypatch):
    cfg = _config(tmp_path, {
        "base_url_env": "MISSING_LLM_BASE_URL",
        "model_env": "MISSING_LLM_MODEL",
        "require_api_key": False,
    })
    monkeypatch.delenv("MISSING_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MISSING_LLM_MODEL", raising=False)

    with pytest.raises(ValueError, match="MISSING_LLM_BASE_URL"):
        create_generator_from_config(cfg)
