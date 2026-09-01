r"""OpenAI 兼容 LLM 后端测试。

覆盖:
  - generate：POST /chat/completions → 解析 choices[0].message.content
  - stream_generate：SSE data 行解析 delta.content
  - 重试：429 / 5xx / 网络异常做指数退避后成功；4xx（鉴权）直接抛
  - 缺 API key 报错；工厂 create_generator 路由

mock requests.post，无需真实 API。

运行:
  .venv\Scripts\python.exe -m pytest tests\generation\test_openai_compat.py -v
"""
import sys
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.openai_compat import OpenAICompatibleGenerator  # noqa: E402


class _FakeResp:
    """模拟 requests.Response。"""

    def __init__(self, status=200, body=None, sse_lines=None):
        self.status_code = status
        self._body = body
        self._sse_lines = sse_lines or []
        self.closed = False

    def json(self):
        return self._body

    @property
    def text(self):
        return "" if self._body is None else str(self._body)

    def iter_lines(self, decode_unicode=False):
        return iter(self._sse_lines)

    def close(self):
        self.closed = True


def _make(monkeypatch, fake_post, **kw):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("requests.post", fake_post)
    defaults = dict(model="test-model", base_url="https://llm.example/v1",
                    retry_times=2, retry_backoff=0)
    defaults.update(kw)
    return OpenAICompatibleGenerator(**defaults)


class TestGenerate:
    def test_parses_content_and_sends_openai_format(self, monkeypatch):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None, stream=False):
            calls.append((url, headers, json, stream))
            return _FakeResp(status=200, body={
                "choices": [{"message": {"content": "你好"}}],
            })

        g = _make(monkeypatch, fake_post)
        messages = [{"role": "user", "content": "hi"}]
        assert g.generate(messages) == "你好"
        url, headers, body, stream = calls[0]
        assert url == "https://llm.example/v1/chat/completions"
        assert body["model"] == "test-model"
        assert body["messages"] == messages
        assert body["stream"] is False
        assert headers["Authorization"] == "Bearer sk-test"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            OpenAICompatibleGenerator(model="m", base_url="https://x/v1")

    def test_retry_on_429_then_success(self, monkeypatch):
        seq = [
            _FakeResp(status=429),
            _FakeResp(status=200, body={"choices": [{"message": {"content": "ok"}}]}),
        ]
        g = _make(monkeypatch, lambda url, **kw: seq.pop(0))
        assert g.generate([]) == "ok"

    def test_retry_on_network_error(self, monkeypatch):
        calls = {"n": 0}

        def fake_post(url, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.exceptions.ConnectionError("down")
            return _FakeResp(status=200, body={
                "choices": [{"message": {"content": "ok"}}],
            })

        g = _make(monkeypatch, fake_post)
        assert g.generate([]) == "ok"
        assert calls["n"] == 2

    def test_retry_exhausted_raises(self, monkeypatch):
        def fake_post(url, **kw):
            return _FakeResp(status=503)

        g = _make(monkeypatch, fake_post, retry_times=1)
        with pytest.raises(RuntimeError, match="重试耗尽"):
            g.generate([])

    def test_non_retryable_4xx_raises_immediately(self, monkeypatch):
        calls = {"n": 0}

        def fake_post(url, **kw):
            calls["n"] += 1
            return _FakeResp(status=401)

        g = _make(monkeypatch, fake_post)
        with pytest.raises(RuntimeError, match="401"):
            g.generate([])
        assert calls["n"] == 1  # 4xx 不重试


class TestStream:
    def test_parses_sse_deltas(self, monkeypatch):
        lines = [
            'data: {"choices":[{"delta":{"content":"你"}}]}',
            'data: {"choices":[{"delta":{"content":"好"}}]}',
            'data: [DONE]',
            "",
        ]
        g = _make(monkeypatch, lambda url, **kw: _FakeResp(status=200, sse_lines=lines))
        out = "".join(g.stream_generate([{"role": "user", "content": "hi"}]))
        assert out == "你好"

    def test_skips_empty_and_non_data_lines(self, monkeypatch):
        lines = [
            ': keepalive',
            'data: {"choices":[{"delta":{"content":""}}]}',
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: {"choices":[]}',
            'data: [DONE]',
        ]
        g = _make(monkeypatch, lambda url, **kw: _FakeResp(status=200, sse_lines=lines))
        assert "".join(g.stream_generate([])) == "A"

    def test_retry_on_429_then_stream(self, monkeypatch):
        seq = [
            _FakeResp(status=429),
            _FakeResp(status=200, sse_lines=[
                'data: {"choices":[{"delta":{"content":"B"}}]}',
                'data: [DONE]',
            ]),
        ]
        g = _make(monkeypatch, lambda url, **kw: seq.pop(0))
        assert "".join(g.stream_generate([])) == "B"


class TestFactory:
    def test_openai_backend_routed(self):
        from app.generation.generator import create_generator

        g = create_generator(
            "openai", model="m", api_key="k", base_url="https://x/v1",
        )
        assert isinstance(g, OpenAICompatibleGenerator)

    def test_unknown_backend_raises(self):
        from app.generation.generator import create_generator

        with pytest.raises(ValueError, match="Unknown generator backend"):
            create_generator("nope")
