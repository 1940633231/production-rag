"""OpenAI 兼容 LLM 生成器：通过任意 OpenAI 兼容 /chat/completions 端点调用。

解除 DashScope 绑定：支持任何提供 OpenAI 兼容接口的 LLM 服务——
OpenAI、DeepSeek、Moonshot(Kimi)、智谱 GLM、阿里百炼 compatible-mode、
vLLM、Ollama、本地推理网关等，只需配置 base_url + api_key + model。

调用格式为 OpenAI messages：[{role, content}, ...]
- generate:       POST {base_url}/chat/completions（stream=false）→ choices[0].message.content
- stream_generate: POST stream=true → 解析 SSE data 行的 choices[0].delta.content
- 重试：429 限流 / 5xx / 网络异常做指数退避；其余 4xx（鉴权/参数）直接抛
- 并发：与 QwenGenerator 相同的 BoundedSemaphore 限流
"""
import json
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from app.core.logger import get_logger
from app.generation.generator import BaseGenerator

logger = get_logger(__name__)


class OpenAICompatibleGenerator(BaseGenerator):
    """通用 OpenAI 兼容生成器（任意 base_url）。"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: Optional[int] = 60,
        retry_times: int = 2,
        retry_backoff: float = 1.0,
        max_concurrency: int = 4,
    ):
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_times = retry_times
        self.retry_backoff = retry_backoff
        # 并发信号量（与 QwenGenerator 一致）：0 表示不限制
        self._sem = (
            threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None
        )

        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            logger.error("API key 缺失: 环境变量 %s 未设置", api_key_env)
            raise ValueError(
                "OpenAI 兼容后端需要 API key：设置环境变量 {} 或构造时传入 api_key".format(
                    api_key_env
                )
            )

        self._chat_url = self.base_url + "/chat/completions"
        self._headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }
        logger.info(
            "OpenAICompatibleGenerator 初始化: base_url=%s, model=%s, "
            "temperature=%.2f, max_tokens=%d",
            self.base_url, model, temperature, max_tokens,
        )

    # ---- 内部 ----

    def _request(self, messages: List[Dict], stream: bool) -> Any:
        """发起 POST 请求（requests），返回 Response。"""
        import requests

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        return requests.post(
            self._chat_url,
            headers=self._headers,
            json=body,
            timeout=self.timeout,
            stream=stream,
        )

    def _wait_backoff(self, attempt: int, exc: Exception):
        """指数退避等待，并记录重试日志。"""
        wait = self.retry_backoff * (2 ** attempt)
        logger.warning(
            "LLM 调用失败（可重试）: attempt=%d/%d, wait=%.1fs, error=%s: %s",
            attempt + 1, self.retry_times + 1, wait, type(exc).__name__, exc,
        )
        time.sleep(wait)

    @staticmethod
    def _retryable_status(status: int) -> bool:
        """429 限流 / 5xx 服务端错误可重试。"""
        return status == 429 or status >= 500

    def _parse_content(self, data: Dict) -> str:
        """从非流式响应提取 content。"""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                "LLM 响应格式异常: {}".format(str(data)[:300])
            ) from e

    # ---- 对外（同步）----

    def generate(self, messages: List[Dict]) -> str:
        """同步调用 /chat/completions，返回完整文本。"""
        if self._sem is not None:
            self._sem.acquire()
        try:
            return self._do_generate(messages)
        finally:
            if self._sem is not None:
                self._sem.release()

    def _do_generate(self, messages: List[Dict]) -> str:
        t = time.time()
        logger.info(
            "LLM 调用开始: base_url=%s, model=%s, messages=%d, total_chars=%d",
            self.base_url, self.model, len(messages),
            sum(len(m.get("content", "")) for m in messages),
        )
        for attempt in range(self.retry_times + 1):
            try:
                resp = self._request(messages, stream=False)
            except Exception as e:  # 网络异常（连接/超时）
                if attempt >= self.retry_times:
                    raise RuntimeError(
                        "LLM 调用失败（重试耗尽）: {}".format(e)
                    ) from e
                self._wait_backoff(attempt, e)
                continue
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as e:
                    raise RuntimeError(
                        "LLM 响应非 JSON: {}".format(resp.text[:300])
                    ) from e
                content = self._parse_content(data)
                logger.info(
                    "LLM 生成成功: answer_len=%d, 耗时=%.3fs",
                    len(content), time.time() - t,
                )
                return content
            if self._retryable_status(resp.status_code):
                if attempt < self.retry_times:
                    self._wait_backoff(attempt, RuntimeError("HTTP {}".format(resp.status_code)))
                    continue
                raise RuntimeError(
                    "LLM 调用失败（重试耗尽）: HTTP {}: {}".format(
                        resp.status_code, resp.text[:300]
                    )
                )
            raise RuntimeError(
                "LLM HTTP {}: {}".format(resp.status_code, resp.text[:300])
            )

    # ---- 对外（流式）----

    def stream_generate(self, messages: List[Dict]) -> Iterator[str]:
        """流式调用 /chat/completions（SSE），逐 delta yield 文本。"""
        if self._sem is not None:
            self._sem.acquire()
        try:
            yield from self._do_stream_generate(messages)
        finally:
            if self._sem is not None:
                self._sem.release()

    def _do_stream_generate(self, messages: List[Dict]) -> Iterator[str]:
        t = time.time()
        logger.info(
            "LLM 流式调用开始: base_url=%s, model=%s, messages=%d",
            self.base_url, self.model, len(messages),
        )
        chunk_count = 0
        total_yielded = 0
        for attempt in range(self.retry_times + 1):
            try:
                resp = self._request(messages, stream=True)
            except Exception as e:  # 网络异常：首包前可重试
                if attempt >= self.retry_times:
                    raise RuntimeError(
                        "LLM 流式调用失败（重试耗尽）: {}".format(e)
                    ) from e
                self._wait_backoff(attempt, e)
                continue
            if resp.status_code != 200:
                if self._retryable_status(resp.status_code):
                    if attempt < self.retry_times:
                        self._wait_backoff(attempt, RuntimeError("HTTP {}".format(resp.status_code)))
                        continue
                    raise RuntimeError(
                        "LLM 流式调用失败（重试耗尽）: HTTP {}: {}".format(
                            resp.status_code, resp.text[:300]
                        )
                    )
                raise RuntimeError(
                    "LLM 流式 HTTP {}: {}".format(resp.status_code, resp.text[:300])
                )
            # 200：迭代 SSE，中途失败不重试（避免重复片段）
            try:
                for delta in self._iter_sse(resp):
                    if delta:
                        chunk_count += 1
                        total_yielded += len(delta)
                        yield delta
            finally:
                resp.close()
            logger.info(
                "LLM 流式生成完成: %.3fs, chunks=%d, total_len=%d",
                time.time() - t, chunk_count, total_yielded,
            )
            return

    @staticmethod
    def _iter_sse(resp) -> Iterator[str]:
        """解析 SSE 流：data: {...} 行 → choices[0].delta.content。"""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
            else:
                continue
            if payload == "[DONE]":
                return
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                yield delta
