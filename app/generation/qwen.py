"""Qwen LLM 生成器：通过 DashScope API 调用。

需要:
- pip install dashscope
- 环境变量 DASHSCOPE_API_KEY（或构造时传入 api_key）

调用方式兼容 OpenAI messages 格式：[{role, content}, ...]
模型选择参考：
- qwen-turbo: 快速、低成本，适合简单问答
- qwen-plus: 平衡质量与成本
- qwen-max: 最高质量，复杂推理

流式支持:
- stream_generate(messages) 调用 DashScope 流式接口，逐 token yield 文本片段
- 用于 SSE 端点真实流式输出，首字延迟≈模型开始输出时间
"""
import os
import time
from typing import Dict, Iterator, List, Optional

from app.core.logger import get_logger
from app.generation.generator import BaseGenerator

logger = get_logger(__name__)


class QwenGenerator(BaseGenerator):
    """通过阿里云 DashScope 调用 Qwen 系列 LLM。"""

    def __init__(
        self,
        model: str = "qwen-turbo",
        api_key: Optional[str] = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            "QwenGenerator 初始化: model=%s, temperature=%.2f, max_tokens=%d",
            model, temperature, max_tokens,
        )

        # API key：构造参数 > 环境变量
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            logger.error("API key 缺失: 环境变量 %s 未设置", api_key_env)
            raise ValueError(
                "QwenGenerator 需要 API key：设置环境变量 {} 或构造时传入 api_key".format(
                    api_key_env
                )
            )

        # 延迟 import，未装 dashscope 时给清晰错误
        try:
            import dashscope
        except ImportError as e:
            logger.error("dashscope 未安装: pip install dashscope")
            raise ImportError(
                "QwenGenerator 需要 dashscope: pip install dashscope"
            ) from e
        self._dashscope = dashscope
        logger.info("QwenGenerator 初始化完成, dashscope 版本=%s", getattr(dashscope, "__version__", "unknown"))

    def generate(self, messages: List[Dict]) -> str:
        """调用 DashScope Generation API 生成回复。

        参数:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}]

        返回:
            LLM 生成的文本
        """
        msg_count = len(messages)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        t = time.time()

        logger.info(
            "DashScope API 调用开始: model=%s, messages=%d, total_chars=%d, "
            "temperature=%.2f, max_tokens=%d",
            self.model, msg_count, total_chars, self.temperature, self.max_tokens,
        )

        try:
            response = self._dashscope.Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",  # 返回 OpenAI 兼容格式
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
            )
        except Exception as e:
            logger.error(
                "DashScope API 网络异常: %.3fs, error=%s",
                time.time() - t, e, exc_info=True,
            )
            raise

        api_elapsed = time.time() - t
        status = getattr(response, "status_code", "unknown")
        logger.info(
            "DashScope API 响应: %.3fs, status=%s",
            api_elapsed, status,
        )

        # DashScope SDK 返回对象含 status_code 字段
        if hasattr(response, "status_code") and response.status_code != 200:
            err_msg = getattr(response, "message", "unknown")
            logger.error(
                "DashScope API 错误: status=%s, message=%s, 耗时=%.3fs",
                status, err_msg, api_elapsed,
            )
            raise RuntimeError(
                "DashScope API error: {} {}".format(
                    response.status_code, err_msg
                )
            )

        # 兼容格式：response.output.choices[0].message.content
        try:
            content = response.output.choices[0].message.content
            logger.info(
                "生成成功: answer_len=%d, API耗时=%.3fs",
                len(content), api_elapsed,
            )
            return content
        except (AttributeError, IndexError, KeyError) as e:
            logger.error(
                "响应解析失败: %s, raw_response=%s", e, response, exc_info=True
            )
            raise RuntimeError(
                "DashScope 响应格式异常: {}".format(response)
            ) from e

    def stream_generate(self, messages: List[Dict]) -> Iterator[str]:
        """流式调用 DashScope Generation API，逐片段 yield 文本。

        用于 SSE 端点真实流式输出，避免等待整段生成完成。
        DashScope 流式返回的每条 response 含增量文本，通过
        response.output.choices[0].message.content 获取。

        参数:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}]

        yields:
            str: 增量文本片段（非累积，需由调用方拼接）
        """
        msg_count = len(messages)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        t = time.time()
        logger.info(
            "DashScope 流式 API 调用开始: model=%s, messages=%d, total_chars=%d",
            self.model, msg_count, total_chars,
        )

        try:
            responses = self._dashscope.Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                stream=True,  # 启用流式
                incremental_output=True,  # 增量输出，每个 response 只含新片段
            )
        except Exception as e:
            logger.error(
                "DashScope 流式 API 网络异常: %.3fs, error=%s",
                time.time() - t, e, exc_info=True,
            )
            raise

        chunk_count = 0
        total_yielded = 0
        for response in responses:
            status = getattr(response, "status_code", None)
            if status is not None and status != 200:
                err_msg = getattr(response, "message", "unknown")
                logger.error(
                    "DashScope 流式 API 错误: status=%s, message=%s",
                    status, err_msg,
                )
                raise RuntimeError(
                    "DashScope stream error: {} {}".format(status, err_msg)
                )
            try:
                delta = response.output.choices[0].message.content
                if delta:
                    chunk_count += 1
                    total_yielded += len(delta)
                    yield delta
            except (AttributeError, IndexError, KeyError) as e:
                logger.warning("流式响应片段解析失败: %s", e)

        logger.info(
            "DashScope 流式生成完成: %.3fs, chunks=%d, total_len=%d",
            time.time() - t, chunk_count, total_yielded,
        )
