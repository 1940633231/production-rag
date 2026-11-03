"""LLM 生成器：可插拔接口 + 默认 Stub + 工厂函数。

设计要点（与 tokenizer.py 一致的可插拔方案）：
- BaseGenerator 抽象接口（含同步 generate 与流式 stream_generate）
- StubGenerator: 零依赖占位，返回固定文本，教学用（默认）
- LLMGenerator: 通过统一的 /chat/completions 协议调用真实模型
  （OpenAI / DeepSeek / Kimi / GLM / vLLM / Ollama ...）
- 工厂函数 create_generator(backend) 负责实例化
- create_generator_from_config(config) 统一从 Config 组装参数
"""
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List


class BaseGenerator(ABC):
    """LLM 生成器抽象接口。"""

    @abstractmethod
    def generate(self, messages: List[Dict]) -> str:
        """输入 messages，返回生成的完整文本（同步）。"""
        raise NotImplementedError

    def stream_generate(self, messages: List[Dict]) -> Iterator[str]:
        """输入 messages，逐片段 yield 生成文本（流式）。

        默认实现：调用 generate() 后一次性 yield 整段，保证后端不支持流式时仍可调用。
        子类若支持真流式，应重写此方法。
        """
        yield self.generate(messages)


class StubGenerator(BaseGenerator):
    """零依赖占位生成器。

    不调用真实 LLM，返回固定提示文本，教学阶段默认使用。
    保证 pipeline 可运行，无需 API key 或 SDK。
    """

    def generate(self, messages: List[Dict]) -> str:
        # 提取 user 消息中的问题，便于调试
        query = ""
        for m in messages:
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        return (
            "[StubGenerator] 未接入真实 LLM。\n"
            "已构建 prompt（system + user），context 已完成去重/合并/压缩/预算控制。\n"
            "如需真实生成，请配置 generation.backend=llm 和 generation.llm。\n"
            "User prompt 前 50 字: {}".format(query[:50])
        )


def create_generator(backend: str = "stub", **kwargs) -> BaseGenerator:
    """工厂函数：按 backend 创建生成器。

    backend:
        - "stub" → StubGenerator（默认，零依赖）
        - "llm"  → LLMGenerator（OpenAI-compatible /chat/completions）
    """
    if backend == "stub":
        return StubGenerator()
    if backend == "llm":
        from app.generation.llm import LLMGenerator
        return LLMGenerator(**kwargs)
    raise ValueError("Unknown generator backend: {}".format(backend))


def create_generator_from_config(config, **overrides) -> BaseGenerator:
    """按 Config 创建生成器，供主链路、Scope、评测等所有入口复用。

    overrides 最后合并，便于评测 Judge 只覆盖 temperature/max_tokens 等参数。
    """
    backend = config.generation_backend
    common = {
        "temperature": config.generation_temperature,
        "max_tokens": config.generation_max_tokens,
        "timeout": config.generation_timeout,
        "retry_times": config.generation_retry_times,
        "retry_backoff": config.generation_retry_backoff,
        "max_concurrency": config.generation_max_concurrency,
    }

    if backend == "llm":
        common.update({
            "model": config.generation_llm_model,
            "base_url": config.generation_llm_base_url,
            "api_key_env": config.generation_llm_api_key_env,
            "require_api_key": config.generation_llm_require_api_key,
        })

    common.update(overrides)
    if backend == "llm":
        if not common.get("base_url"):
            raise ValueError(
                "LLM base_url 未配置：请设置环境变量 {}".format(
                    config.generation_llm_config.get(
                        "base_url_env", "LLM_BASE_URL"
                    )
                )
            )
        if not common.get("model"):
            raise ValueError(
                "LLM 模型名未配置：请设置环境变量 {}".format(
                    config.generation_llm_config.get("model_env", "LLM_MODEL")
                )
            )
    return create_generator(backend, **common)
