"""LLM 生成器：可插拔接口 + 默认 Stub + 工厂函数。

设计要点（与 tokenizer.py 一致的可插拔方案）：
- BaseGenerator 抽象接口（含同步 generate 与流式 stream_generate）
- StubGenerator: 零依赖占位，返回固定文本，教学用（默认）
- QwenGenerator: 通过 DashScope API 调用 Qwen（可选，需 dashscope + API key）
- OpenAICompatibleGenerator: 任意 OpenAI 兼容 /chat/completions 端点
  （OpenAI / DeepSeek / Kimi / GLM / 百炼 compatible-mode / vLLM / Ollama ...），
  解除 DashScope 绑定，只需 base_url + api_key + model
- 工厂函数 create_generator(backend) 便于 Config 切换
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
            "如需真实生成，请配置 generation.backend=qwen（DashScope）或 "
            "generation.backend=openai（任意 OpenAI 兼容端点）。\n"
            "User prompt 前 50 字: {}".format(query[:50])
        )


def create_generator(backend: str = "stub", **kwargs) -> BaseGenerator:
    """工厂函数：按 backend 创建生成器。

    backend:
        - "stub"  → StubGenerator（默认，零依赖）
        - "qwen"  → QwenGenerator（DashScope，需 dashscope + DASHSCOPE_API_KEY）
        - "openai" → OpenAICompatibleGenerator（任意 OpenAI 兼容端点，
          需 base_url + api_key，如 OPENAI_API_KEY / DEEPSEEK_API_KEY 等）
    """
    if backend == "stub":
        return StubGenerator()
    if backend == "qwen":
        from app.generation.qwen import QwenGenerator
        return QwenGenerator(**kwargs)
    if backend == "openai":
        from app.generation.openai_compat import OpenAICompatibleGenerator
        return OpenAICompatibleGenerator(**kwargs)
    raise ValueError("Unknown generator backend: {}".format(backend))
