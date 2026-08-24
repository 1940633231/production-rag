"""Token 计数器。

生产 RAG 中 Token 预算管理依赖准确的 token 计数。
本模块提供可插拔的计数后端，便于后续切换到真实 tokenizer：

- "char"    : 字符长度近似（默认，零依赖，教学友好）
- "tiktoken": OpenAI cl100k_base BPE（中英文混合的合理近似，需 pip install tiktoken）

通过 Config.context.tokenizer_backend 切换。
后续若接入 Qwen 真实 tokenizer，新增一个 BaseTokenCounter 子类即可。
"""
from abc import ABC, abstractmethod


class BaseTokenCounter(ABC):
    """Token 计数器抽象接口。"""

    @abstractmethod
    def count(self, text: str) -> int:
        """返回 text 的 token 数。"""
        raise NotImplementedError


class CharLengthCounter(BaseTokenCounter):
    """字符长度近似。

    对中文场景偏粗略（一个汉字≈1 token，一个英文单词≈1-2 token），
    但零依赖、可立刻跑通。教学阶段默认使用，保证 pipeline 可运行。
    """

    def count(self, text: str) -> int:
        return len(text)


class TiktokenCounter(BaseTokenCounter):
    """tiktoken (cl100k_base) 计数。

    对 GPT-3.5/4 系列贴近真实分词；对 Qwen 等国产模型为合理近似。
    需要 pip install tiktoken，未安装时抛出明确错误。
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "TiktokenCounter 需要 `tiktoken`，请安装: pip install tiktoken"
            ) from e
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def create_token_counter(backend: str = "char") -> BaseTokenCounter:
    """工厂函数：按 backend 名称创建计数器。

    backend:
        - "char"     → CharLengthCounter（默认）
        - "tiktoken" → TiktokenCounter
    """
    if backend == "char":
        return CharLengthCounter()
    if backend == "tiktoken":
        return TiktokenCounter()
    raise ValueError(f"Unknown tokenizer backend: {backend}")
