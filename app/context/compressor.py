"""Context Compressor：句子窗口截断压缩。

在 TokenBudget 硬约束前，对单个过长的 chunk 进行压缩，避免一个 chunk 吃掉整个预算。

策略（对应方案 5A，纯规则、零 LLM 成本）：
- 句子级窗口提取：按句末标点切分，从头贪心装入句子，直到达到预算
- 尾部截断兜底：若单个句子仍超预算，按字符二分查找截断到预算内

不接 LLM 摘要（方案 5B），留待 generation 模块补齐后再扩展。
"""
import re
from typing import Dict, List

from app.ingestion.tokenizer import BaseTokenCounter


class ContextCompressor:
    """句子窗口截断压缩器。"""

    # 句子模式：非标点串 + 标点串，或末尾无标点的串
    _SENT_PATTERN = re.compile(r'[^。！？；\n.!?]*[。！？；\n.!?]+|[^。！？；\n.!?]+$')

    def __init__(self, token_counter: BaseTokenCounter):
        self.token_counter = token_counter

    def compress(self, chunk: Dict, max_tokens: int) -> Dict:
        """压缩单个 chunk 到 max_tokens 以内。

        - 若 content token 数 <= max_tokens，原样返回（不加额外字段）。
        - 若超出，按句子从头贪心装入，尾部截断兜底；结果带 compressed 标记。
        """
        content = chunk["content"]
        total = self.token_counter.count(content)
        if total <= max_tokens:
            return chunk

        sentences = self._split_sentences(content)
        kept: List[str] = []
        used = 0
        for s in sentences:
            n = self.token_counter.count(s)
            if used + n > max_tokens:
                # 当前句子装不下
                if not kept:
                    # 兜底：至少保留这个句子的前 (max_tokens - used) 部分
                    kept.append(self._truncate_to_tokens(s, max_tokens - used))
                break
            kept.append(s)
            used += n

        compressed = {**chunk}
        compressed["content"] = "".join(kept)
        compressed["compressed"] = True
        compressed["original_tokens"] = total
        compressed["compressed_tokens"] = self.token_counter.count(compressed["content"])
        return compressed

    def compress_all(self, chunks: List[Dict], max_tokens_per_chunk: int) -> List[Dict]:
        """批量压缩。"""
        return [self.compress(c, max_tokens_per_chunk) for c in chunks]

    def _split_sentences(self, text: str) -> List[str]:
        """按句末标点或换行切分，保留标点在前一段。"""
        matches = self._SENT_PATTERN.findall(text)
        return [m for m in matches if m.strip()] or [text]

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """按字符二分查找，截断到 token 数 <= max_tokens。"""
        if max_tokens <= 0:
            return ""
        if self.token_counter.count(text) <= max_tokens:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.token_counter.count(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]
