"""Context Manager：编排去重、合并、压缩、排序、Token 预算控制。

作为 Reranker 与 LLM 之间的独立层，解决生产 RAG 的三大问题：
1. Token 超限：Top-N chunks 可能远超模型 context window，必须控制
2. Chunk 重复：检索召回高度重叠的 chunk，需要去重
3. 相邻 Chunk：同文档相邻 chunk 可合并，减少碎片、提升连贯性

流水线：
  Reranker 输出
    → deduplicate       # 去重（span 重叠为主信号）
    → merge_neighbors   # 邻接合并（chunk_index 连续或 span 贴合）
    → compress          # 句子窗口压缩（防单 chunk 吃满）
    → order             # 排序（score / interleaved）
    → fit_budget        # Token 预算贪心装填（尾部截断兜底）
    → 最终 context
"""
import time
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.ingestion.tokenizer import BaseTokenCounter, create_token_counter
from app.context.builder import ContextBuilder
from app.context.compressor import ContextCompressor

logger = get_logger(__name__)


class ContextManager:
    """Context 管理器：编排 builder + compressor + token 预算硬约束。"""

    def __init__(
        self,
        token_counter: Optional[BaseTokenCounter] = None,
        max_context_tokens: int = 4096,
        reserved_tokens: int = 1024,
        builder: Optional[ContextBuilder] = None,
        compressor: Optional[ContextCompressor] = None,
        order_strategy: str = "score",
    ):
        self.token_counter = token_counter or create_token_counter("char")
        self.max_context_tokens = max_context_tokens
        self.reserved_tokens = reserved_tokens
        self.builder = builder or ContextBuilder()
        self.compressor = compressor or ContextCompressor(self.token_counter)
        self.order_strategy = order_strategy

    def build(self, query: str, results: List[Dict]) -> Dict:
        """编排完整 context 构建流程。

        参数:
            query: 用户查询（用于计算 query 占用 token，从预算中扣除）
            results: Reranker 输出的候选列表，每个 dict 含
                     content/start_offset/end_offset/metadata/score或rerank_score

        返回:
            {
                "context": str,          # 最终拼装的上下文文本（带 [1][2] 编号）
                "chunks": List[Dict],   # 最终保留的 chunks（含元信息，供 citation 用）
                "stats": Dict,          # 各阶段统计（输入/去重/合并/压缩/预算）
            }
        """
        build_start = time.time()
        stats = {
            "input_count": len(results),
            "input_tokens": sum(
                self.token_counter.count(r["content"]) for r in results
            ),
        }
        logger.info(
            "ContextManager 开始: input_count=%d, input_tokens=%d",
            stats["input_count"], stats["input_tokens"],
        )

        # 1. 去重：同文档 span 重叠为主信号，跨文档 Jaccard 兜底
        t = time.time()
        deduped = self.builder.deduplicate(results)
        stats["after_dedup"] = len(deduped)
        logger.info(
            "去重完成: %.3fs, %d → %d（去除 %d 条）",
            time.time() - t, len(results), len(deduped), len(results) - len(deduped),
        )

        # 2. 邻接合并：同 document_id 且 chunk_index 连续或 span 贴合
        t = time.time()
        merged = self.builder.merge_neighbors(deduped)
        stats["after_merge"] = len(merged)
        logger.info(
            "邻接合并完成: %.3fs, %d → %d（合并 %d 条）",
            time.time() - t, len(deduped), len(merged), len(deduped) - len(merged),
        )

        # 3. 计算可用 token 预算（扣除 query + prompt + answer 预留）
        query_tokens = self.token_counter.count(query)
        available = max(
            0, self.max_context_tokens - self.reserved_tokens - query_tokens
        )
        stats["query_tokens"] = query_tokens
        stats["available_tokens"] = available
        logger.info(
            "预算计算: max=%d, reserved=%d, query_tokens=%d, available=%d",
            self.max_context_tokens, self.reserved_tokens, query_tokens, available,
        )

        # 4. 压缩：per-chunk cap = 均分预算，防止单 chunk 吃满
        t = time.time()
        n = max(len(merged), 1)
        per_chunk_cap = available // n if n > 0 else available
        compressed = self.compressor.compress_all(merged, per_chunk_cap)
        stats["compressed_count"] = sum(
            1 for c in compressed if c.get("compressed")
        )
        logger.info(
            "压缩完成: %.3fs, per_chunk_cap=%d, compressed_count=%d",
            time.time() - t, per_chunk_cap, stats["compressed_count"],
        )

        # 5. 排序：score（默认保留 rerank 序）/ document / interleaved
        t = time.time()
        ordered = self.builder.order(compressed, self.order_strategy)
        logger.info(
            "排序完成: %.3fs, strategy=%s",
            time.time() - t, self.order_strategy,
        )

        # 6. TokenBudget 贪心装填：按排序顺序累加，超出则截断最后一个
        t = time.time()
        final_chunks, used = self._fit_budget(ordered, available)
        stats["final_count"] = len(final_chunks)
        stats["used_tokens"] = used
        stats["budget_utilization"] = (
            used / available if available > 0 else 0.0
        )
        logger.info(
            "预算装填完成: %.3fs, final=%d, used=%d, utilization=%.2f%%",
            time.time() - t, len(final_chunks), used,
            stats["budget_utilization"] * 100,
        )

        # 7. 拼装 context 文本（带 chunk 编号，便于 citation 引用）
        context_text = self._format_context(final_chunks)

        logger.info(
            "ContextManager 完成: 总耗时=%.3fs, context_len=%d",
            time.time() - build_start, len(context_text),
        )

        return {
            "context": context_text,
            "chunks": final_chunks,
            "stats": stats,
        }

    def _fit_budget(self, chunks: List[Dict], available: int):
        """贪心装填到 available token 预算内。

        按排序顺序逐个累加；最后一个装不下的，按剩余预算截断尾部。
        compress 已保证单 chunk <= per_chunk_cap，这里兜底防总和超限。
        """
        final: List[Dict] = []
        used = 0
        for chunk in chunks:
            chunk_tokens = self.token_counter.count(chunk["content"])
            if used + chunk_tokens <= available:
                final.append(chunk)
                used += chunk_tokens
            else:
                # 装不下的最后一个，按剩余预算截断
                remaining = available - used
                if remaining > 0:
                    truncated = self.compressor._truncate_to_tokens(
                        chunk["content"], remaining
                    )
                    if truncated:
                        new_chunk = {**chunk}
                        new_chunk["content"] = truncated
                        new_chunk["budget_truncated"] = True
                        final.append(new_chunk)
                        used += self.token_counter.count(truncated)
                        logger.info(
                            "预算截断: chunk_id=%s, remaining=%d, truncated_len=%d",
                            chunk.get("chunk_id", "unknown"), remaining, len(truncated),
                        )
                break  # 预算用完，停止
        return final, used

    def _format_context(self, chunks: List[Dict]) -> str:
        """拼装最终 context 文本，带 chunk 编号（便于 citation 引用）。"""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append("[{}] {}".format(i, c["content"]))
        return "\n\n".join(parts)
