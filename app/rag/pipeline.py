"""RAG Pipeline：编排 Retriever → Reranker → ContextManager → LLM Generator 检索链路。

职责边界：
- 只编排已注入的组件，不负责初始化（由 RAGService 负责）
- LLM 生成可选：有 generator 则填充 answer，无则 answer=None
- 不依赖具体检索器/生成器实现（任意实现 search/generate 接口的对象均可）

流水线：
  query
    → Retriever.search（rerank 启用时取宽候选池，否则直接取 top_k）
    → Reranker.rerank（可选，Cross-Encoder 重排截断）
    → ContextManager.build（去重/合并/压缩/排序/预算控制）
    → PromptBuilder.build（组装 system + user messages）
    → Generator.generate（可选，LLM 生成答案）
    → RAGResponse（context + chunks + stats + answer）
"""
import time
from typing import Dict, Iterator, List, Optional

from app.core.logger import get_logger
from app.citation.citation import CitationExtractor
from app.generation.prompt import PromptBuilder
from app.rag.result import RAGResponse

logger = get_logger(__name__)


class RAGPipeline:
    """RAG 检索链路编排器。

    所有组件均可选注入，便于测试和逐步接入：
    - retriever: 必需（vector/bm25/hybrid 任意实现 search(query, top_k) 接口的对象）
    - reranker: 可选（None 表示跳过重排）
    - context_manager: 可选（None 表示 fallback 直接拼接原始结果）
    - generator: 可选（None 表示 answer 为 None，不调 LLM）
    - prompt_builder: 可选（默认 PromptBuilder，可自定义 system prompt）
    """

    def __init__(
        self,
        retriever,
        reranker=None,
        context_manager=None,
        generator=None,
        prompt_builder=None,
        query_rewriter=None,
        multi_query_expander=None,
        top_k: int = 5,
        rerank_candidate_pool: int = 50,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_manager = context_manager
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.query_rewriter = query_rewriter
        self.multi_query_expander = multi_query_expander
        self.top_k = top_k
        self.rerank_candidate_pool = rerank_candidate_pool

    def _rewrite_query(self, query: str, history: Optional[List] = None):
        """有 history 且有 query_rewriter 时改写查询，返回 (effective_query, changed)。"""
        if not history or self.query_rewriter is None:
            return query, False
        t = time.time()
        rewritten = self.query_rewriter.rewrite(query, history)
        logger.info(
            "查询改写: %.3fs, %r → %r",
            time.time() - t, query[:50], rewritten[:50],
        )
        return rewritten, rewritten != query

    def _retrieve_and_context(self, query: str):
        """执行 multi-query → retrieve → rerank → context_manager。

        返回 (context, chunks, stats)。供 run() 和 run_stream() 复用。
        同时记录检索/重排阶段 Prometheus 指标。
        """
        from app.core.metrics import metrics

        # 推断 strategy（用于指标标签）
        strategy = getattr(self.retriever, "strategy", "unknown")

        # 0. Multi-Query 扩展（多路召回）
        queries = [query]
        if self.multi_query_expander is not None:
            t = time.time()
            queries = self.multi_query_expander.expand(query)
            logger.info(
                "多路召回: %.3fs, %d 路查询: %r",
                time.time() - t, len(queries), [q[:30] for q in queries],
            )

        # 1. 多路检索 + 合并去重
        t = time.time()
        candidates = self._search_all(query, queries)
        metrics.record_retrieval(strategy, time.time() - t)

        # 1b. 重排（整体重排，query 用主查询）
        if self.reranker is not None:
            t = time.time()
            results = self.reranker.rerank(query, candidates, top_k=self.top_k)
            metrics.record_rerank(time.time() - t)
        else:
            # 未启用 rerank：多路合并后按 score 降序取 top_k
            results = sorted(
                candidates, key=lambda r: r.get("score", 0.0), reverse=True
            )[:self.top_k]

        # 2. ContextManager
        if self.context_manager is not None:
            ctx_result = self.context_manager.build(query, results)
            ctx_result["stats"]["query_count"] = len(queries)
            ctx_result["stats"]["merged_candidates"] = len(candidates)
            return ctx_result["context"], ctx_result["chunks"], ctx_result["stats"]

        # Fallback
        context = "\n\n".join(
            "[{}] {}".format(i + 1, r["content"]) for i, r in enumerate(results)
        )
        return context, results, {
            "input_count": len(results),
            "query_count": len(queries),
            "merged_candidates": len(candidates),
        }

    def _search_all(self, query: str, queries: List[str]) -> List[Dict]:
        """对每路子查询检索并合并去重（按 chunk_id）。

        每路候选数：rerank 启用时均分 rerank_candidate_pool（合并后总量≈pool），
        未启用时每路取 top_k。合并结果供重排或按分数排序。
        """
        if len(queries) <= 1:
            topk = (
                self.rerank_candidate_pool
                if self.reranker is not None
                else self.top_k
            )
            return self.retriever.search(query, top_k=topk)

        per_query = (
            self.rerank_candidate_pool // len(queries)
            if self.reranker is not None
            else self.top_k
        )
        per_query = max(per_query, 1)
        merged: List[Dict] = []
        seen = set()
        for q in queries:
            for r in self.retriever.search(q, top_k=per_query):
                cid = r.get("chunk_id")
                if cid is None or cid not in seen:
                    seen.add(cid)
                    merged.append(r)
        logger.info("多路检索合并: %d 路 → %d 候选", len(queries), len(merged))
        return merged

    def run(self, query: str, history: Optional[List] = None) -> RAGResponse:
        """执行 retrieve → rerank → context_manager → generator 链路。

        history: 多轮对话历史 [{role, content}, ...]，透传给 LLM 理解指代（默认 None）
        """
        logger.info("pipeline 开始: query=%r, history_turns=%d", query, len(history or []))
        pipeline_start = time.time()

        # 0. Query 改写（多轮指代消解）
        effective_query, rewritten = self._rewrite_query(query, history)

        # 1+2: 检索 + ContextManager（用改写后的 query）
        context, chunks, stats = self._retrieve_and_context(effective_query)
        if rewritten:
            stats = dict(stats)
            stats["rewritten_query"] = effective_query

        # 3. 生成：有 generator 则构建 prompt 并调用 LLM，填充 answer
        answer = None
        if self.generator is not None:
            t = time.time()
            generator_name = type(self.generator).__name__
            logger.info(
                "LLM 生成开始: backend=%s, context_len=%d",
                generator_name, len(context),
            )
            try:
                messages = self.prompt_builder.build(
                    effective_query, context, history=history
                )
                answer = self.generator.generate(messages)
                # 记录生成耗时指标
                from app.core.metrics import metrics
                metrics.record_generation(generator_name, time.time() - t)
            except Exception as e:
                logger.error(
                    "LLM 生成失败: %.3fs, backend=%s, error=%s",
                    time.time() - t, generator_name, e, exc_info=True,
                )
                raise
            logger.info(
                "LLM 生成完成: %.3fs, answer_len=%d",
                time.time() - t, len(answer or ""),
            )

        logger.info(
            "pipeline 完成: 总耗时=%.3fs, answer=%s",
            time.time() - pipeline_start,
            "已生成" if answer else "未生成",
        )

        # 4. Citation 提取：从 answer 解析 [1][2] 引用，映射到 chunks
        citations = []
        if answer:
            t = time.time()
            citations = CitationExtractor().extract(answer, chunks)
            logger.info(
                "Citation 提取: %.3fs, %d 条引用",
                time.time() - t, len(citations),
            )

        return RAGResponse(
            query=query,
            context=context,
            chunks=chunks,
            stats=stats,
            answer=answer,
            citations=citations,
        )

    def run_stream(self, query: str, history: Optional[List] = None) -> Iterator[dict]:
        """流式执行 RAG 链路，逐事件 yield。

        history: 多轮对话历史 [{role, content}, ...]，透传给 LLM 理解指代（默认 None）

        事件类型（dict 含 "type" 字段）:
          - {"type": "meta", "chunks": [...], "stats": {...}}      检索+上下文阶段完成
          - {"type": "delta", "content": str}                       LLM 增量文本片段
          - {"type": "citations", "citations": [...]}              引用提取完成
          - {"type": "done", "answer_length": int}                 生成结束
          - {"type": "error", "error": str}                        异常

        当 generator 为 None（未接入 LLM）时，跳过 delta/citations 事件。
        """
        logger.info("pipeline stream 开始: query=%r", query)
        pipeline_start = time.time()

        try:
            # 0. Query 改写（多轮指代消解）
            effective_query, rewritten = self._rewrite_query(query, history)

            # 1+2: 检索 + ContextManager（用改写后的 query）
            context, chunks, stats = self._retrieve_and_context(effective_query)
            if rewritten:
                stats = dict(stats)
                stats["rewritten_query"] = effective_query
            yield {"type": "meta", "chunks": chunks, "stats": stats}

            # 3. 流式生成
            if self.generator is None:
                logger.info("未配置 generator，跳过流式生成")
                yield {"type": "done", "answer_length": 0}
                return

            messages = self.prompt_builder.build(
                effective_query, context, history=history
            )
            answer_parts = []
            try:
                for delta in self.generator.stream_generate(messages):
                    if delta:
                        answer_parts.append(delta)
                        yield {"type": "delta", "content": delta}
            except Exception as e:
                logger.error("LLM 流式生成失败: %s", e, exc_info=True)
                yield {"type": "error", "error": "LLM 流式生成失败: {}".format(e)}
                return

            answer = "".join(answer_parts)
            logger.info(
                "LLM 流式生成完成: answer_len=%d, 总耗时=%.3fs",
                len(answer), time.time() - pipeline_start,
            )

            # 4. Citation 提取
            citations = CitationExtractor().extract(answer, chunks) if answer else []
            yield {"type": "citations", "citations": citations}

            yield {"type": "done", "answer_length": len(answer)}
        except Exception as e:
            logger.error(
                "pipeline stream 失败: %.3fs, error=%s",
                time.time() - pipeline_start, e, exc_info=True,
            )
            yield {"type": "error", "error": "pipeline stream 失败: {}".format(e)}
