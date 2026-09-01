"""Elasticsearch 全文检索后端：替代本地 BM25Search，复用 ES 索引做稀疏检索。

与 bm25_search.BM25Search 接口完全对齐（search(query, top_k) → List[Dict] 字段一致），
可直接作为 HybridSearch 的 sparse_retriever 注入，RRF 融合流程无需改动。

当 ES 不可用（es_client 初始化失败 / ping 失败 / 索引空）：
  __init__ 抛异常 → 上层 service.py 捕获后回退到本地 BM25Search。

输出字段（与 BM25Search / Retriever 对齐，RRF 识别用 vector_id/chunk_id）:
    {
        "rank": int,          # 当前查询内排名 0-based
        "score": float,       # ES _score
        "vector_id": int,     # 向量位置 id（与 FAISS/Milvus 对齐，RRF 融合主键）
        "chunk_id": str,
        "content": str,
        "start_offset": int,
        "end_offset": int,
        "metadata": dict,
    }
"""
import time
from typing import Dict, List

from app.core.logger import get_logger

logger = get_logger(__name__)


class ESFulltextSearch:
    """ES 全文检索：与 BM25Search 接口对齐的 sparse 检索器。"""

    def __init__(
        self,
        strategy: str,
        es_client=None,
        min_score: float = 0.0,
    ):
        """初始化。

        参数:
            strategy: 分块策略 fixed/recursive（对应 ES 索引名 {prefix}_{strategy}）
            es_client: 可选，外部传入 ESClient 实例（测试注入）；None 时新建
            min_score: 低于此分数的命中直接过滤（ES 零命中去噪，默认 0.0 不做额外过滤）
        """
        self.strategy = strategy
        self.min_score = min_score

        if es_client is not None:
            self._es = es_client
        else:
            from app.storage.es_client import ESClient

            self._es = ESClient()

        if not self._es.ping():
            raise RuntimeError("ES 服务不可达（ping 失败），请检查 ES_HOSTS/ES_USER/ES_PASSWORD")

        # 懒验证：索引需存在且有数据，否则提醒调用方（仍允许初始化，首次 search 会返回空）
        self._chunks_total = self._es.count(self.strategy)
        if self._chunks_total == 0:
            logger.warning(
                "ESFulltextSearch: strategy=%s 的索引为空（count=0），"
                "全文检索将无结果。请先运行 upload/rebuild 写入 ES。",
                self.strategy,
            )
        logger.info(
            "ESFulltextSearch 初始化完成: strategy=%s, index=%s, chunks=%d",
            self.strategy, self._es._index_name(self.strategy), self._chunks_total,
        )

    # ------------------------------------------------------------------
    # 对外接口（与 BM25Search / Retriever 一致）
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10, document_ids=None) -> List[Dict]:
        """ES match 查询，返回与 BM25Search 格式相同的命中列表。

        document_ids: 文档级 ACL 可读文档集合（None 表示不设文档级过滤）。
        """
        t = time.time()
        logger.info(
            "ES 全文检索开始: strategy=%s, query=%r, top_k=%d",
            self.strategy, query, top_k,
        )

        try:
            hits = self._es.search(
                strategy=self.strategy,
                query=query,
                top_k=top_k,
                sort_by_vector_id=False,  # 默认按 score 降序
                document_ids=document_ids,  # 先过滤后检索：ES 端 terms 预过滤
            )
        except Exception as e:
            logger.warning(
                "ES 全文检索异常（返回空结果）: strategy=%s, query=%r, error=%s: %s",
                self.strategy, query, type(e).__name__, e, exc_info=True,
            )
            return []

        results: List[Dict] = []
        for rank, h in enumerate(hits):
            score = float(h.get("score", 0) or 0)
            if score <= self.min_score:
                continue
            # 文档级 ACL 已在 ES 端 terms 预过滤，无需后置过滤
            vector_id = h.get("vector_id")
            if vector_id is None:
                # 缺少 vector_id 就无法参与 RRF + chunk_repo 映射，跳过
                logger.warning(
                    "ES 命中缺少 vector_id 字段，已跳过: chunk_id=%s",
                    h.get("chunk_id"),
                )
                continue
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "vector_id": int(vector_id),
                    "chunk_id": h.get("chunk_id", ""),
                    "content": h.get("content", ""),
                    "start_offset": int(h.get("start_offset", 0) or 0),
                    "end_offset": int(h.get("end_offset", 0) or 0),
                    "metadata": h.get("metadata", {}),
                }
            )

        logger.info(
            "ES 全文检索完成: %.3fs, 命中=%d, score_range=[%.4f, %.4f]",
            time.time() - t, len(results),
            results[0]["score"] if results else 0.0,
            results[-1]["score"] if results else 0.0,
        )
        return results

    # ------------------------------------------------------------------
    # 诊断辅助
    # ------------------------------------------------------------------

    @property
    def indexed_chunks(self) -> int:
        """返回 ES 索引中当前的 chunk 总数。"""
        return self._chunks_total
