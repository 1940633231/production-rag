import time
from typing import Dict, List

from app.search.rrf import rrf_fuse
from app.core.logger import get_logger

logger = get_logger(__name__)


class HybridSearch:
    """dense(向量) + sparse(BM25) 双路检索 + RRF 融合。

    接口与 Retriever / BM25Search 对齐：search(query, top_k) → List[dict]（含 offset）。
    内部先从两路各取更宽的候选池（candidate_k），再 RRF 融合后截断到 top_k。
    宽候选池是 hybrid 的关键：若只取 top_k 候选，融合空间过窄，互补性无从发挥。
    """

    def __init__(self, dense_retriever, sparse_retriever, rrf_k: int = 60):

        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, document_ids=None) -> List[Dict]:

        t = time.time()
        # 候选池放宽到 5×top_k（至少 20），给 RRF 足够融合空间
        candidate_k = max(top_k * 5, 20)

        logger.info(
            "Hybrid 检索开始: query=%r, top_k=%d, candidate_k=%d, rrf_k=%d",
            query, top_k, candidate_k, self.rrf_k,
        )

        dense_results = self.dense.search(query, top_k=candidate_k, document_ids=document_ids)
        sparse_results = self.sparse.search(query, top_k=candidate_k, document_ids=document_ids)

        logger.info(
            "Hybrid 两路检索完成: dense=%d, sparse=%d",
            len(dense_results), len(sparse_results),
        )

        # RRF 融合阶段计时
        t = time.time()
        fused = rrf_fuse([dense_results, sparse_results], k=self.rrf_k)

        from app.core.metrics import metrics
        metrics.record_rrf(
            getattr(self, "strategy", "unknown"), time.time() - t
        )

        result = fused[:top_k]

        logger.info(
            "Hybrid 检索完成: %.3fs, fused=%d, returned=%d",
            time.time() - t, len(fused), len(result),
        )
        return result
