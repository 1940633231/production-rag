import time
from typing import Dict, List

from sentence_transformers import CrossEncoder

from app.core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Cross-Encoder 重排器。

    与 Bi-encoder（双塔：query/doc 各编码一次后做余弦相似）不同，
    Cross-Encoder 把 (query, doc) 拼接送入 transformer 做联合 attention，
    精度显著更高，但复杂度 O(N) 且无法预编码，慢得多。

    工业标准用法：retrieve 宽候选池（如 top-50）→ rerank 截断到 top-5。
    这样兼顾召回（双塔快、覆盖广）与精度（cross-encoder 准）。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        t = time.time()
        logger.info("加载 reranker 模型: %s", model_name)
        self.model = CrossEncoder(model_name)
        logger.info("reranker 模型加载完成: %.3fs", time.time() - t)

    def rerank(self, query: str, results: List[Dict], top_k: int = None) -> List[Dict]:
        """对候选 results 按 query 重新打分排序。

        results: 检索器返回的候选列表（每个 dict 含 content 等字段）
        top_k: 截断长度；None 表示不截断
        return: 按 rerank_score 降序的结果列表（保留 offset 等字段）
        """
        if not results:
            logger.info("重排跳过: 候选为空")
            return []

        t = time.time()
        logger.info(
            "重排开始: candidates=%d, top_k=%s, query=%r",
            len(results), top_k, query,
        )

        # CrossEncoder.predict 接受 [(query, doc), ...]，返回每对的相关性分数
        pairs = [(query, r["content"]) for r in results]

        scores = self.model.predict(pairs)

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        logger.info(
            "重排完成: %.3fs, 结果数=%d, top_score=%.4f",
            time.time() - t, len(reranked),
            reranked[0]["rerank_score"] if reranked else 0,
        )
        return reranked
