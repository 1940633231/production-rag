import time
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from app.core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """Cross-Encoder 重排器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 256,
    ):
        t = time.time()

        logger.info("加载 reranker 模型: %s", model_name)

        # device:
        #   None -> SentenceTransformer 自动选择
        #   "cpu" -> 强制 CPU
        #   "cuda" -> 强制 GPU
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(
            "reranker 模型加载完成: %.3fs, device=%s, batch_size=%d, max_length=%d",
            time.time() - t,
            self.model.device,
            self.batch_size,
            self.max_length,
        )

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """对候选 results 按 query 重新打分排序。"""

        if not results:
            logger.info("重排跳过: 候选为空")
            return []

        t = time.time()

        logger.info(
            "重排开始: candidates=%d, top_k=%s, query=%r",
            len(results),
            top_k,
            query,
        )

        pairs = [
            (query, r["content"])
            for r in results
        ]

        # 一次 predict，内部按 batch 推理
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        reranked = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        elapsed = time.time() - t

        logger.info(
            "重排完成: %.3fs, %.1fms/candidate, 结果数=%d, top_score=%.4f",
            elapsed,
            elapsed / len(results) * 1000,
            len(reranked),
            reranked[0]["rerank_score"] if reranked else 0,
        )

        return reranked