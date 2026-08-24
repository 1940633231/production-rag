import time
from typing import Dict, List

import jieba
from rank_bm25 import BM25Okapi

from app.core.logger import get_logger

logger = get_logger(__name__)


class BM25Search:
    """基于 rank_bm25 的稀疏检索器。

    与 app.rag.retriever.Retriever 接口对齐：
    - 输入：BaseChunkRepository 实例（通过 list_all() 获取全量 chunks）
    - search(query, top_k) 返回结构与 Retriever.search 一致（含 start_offset/end_offset）
    便于在 span 评测中与向量检索直接互换对比。
    """

    def __init__(self, chunk_repo):

        t = time.time()
        logger.info("BM25 初始化开始")

        self.chunk_repo = chunk_repo

        # list_all() 返回按向量位置排序的 chunk 列表，index 即为 vector_id
        self.chunks = chunk_repo.list_all()

        logger.info("BM25 初始化开始: chunks=%d", len(self.chunks))

        # 对每个 chunk 内容做中文分词，构建 BM25 语料
        corpus_tokens = [
            list(jieba.cut(c["content"])) for c in self.chunks
        ]

        self.bm25 = BM25Okapi(corpus_tokens)

        logger.info("BM25 初始化完成: %.3fs, 语料=%d", time.time() - t, len(corpus_tokens))

    def search(self, query: str, top_k: int = 10) -> List[Dict]:

        t = time.time()
        logger.info("BM25 检索开始: query=%r, top_k=%d", query, top_k)

        query_tokens = list(jieba.cut(query))

        scores = self.bm25.get_scores(query_tokens)

        # 按分数降序取 top_k；local_idx 即为 vector_id（list_all 顺序）
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []

        for rank, (local_idx, score) in enumerate(ranked):

            # BM25 对零命中 doc 返回 0 分，过滤掉避免噪声
            if score <= 0:
                continue

            document = self.chunks[local_idx]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "vector_id": local_idx,
                    "chunk_id": document["chunk_id"],
                    "content": document["content"],
                    "start_offset": document.get("start_offset", 0),
                    "end_offset": document.get("end_offset", 0),
                    "metadata": document["metadata"],
                }
            )

        logger.info(
            "BM25 检索完成: %.3fs, 结果数=%d, top_score=%.4f",
            time.time() - t, len(results),
            results[0]["score"] if results else 0,
        )
        return results
