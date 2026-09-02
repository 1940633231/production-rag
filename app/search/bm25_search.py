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

        # 先过滤后检索：document_id → 语料索引列表，
        # search 时用 get_batch_scores 只对可读文档打分
        self._doc_indexes = {}
        for i, c in enumerate(self.chunks):
            doc = c.get("document_id")
            if doc:
                self._doc_indexes.setdefault(doc, []).append(i)

        logger.info("BM25 初始化完成: %.3fs, 语料=%d", time.time() - t, len(corpus_tokens))

    def search(self, query: str, top_k: int = 10, document_ids=None) -> List[Dict]:

        t = time.time()
        logger.info("BM25 检索开始: query=%r, top_k=%d", query, top_k)

        query_tokens = list(jieba.cut(query))

        # BM25 打分排序阶段计时
        st = time.time()
        if document_ids is not None:
            # 先过滤后检索：只对可读文档的 chunk 打分，
            # 不可读文档不参与排序，不会占用 top_k 名额。
            allowed_indexes = []
            for doc_id in document_ids:
                allowed_indexes.extend(self._doc_indexes.get(doc_id, ()))
            logger.info(
                "BM25 预过滤: 可读文档=%d, 可打分 chunk=%d",
                len(document_ids), len(allowed_indexes),
            )
            scores = self.bm25.get_batch_scores(query_tokens, allowed_indexes)
            scored = list(zip(allowed_indexes, scores))
        else:
            scores = self.bm25.get_scores(query_tokens)
            scored = list(enumerate(scores))

        # 按分数降序取 top_k；vector_id 用 chunk 自带的稳定 ID（非列表下标）
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

        from app.core.metrics import metrics
        metrics.record_bm25(
            getattr(self, "strategy", "unknown"), time.time() - st
        )

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
                    "vector_id": int(document.get("vector_id", 0)),
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
