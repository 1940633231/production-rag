import time

import numpy as np

from app.embedding.model import EmbeddingModel
from app.core.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """向量检索器：依赖 BaseChunkRepository 获取 chunk 元数据。

    通过 chunk_repo.get_by_id(int(index)) 查询向量位置对应的 chunk，
    本地 LRU 缓存避免同一 Retriever 实例内重复查询存储后端。
    """

    def __init__(self, embedding_model, vector_store, chunk_repo):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunk_repo = chunk_repo
        # 本地缓存：vector_id(int) → chunk_dict，同一实例内避免重复查 repo
        self._cache = {}

    def search(self, query, top_k=10, document_ids=None):

        t = time.time()
        logger.info("向量检索开始: query=%r, top_k=%d", query, top_k)

        query_vector = self.embedding_model.encode([query])

        scores, ids = self.vector_store.search(query_vector, top_k)

        results = []

        for score, index in zip(scores[0], ids[0]):

            if index < 0:
                continue

            idx = int(index)

            # LRU 缓存：命中则跳过 chunk_repo 查询
            if idx not in self._cache:
                self._cache[idx] = self.chunk_repo.get_by_id(idx)

            document = self._cache[idx]
            if document is None:
                logger.warning("chunk_repo 未找到 vector_id=%d，跳过", idx)
                continue

            # 文档级 ACL：仅返回用户可读文档的 chunk
            if document_ids is not None:
                doc_id = document.get("document_id")
                if doc_id not in document_ids:
                    logger.debug(
                        "ACL 过滤向量结果: vector_id=%d, document_id=%s 不可读",
                        idx, doc_id,
                    )
                    continue

            results.append(
                {
                    "vector_id": idx,
                    "score": float(score),
                    "chunk_id": document["chunk_id"],
                    "content": document["content"],
                    "start_offset": document.get("start_offset", 0),
                    "end_offset": document.get("end_offset", 0),
                    "metadata": document["metadata"],
                }
            )

        logger.info(
            "向量检索完成: %.3fs, 结果数=%d, top_score=%.4f",
            time.time() - t, len(results),
            results[0]["score"] if results else 0,
        )
        return results
