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
        # 懒加载缓存：document_id → {vector_id, ...}（用于先过滤后检索）
        self._doc_vectors = None

    def _document_vector_ids(self, document_ids) -> set:
        """把可读文档集合翻译为允许的 vector_id 集合（先过滤后检索）。

        基于 chunk_repo.list_all()（当前 strategy 全量 chunks，已缓存）构建
        document_id → vector_id 映射，一次构建、后续查询零开销。
        """
        if self._doc_vectors is None:
            m = {}
            for c in self.chunk_repo.list_all():
                doc = c.get("document_id")
                vid = int(c.get("vector_id", 0))
                if doc:
                    m.setdefault(doc, set()).add(vid)
            self._doc_vectors = m
        allowed: set = set()
        for doc_id in document_ids:
            allowed |= self._doc_vectors.get(doc_id, set())
        return allowed

    def search(self, query, top_k=10, document_ids=None):

        from app.core.metrics import metrics
        t = time.time()
        logger.info("向量检索开始: query=%r, top_k=%d", query, top_k)

        # query embedding 阶段（单独计时）
        et = time.time()
        query_vector = self.embedding_model.encode([query])
        metrics.record_embedding(
            getattr(self, "strategy", "unknown"), time.time() - et
        )

        # 先过滤后检索：可读文档 → 允许的 vector_id 集合 → 向量后端按 id 预过滤
        vector_ids = None
        if document_ids is not None:
            vector_ids = self._document_vector_ids(document_ids)
            logger.info(
                "向量预过滤: 可读文档=%d, 允许向量=%d",
                len(document_ids), len(vector_ids),
            )

        # 向量库检索阶段（单独计时）
        vt = time.time()
        scores, ids = self.vector_store.search(
            query_vector, top_k, vector_ids=vector_ids
        )
        metrics.record_vector(
            getattr(self, "strategy", "unknown"), time.time() - vt
        )

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

            # 文档级 ACL 兜底：预过滤后仍只返回可读文档（防御性校验）
            if document_ids is not None:
                doc_id = document.get("document_id")
                if doc_id not in document_ids:
                    logger.debug(
                        "ACL 兜底过滤向量结果: vector_id=%d, document_id=%s 不可读",
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
