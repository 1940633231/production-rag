"""Elasticsearch 分块仓库：基于 ESClient 实现读写接口。

支持增量写入：重建索引时可不删除整个索引，而是只删除被移除文档的 chunks。

当前为占位实现：ES 库未安装或连接失败时 __init__ 抛 RuntimeError，
触发上层降级到 MetadataStore / MySQL。
"""
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.storage.base import BaseChunkRepository

logger = get_logger(__name__)


class ChunkESRepository(BaseChunkRepository):
    """ES 后端分块仓库。

    每个 strategy 对应一个 ES 索引（如 production_rag_recursive）。
    支持增量更新：incremental_reindex 只删除被移除文档的 chunks。
    """

    def __init__(self, strategy: str = "recursive",
                 es_client=None, **kwargs):
        self.strategy = strategy

        if es_client is not None:
            self._es = es_client
        else:
            from app.storage.es_client import ESClient
            self._es = ESClient(**kwargs)

        # 懒加载缓存
        self._cache_list: Optional[List[Dict]] = None
        self._cache_map: Optional[Dict[int, Dict]] = None

    def _ensure_loaded(self):
        """首次访问时从 ES 加载所有 chunks 到内存缓存。

        关键：cache_map 的 key 必须是 ES 文档中的 vector_id 字段
        （与 FAISS IndexFlatIP 的位置 ID 对齐），而不是 enumerate 顺序索引。
        否则 Retriever.get_by_id(int(faiss_id)) 会取到错误的 chunk。
        """
        if self._cache_list is not None:
            return
        # 按 vector_id 升序，保证 list_all 顺序与写入顺序一致
        result = self._es.search(
            self.strategy, query="*", top_k=10000, sort_by_vector_id=True
        )
        # 用 vector_id 作为 key（对齐 FAISS ID），缺失 vector_id 的文档跳过
        self._cache_list = result
        self._cache_map = {}
        missing_vector_id = 0
        for r in result:
            vid = r.get("vector_id")
            if vid is None:
                missing_vector_id += 1
                continue
            self._cache_map[int(vid)] = r
        if missing_vector_id:
            logger.warning(
                "ChunkESRepository 加载: %d 个文档缺少 vector_id 字段，已跳过",
                missing_vector_id,
            )
        logger.info(
            "ChunkESRepository 加载: strategy=%s, chunks=%d, map_keys=%d",
            self.strategy, len(result), len(self._cache_map),
        )

    def get_by_id(self, id: int) -> Optional[Dict]:
        self._ensure_loaded()
        return self._cache_map.get(id)

    def batch_get_by_ids(self, ids: List[int]) -> List[Dict]:
        self._ensure_loaded()
        result = []
        for i in ids:
            doc = self._cache_map.get(i)
            if doc is not None:
                result.append(doc)
        return result

    def list_all(self) -> List[Dict]:
        self._ensure_loaded()
        return self._cache_list

    def count(self) -> int:
        if self._cache_list is not None:
            return len(self._cache_list)
        return self._es.count(self.strategy)

    # ---- 写接口 ----

    def batch_insert(self, chunks: List, strategy: Optional[str] = None):
        """批量写入 chunks 到 ES（增量写入，不删旧数据）。

        vector_id 字段使用 enumerate(chunks) 的下标，与 FAISS IndexFlatIP
        的位置 ID 对齐（pipeline.write() 中 vector_store.add(vectors) 按相同
        顺序写入向量）。后续 Retriever.get_by_id(int(faiss_id)) 据此取回 chunk。
        """
        strat = strategy or self.strategy
        es_docs = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "strategy": strat,
                "chunk_index": c.chunk_index,
                "vector_id": i,
                "content": c.content,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "metadata": c.metadata or {},
            }
            for i, c in enumerate(chunks)
        ]
        self._es.bulk_index(strat, es_docs)
        # 失效缓存
        self._cache_list = None
        self._cache_map = None

    def incremental_reindex(self, chunks: List, deleted_doc_ids: List[str] = None):
        """增量更新：只删除被移除文档的 chunks，再写入新 chunks。

        比全量重建更高效，适用于仅删除少量文档的场景。
        """
        strat = self.strategy
        if deleted_doc_ids:
            for doc_id in deleted_doc_ids:
                idx = "{}_{}".format(self._es.index_prefix, strat)
                self._es._client.delete_by_query(
                    index=idx,
                    body={"query": {"term": {"document_id": doc_id}}},
                )
                logger.info("ES 增量删除: doc_id=%s, strategy=%s", doc_id, strat)
        self.batch_insert(chunks, strat)

    def drop_index(self):
        """删除整个 ES 索引（全量重建时使用）。"""
        self._es.drop_index(self.strategy)
        self._cache_list = None
        self._cache_map = None
