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

    每个 (tenant, strategy) 对应一个 ES 索引（如 production_rag_recursive /
    production_rag_tenantA_recursive），租户间完全隔离。
    支持增量更新：incremental_reindex 只删除被移除文档的 chunks。
    """

    def __init__(self, strategy: str = "recursive",
                 es_client=None, tenant_id: str = "default", **kwargs):
        self.strategy = strategy
        self.tenant_id = tenant_id

        if es_client is not None:
            self._es = es_client
        else:
            from app.storage.es_client import ESClient
            self._es = ESClient(tenant_id=tenant_id, **kwargs)

    def get_by_id(self, id: int) -> Optional[Dict]:
        """按稳定 vector_id 按需单查（无全量内存缓存）。"""
        return self._es.get_by_vector_id(self.strategy, int(id))

    def batch_get_by_ids(self, ids: List[int]) -> List[Dict]:
        """批量按向量 ID 查询 chunks（逐条按需，命中数通常很小）。"""
        result = []
        for i in ids:
            doc = self._es.get_by_vector_id(self.strategy, int(i))
            if doc is not None:
                result.append(doc)
        return result

    def list_all(self) -> List[Dict]:
        """返回当前 strategy 的所有 chunks（分页拉取，避免单次 size 截断）。"""
        return self._es.search_all(self.strategy, query="*")

    def vector_ids_by_documents(self, document_ids) -> Dict[str, set]:
        """按可读文档集合返回 {document_id: {vector_id}}（先过滤后检索，按需）。"""
        return self._es.vector_ids_by_documents(self.strategy, list(document_ids))

    def count(self) -> int:
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
                "vector_id": int(getattr(c, "vector_id", 0) or 0),
                "version": int(getattr(c, "version", 1) or 1),
                "content": c.content,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "metadata": c.metadata or {},
            }
            for c in chunks
        ]
        self._es.bulk_index(strat, es_docs)
        # 显式 refresh：ES 默认 ~1s 才可搜，立即 refresh 保证紧随其后的
        # 校验（_validate_build 用 list_all 核对 vector_id）能读到刚写入的数据，
        # 避免「写入成功却被判缺失」的时序误报。
        self._es.refresh_index(strat)

    def incremental_reindex(self, chunks: List, deleted_doc_ids: List[str] = None):
        """增量更新：只删除被移除文档的 chunks，再写入新 chunks。

        比全量重建更高效，适用于仅删除少量文档的场景。
        """
        strat = self.strategy
        if deleted_doc_ids:
            idx = self._es._index_name(strat)
            for doc_id in deleted_doc_ids:
                self._es._client.delete_by_query(
                    index=idx,
                    body={"query": {"term": {"document_id": doc_id}}},
                    refresh=True,
                )
        self.batch_insert(chunks, strat)

    def delete_by_document_version(self, document_id: str, version: int):
        """删除某文档指定版本的 chunks（版本 GC 用）。"""
        idx = self._es._index_name(self.strategy)
        self._es._client.delete_by_query(
            index=idx,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"document_id": document_id}},
                            {"term": {"version": int(version)}},
                        ]
                    }
                }
            },
            refresh=True,
        )
        logger.info(
            "ES 删除文档版本 chunks: strategy=%s, doc=%s, version=%s",
            self.strategy, document_id, version,
        )

    def drop_index(self):
        """删除整个 ES 索引（全量重建时使用）。"""
        self._es.drop_index(self.strategy)
