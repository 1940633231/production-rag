"""删除 / 清理 mixin：覆盖更新 purge + 按 vector_id 移除向量与 metadata。

- 策略隔离：只作用于 (strategy, tenant)，不误伤其他策略索引
- MySQL 删除失败抛异常中止（避免半覆盖不一致）；向量/metadata/ES 失败仅告警
- 删除后做残留校验（孤儿可观测），失败列表返回给调用方
"""
from pathlib import Path
from typing import List

from app.core.logger import get_logger

logger = get_logger(__name__)


class RemovalMixin:
    """文档/版本数据移除 mixin（由 IndexWriter 组合）。"""

    def _remove_vectors_verified(self, vector_ids: List[int], strategy: str,
                                 tenant_id: str) -> List[str]:
        """按 vector_id 移除向量并校验无残留；返回失败描述列表（空=干净）。"""
        failures: List[str] = []
        if not vector_ids:
            return failures
        target = set(int(v) for v in vector_ids)

        # Milvus：remove 失败即异常；无法低成本枚举校验，依赖 remove 自身一致性
        if self.config.storage_milvus_enabled:
            try:
                from app.vector import create_vector_store

                col = self.config.milvus_collection_for(strategy, tenant_id)
                store = create_vector_store(
                    backend="milvus", dimension=1,
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                    collection_name=col,
                )
                store.load(col)
                store.remove(list(target))
                logger.info(
                    "Milvus 移除向量: collection=%s, ids=%d",
                    col, len(target),
                )
            except Exception as e:
                failures.append("Milvus 移除向量失败: {}".format(e))
            return failures

        # FAISS：移除后通过 ids() 枚举校验，确认目标 id 已不存在
        try:
            from app.vector import create_vector_store

            index_path = self.config.index_dir_for(strategy, tenant_id) / "faiss.index"
            if not Path(index_path).exists():
                # 该策略无本地向量索引 → 无孤儿，视为干净
                logger.info("FAISS 索引不存在，跳过向量移除: %s", index_path)
                return failures
            store = create_vector_store(
                backend="faiss", dimension=1,
                index_type=self.config.vector_index_type,
            )
            store.load(str(index_path))
            store.remove(list(target))
            store.save(str(index_path))
            remaining = set(int(i) for i in store.ids()) & target
            if remaining:
                samples = sorted(remaining)[:10]
                failures.append(
                    "FAISS 移除后仍残留 {} 个向量: {}".format(
                        len(remaining), samples
                    )
                )
            else:
                logger.info(
                    "FAISS 移除向量并校验通过: path=%s, ids=%d",
                    index_path, len(target),
                )
        except Exception as e:
            failures.append("FAISS 移除向量异常: {}".format(e))
        return failures

    def _remove_metadata_by_ids(self, vector_ids: List[int], strategy: str,
                                tenant_id: str) -> None:
        """从 metadata.json 按 vector_id 摘除条目（并发安全：读-改-写）。"""
        from app.storage.metadata_store import MetadataStore

        meta_path = self.config.index_dir_for(strategy, tenant_id) / "metadata.json"
        if not Path(meta_path).exists():
            return
        ms = MetadataStore()
        entries = ms.load(str(meta_path)) or {}
        removed = 0
        for vid in vector_ids:
            if entries.pop(str(vid), None) is not None:
                removed += 1
        if removed:
            ms.save_entries(entries, str(meta_path))
            logger.info(
                "metadata.json 移除条目: path=%s, 移除=%d, 剩余=%d",
                meta_path, removed, len(entries),
            )

    def _remove_vectors(self, vector_ids: List[int], strategy: str,
                        tenant_id: str) -> None:
        """按 vector_id 从向量后端（Milvus/FAISS）移除向量。"""
        if not vector_ids:
            return
        if self.config.storage_milvus_enabled:
            try:
                from app.vector import create_vector_store

                col = self.config.milvus_collection_for(strategy, tenant_id)
                store = create_vector_store(
                    backend="milvus", dimension=1,
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                    collection_name=col,
                )
                store.load(col)
                store.remove(vector_ids)
                logger.info(
                    "Milvus 移除向量: collection=%s, ids=%d",
                    col, len(vector_ids),
                )
                return
            except Exception as e:
                logger.warning(
                    "Milvus 移除向量失败（可重建修复）: %s", e,
                )
                return
        try:
            from app.vector import create_vector_store

            index_path = self.config.index_dir_for(strategy, tenant_id) / "faiss.index"
            if Path(index_path).exists():
                store = create_vector_store(
                    backend="faiss", dimension=1,
                    index_type=self.config.vector_index_type,
                )
                store.load(str(index_path))
                store.remove(vector_ids)
                store.save(str(index_path))
                logger.info(
                    "FAISS 移除向量: path=%s, ids=%d",
                    index_path, len(vector_ids),
                )
        except Exception as e:
            logger.warning(
                "FAISS 移除向量失败（可重建修复）: %s", e,
            )

    def _remove_metadata(self, doc_id: str, version: int, strategy: str,
                         tenant_id: str) -> None:
        """从 metadata.json 摘除指定 (document_id, version) 的条目。"""
        try:
            from app.storage.metadata_store import MetadataStore

            meta_path = self.config.index_dir_for(strategy, tenant_id) / "metadata.json"
            if not Path(meta_path).exists():
                return
            ms = MetadataStore()
            entries = ms.load(str(meta_path)) or {}
            kept = {
                vid: e for vid, e in entries.items()
                if not (
                    e.get("document_id") == doc_id
                    and int(e.get("version", 1)) == int(version)
                )
            }
            if len(kept) != len(entries):
                ms.save_entries(kept, str(meta_path))
                logger.info(
                    "metadata 摘除版本条目: doc=%s, v=%s, 移除=%d, 剩余=%d",
                    doc_id, version, len(entries) - len(kept), len(kept),
                )
        except Exception as e:
            logger.warning(
                "metadata 摘除版本条目失败（可重建修复）: %s", e,
            )

    # ---- 覆盖更新（版本化关闭时的同名文档处理）----
    def _overwrite_purge(self, documents, strategy: str, tenant_id: str) -> None:
        """覆盖更新前置：同名文档（documents 表已存在）先清旧内容。

        仅 MySQL 启用时生效（依赖 documents 表判定存在性；MySQL 关闭的纯本地
        模式保持原半覆盖行为）。MySQL chunks 删除失败抛异常（中止写入，避免
        半覆盖不一致）；向量/metadata/ES 清理失败仅告警（孤儿可重建修复）。
        """
        if not self.config.storage_mysql_enabled:
            return
        from app.storage import ChunkRepository, DocumentRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        doc_repo = DocumentRepository(mgr)
        chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)
        for doc in documents:
            existing = doc_repo.get(doc.document_id, tenant_id=tenant_id)
            if existing is None:
                continue  # 新文档，无旧内容
            self._purge_document(doc.document_id, strategy, tenant_id, chunk_repo)

    def _purge_document(self, doc_id: str, strategy: str, tenant_id: str,
                        chunk_repo) -> None:
        """清空某文档在当前策略下的旧内容（向量/metadata/MySQL/ES）。

        **策略隔离**：只清理当前上传策略（strategy），不误伤其他策略索引；
        documents 行保留（ACL 不丢）。MySQL 删除失败抛异常（中止写入，
        避免半覆盖不一致）；向量/metadata/ES 失败仅告警（可重建修复）。
        """
        try:
            vector_ids = chunk_repo.get_vector_ids_by_document(
                doc_id, tenant_id=tenant_id, strategy=strategy
            )
            if vector_ids:
                self._remove_vectors(vector_ids, strategy, tenant_id)
            self._remove_metadata_document(doc_id, strategy, tenant_id)
            if self.config.storage_es_enabled:
                from app.storage.es_repository import ChunkESRepository
                ChunkESRepository(
                    strategy=strategy, tenant_id=tenant_id
                ).incremental_reindex(chunks=[], deleted_doc_ids=[doc_id])
            # MySQL chunks：只删当前策略（策略隔离）；失败抛异常中止写入
            chunk_repo.delete_by_document(doc_id, tenant_id, strategy=strategy)
            logger.info(
                "覆盖更新清理完成: doc=%s, vectors=%d, strategy=%s",
                doc_id, len(vector_ids), strategy,
            )
        except Exception as e:
            logger.warning(
                "覆盖更新清理失败（中止写入）: doc=%s, %s", doc_id, e, exc_info=True,
            )
            raise

    def _remove_metadata_document(self, doc_id: str, strategy: str,
                                  tenant_id: str) -> None:
        """从 metadata.json 摘除某文档的全部条目（覆盖更新用）。"""
        try:
            from app.storage.metadata_store import MetadataStore

            meta_path = self.config.index_dir_for(strategy, tenant_id) / "metadata.json"
            if not Path(meta_path).exists():
                return
            ms = MetadataStore()
            entries = ms.load(str(meta_path)) or {}
            kept = {
                vid: e for vid, e in entries.items()
                if e.get("document_id") != doc_id
            }
            if len(kept) != len(entries):
                ms.save_entries(kept, str(meta_path))
                logger.info(
                    "metadata 摘除文档条目: doc=%s, 移除=%d, 剩余=%d",
                    doc_id, len(entries) - len(kept), len(kept),
                )
        except Exception as e:
            logger.warning(
                "metadata 摘除文档条目失败（可重建修复）: %s", e,
            )
