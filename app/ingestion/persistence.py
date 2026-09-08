"""多后端持久化 mixin：MySQL / ES 写入与清理（软失败策略）。

- MySQL 是「提交点/事实源」：documents/chunks 写入失败抛异常中止（严格一致，
  宁可失败不可半截，派生索引由对账任务以 MySQL 为基准最终一致补齐）
- ES / Milvus 清理与持久化失败仅告警（孤儿由对账/重建兜底）
"""
import time
from typing import List

from app.core.logger import get_logger

logger = get_logger(__name__)


class PersistenceMixin:
    """MySQL/ES 持久化与清理 mixin（由 IndexWriter 组合）。"""

    def _cleanup_es_incremental(self, strategy: str, deleted_doc_ids: List[str],
                                tenant_id: str = "default") -> bool:
        """ES 增量清理：仅删除被移除文档的 chunks（保留其他文档）。

        返回 True 表示执行成功（无数据可删也算成功），False=异常。
        """
        if not self.config.storage_es_enabled:
            logger.info("ES 增量清理跳过: storage_es_enabled=false")
            return False
        if not deleted_doc_ids:
            return True

        t = time.time()
        try:
            from app.storage.es_repository import ChunkESRepository

            es_repo = ChunkESRepository(strategy=strategy, tenant_id=tenant_id)
            es_repo.incremental_reindex(chunks=[], deleted_doc_ids=deleted_doc_ids)
            logger.info(
                "ES 增量清理完成: %.3fs, strategy=%s, tenant=%s, deleted_doc_ids=%s",
                time.time() - t, strategy, tenant_id, deleted_doc_ids,
            )
            return True
        except Exception as e:
            logger.warning(
                "ES 增量清理失败（不影响重建，后续会通过 insert_ignore 幂等写入）: "
                "strategy=%s, tenant=%s, deleted_doc_ids=%s, error=%s: %s",
                strategy, tenant_id, deleted_doc_ids, type(e).__name__, e, exc_info=True,
            )
            return False

    # ---- 内部：MySQL 持久化（软失败）----
    def _persist_to_mysql(self, documents, chunks, strategy,
                          tenant_id: str = "default",
                          owner_user_id: str = "") -> bool:
        """将文档和 chunks 写入 MySQL（如果启用）——**提交点严格模式**。

        租户隔离：documents/chunks 均写入 tenant_id。
        文档级 ACL：documents 写入 owner_user_id（上传者）。

        一致性设计（多写无共享事务）：
          - MySQL（documents + chunks）是「事实源/提交点」：任一步失败即抛异常
            中止本次写入 —— 上传对外失败、版本不切换，检索看不到半截数据
          - 派生索引（FAISS/Milvus、metadata、ES）失败不会导致提交点回滚，
            由对账任务（Reconciler）以 MySQL 为基准最终一致补齐
        """
        if not self.config.storage_mysql_enabled:
            logger.info("MySQL 持久化跳过: storage_mysql_enabled=false")
            return False

        t = time.time()
        from app.storage import DocumentRepository, ChunkRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        try:
            mgr.init_schema()
        except Exception as e:
            # MySQL 启用但不可用 = 提交点不可用：中止（严格一致，宁可失败不可半截）
            logger.error(
                "MySQL 不可用，中止写入（提交点严格模式）: %s", e, exc_info=True,
            )
            raise RuntimeError(
                "MySQL 不可用，无法完成写入（提交点严格模式）。"
                "请确认 MySQL 服务正常后重试。" 
            ) from e

        doc_repo = DocumentRepository(mgr)
        chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)

        # documents 行（主记录）：失败中止（版本推进/ACL 依赖它）
        for doc in documents:
            try:
                doc_repo.insert(
                    document_id=doc.document_id,
                    file_name=doc.metadata.get("source", doc.document_id),
                    content_length=len(doc.content),
                    source=doc.metadata.get("source"),
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                )
            except Exception as de:
                logger.error(
                    "MySQL 文档插入失败，中止写入（提交点）: doc_id=%s, tenant=%s, %s",
                    doc.document_id, tenant_id, de, exc_info=True,
                )
                raise

        # chunks（主记录）：失败中止
        try:
            chunk_repo.batch_insert(chunks)
        except Exception as ce:
            logger.error(
                "MySQL chunks 批量插入失败，中止写入（提交点）: "
                "strategy=%s, tenant=%s, chunks=%d, %s",
                strategy, tenant_id, len(chunks), ce, exc_info=True,
            )
            raise

        logger.info(
            "MySQL 持久化完成（提交点提交）: %.3fs, strategy=%s, tenant=%s, docs=%d, chunks=%d",
            time.time() - t, strategy, tenant_id, len(documents), len(chunks),
        )
        return True

    def _cleanup_mysql(self, strategy, tenant_id: str = "default") -> int:
        """删除指定 strategy（+tenant）的所有 chunks（重建前清理）。

        软失败：异常时记 warning 返回 0。
        """
        if not self.config.storage_mysql_enabled:
            logger.info("MySQL 清理跳过: storage_mysql_enabled=false")
            return 0

        t = time.time()
        try:
            from app.storage.chunk_repository import ChunkRepository
            from app.storage.mysql import MySQLManager

            mgr = MySQLManager(pool_size=self.config.storage_pool_size)
            chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)
            deleted = chunk_repo.delete_by_strategy()
            logger.info(
                "MySQL 清理完成: %.3fs, strategy=%s, tenant=%s, deleted=%d",
                time.time() - t, strategy, tenant_id, deleted,
            )
            return deleted
        except Exception as e:
            logger.warning(
                "MySQL 清理失败（不影响重建）: strategy=%s, tenant=%s, error=%s: %s",
                strategy, tenant_id, type(e).__name__, e, exc_info=True,
            )
            return 0

    # ---- 内部：ES 持久化（软失败）----
    def _persist_to_es(self, chunks, strategy, tenant_id: str = "default") -> bool:
        """将 chunks 写入 ES（如果启用）。

        租户隔离：使用 {prefix}_{tenant}_{strategy} 索引。
        增量写入：不删旧数据，直接追加。
        软失败：异常时记 warning 返回 False。
        """
        if not self.config.storage_es_enabled:
            logger.info("ES 持久化跳过: storage_es_enabled=false")
            return False

        t = time.time()
        try:
            from app.storage.es_repository import ChunkESRepository

            es_repo = ChunkESRepository(strategy=strategy, tenant_id=tenant_id)
            es_repo.batch_insert(chunks)
            logger.info(
                "ES 持久化完成: %.3fs, strategy=%s, tenant=%s, chunks=%d",
                time.time() - t, strategy, tenant_id, len(chunks),
            )
            return True
        except Exception as e:
            logger.warning(
                "ES 持久化失败（不影响索引）: strategy=%s, tenant=%s, chunks=%d, "
                "error=%s: %s",
                strategy, tenant_id, len(chunks), type(e).__name__, e, exc_info=True,
            )
            return False

    def _cleanup_es(self, strategy, tenant_id: str = "default") -> bool:
        """删除 ES 索引（重建前清理，租户索引）。

        软失败：异常时记 warning 返回 False。
        """
        if not self.config.storage_es_enabled:
            logger.info("ES 清理跳过: storage_es_enabled=false")
            return False

        t = time.time()
        try:
            from app.storage.es_repository import ChunkESRepository

            es_repo = ChunkESRepository(strategy=strategy, tenant_id=tenant_id)
            es_repo.drop_index()
            logger.info(
                "ES 清理完成: %.3fs, strategy=%s, tenant=%s", time.time() - t, strategy, tenant_id
            )
            return True
        except Exception as e:
            logger.warning(
                "ES 清理失败（不影响重建）: strategy=%s, tenant=%s, error=%s: %s",
                strategy, tenant_id, type(e).__name__, e, exc_info=True,
            )
            return False

    def _cleanup_milvus(self, strategy, tenant_id: str = "default") -> bool:
        """删除 Milvus collection（重建前清理，租户 collection）。

        软失败：异常时记 warning 返回 False。
        """
        if not self.config.storage_milvus_enabled:
            logger.info("Milvus 清理跳过: storage_milvus_enabled=false")
            return False

        t = time.time()
        try:
            from app.vector import create_vector_store

            collection_name = self.config.milvus_collection_for(strategy, tenant_id)
            # 只需要维度存在，Milvus drop 不依赖维度；给个占位默认值
            store = create_vector_store(
                backend="milvus",
                dimension=1,
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                collection_name=collection_name,
            )
            store.drop()
            logger.info(
                "Milvus 清理完成: %.3fs, strategy=%s, tenant=%s, collection=%s",
                time.time() - t, strategy, tenant_id, collection_name,
            )
            return True
        except Exception as e:
            logger.warning(
                "Milvus 清理失败（不影响重建）: strategy=%s, tenant=%s, error=%s: %s",
                strategy, tenant_id, type(e).__name__, e, exc_info=True,
            )
            return False
