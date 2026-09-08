"""文档版本化 mixin：按 (document, strategy) 管理版本生命周期。

- _resolve_doc_versions / _apply_version_to_chunks：chunk_id 含版本（_v{N}_），
  各端（向量/metadata/MySQL/ES）按版本隔离写入
- _finalize_versions / _record_version：document_versions 台账 + retention 清理
- _gc_version：回收旧版本各端数据（只清当前策略，避免误伤其他策略旧版本）
- rollback_document：单活跃回滚（切到目标版本并丢弃更新版本）
- _bump_version：索引版本 +1（DB 唯一权威源，使查询缓存失效）
"""
from typing import Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


class VersioningMixin:
    """文档版本化 mixin（由 IndexWriter 组合）。"""

    def _bump_version(self, strategy: str, tenant_id: str) -> None:
        """索引版本 +1（数据库为唯一权威源；strict：失败抛异常）。

        仅 MySQL 后端启用时登记；未启用（本地纯文件模式）时跳过——
        此时版本恒为 unknown，查询缓存自动禁用，不影响功能正确性。
        """
        if not self.config.storage_mysql_enabled:
            return
        from app.storage.index_version_repository import IndexVersionRepository
        IndexVersionRepository().bump(tenant_id, strategy)

    # ---- 文档版本化（同名上传 = 新版本，活跃指针切换 + GC）----
    def _versioning_active(self) -> bool:
        """版本化是否生效：开关开启 + MySQL 后端启用（版本状态权威源）。"""
        return bool(
            self.config.document_versioning_enabled
            and self.config.storage_mysql_enabled
        )

    def _ensure_schema(self):
        """确保 MySQL 表结构就绪并完成迁移（幂等，失败放延迟到持久化阶段）。

        版本解析/台账写入前显式 init_schema：把老库缺列的迁移（如
        document_versions 补 strategy）在查询前完成，避免 Unknown column。
        """
        try:
            from app.storage.mysql import MySQLManager
            MySQLManager(pool_size=self.config.storage_pool_size).init_schema()
        except Exception as e:
            logger.warning("前置 schema 初始化失败（持久化阶段将重试）: %s", e)

    def _resolve_doc_versions(self, documents, tenant_id: str,
                              strategy: str, bump: bool) -> Dict:
        """解析每文档目标版本：{document_id: version}（按 (document, strategy) 维度）。

        - bump=True（upload）：该 (doc, strategy) 现存最大版本 + 1（新版本）；无则 1
        - bump=False（rebuild/incremental）：保持该 (doc, strategy) 当前活跃版本（不 bump，
          保证 rebuild 后 vector_id/chunk_id 稳定，ACL 不丢）
        - 版本化未生效：全部返回 1（chunk_id 保持旧格式，行为零变化）

        版本按策略隔离：同一文档在 fixed / recursive 各自独立版本，互不串扰。
        """
        versions = {d.document_id: 1 for d in documents}
        if not self._versioning_active():
            return versions
        from app.storage import DocumentRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        doc_repo = DocumentRepository(mgr)
        if not bump:
            active = doc_repo.get_active_versions(
                [d.document_id for d in documents], strategy, tenant_id
            )
        for doc in documents:
            if bump:
                versions[doc.document_id] = doc_repo.next_version(
                    doc.document_id, strategy, tenant_id
                )
            else:
                versions[doc.document_id] = active.get(doc.document_id, 1)
        return versions

    def _apply_version_to_chunks(self, chunks: List, version_by_doc: Dict) -> None:
        """版本化 chunk：chunk_id 含版本、重算 vector_id、metadata 带 version。

        幂等：重复执行（rebuild 重新写入同版本）得到相同 chunk_id/vector_id。
        版本化未启用时：version=1、chunk_id 保持 chunker 生成格式，零变化。
        """
        for c in chunks:
            ver = int(version_by_doc.get(c.document_id, 1) or 1)
            if self._versioning_active():
                c.chunk_id = "{0}_v{1}_chunk_{2}".format(
                    c.document_id, ver, c.chunk_index
                )
                c.metadata = dict(c.metadata or {})
                c.metadata["version"] = ver
            c.version = ver

    def _retention_keep(self) -> Optional[int]:
        """返回版本保留策略：latest → None（只留活跃）；N（int）→ 保留最近 N 版。"""
        raw = str(self.config.document_versioning_retention or "latest").lower().strip()
        if raw == "latest":
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "document_versioning.retention=%r 无法解析，按 latest（只留活跃）处理", raw,
            )
            return None

    def _finalize_versions(self, version_by_doc: Dict,
                           strategy: str, tenant_id: str) -> None:
        """写入完成后收尾：记录 (document, strategy) 版本台账 + 按 retention 清理旧版本。

        活跃版本 = 该 (document, strategy) 现存版本最大者（rehydration 自洽），
        因此无需额外的活跃指针切换；版本号按 (doc,strategy) 单调递增 + PK 去重保证并发安全。

        - retention=latest：GC 上一版本（当前策略，只留活跃）
        - retention=N：保留最近 N 版，清理 N 之前的旧版（老版本仍在索引，由检索
          活跃过滤保证只命中活跃版）
        """
        if not self._versioning_active():
            return
        keep = self._retention_keep()
        from app.storage import ChunkRepository, DocumentRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        doc_repo = DocumentRepository(mgr)
        chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)
        for doc_id, new_ver in version_by_doc.items():
            old_ver = new_ver - 1
            # 记录新版本到台账（版本列表/回滚/活跃版本派生的数据源）
            self._record_version(doc_id, strategy, new_ver, tenant_id)

            if keep is None:
                # retention=latest：GC 上一版本（当前策略），台账只留活跃
                if old_ver >= 1:
                    self._gc_version(doc_id, old_ver, strategy, tenant_id, chunk_repo)
                    try:
                        doc_repo.delete_version(doc_id, strategy, old_ver, tenant_id)
                    except Exception as e:
                        logger.warning(
                            "删除旧版本台账失败: doc=%s, strategy=%s, v=%s, %s",
                            doc_id, strategy, old_ver, e,
                        )
            else:
                # retention=N：保留最近 N 版，清理 N 之前的旧版本（当前策略）
                for v in range(1, new_ver - keep + 1):
                    if doc_repo.has_version(doc_id, strategy, v, tenant_id):
                        self._gc_version(doc_id, v, strategy, tenant_id, chunk_repo)
                        doc_repo.delete_version(doc_id, strategy, v, tenant_id)

    def _record_version(self, doc_id: str, strategy: str, version: int,
                        tenant_id: str) -> None:
        """把 (doc, strategy, version) 记录到 document_versions 台账（含该策略 chunk 数）。"""
        try:
            from app.storage import ChunkRepository, DocumentRepository
            from app.storage.mysql import MySQLManager

            mgr = MySQLManager(pool_size=self.config.storage_pool_size)
            chunk_count = len(
                ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)
                .get_vector_ids_by_document(
                    doc_id, tenant_id=tenant_id, version=version, strategy=strategy
                )
            )
            DocumentRepository(mgr).insert_version(
                doc_id, strategy, version, tenant_id, chunk_count=chunk_count
            )
        except Exception as e:
            logger.warning(
                "记录文档版本台账失败: doc=%s, strategy=%s, v=%s, %s",
                doc_id, strategy, version, e,
            )

    def rollback_document(self, document_id: str, target_version: int,
                          strategy: str, tenant_id: str = "default") -> int:
        """把文档在**指定策略**下回滚到较早版本（单活跃：切到目标并丢弃所有更新版本）。

        前置（按 (document, strategy)）：
          - 版本化必须启用
          - 目标版本必须存在于台账（未回收）且早于当前活跃版本
        成功后返回新活跃版本号。持该策略写锁与其他写入互斥。
        """
        if not self._versioning_active():
            raise RuntimeError("文档版本化未启用，无法执行回滚")
        from app.storage import ChunkRepository, DocumentRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        doc_repo = DocumentRepository(mgr)

        with self._write_lock(strategy, tenant_id):
            active = doc_repo.get_active_versions(
                [document_id], strategy, tenant_id
            ).get(document_id)
            if active is None:
                raise ValueError("文档不存在或尚无版本: {}".format(document_id))
            target = int(target_version)
            if target < 1 or target >= active:
                raise ValueError(
                    "仅支持回滚到较早版本（1 <= target < 当前版本 {}）".format(active)
                )
            if not doc_repo.has_version(document_id, strategy, target, tenant_id):
                raise ValueError("目标版本 {} 已回收/不存在，无法回滚".format(target))
            chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)
            # 单活跃：丢弃 target 之后的所有版本（数据 + 台账）
            for v in range(target + 1, active + 1):
                if doc_repo.has_version(document_id, strategy, v, tenant_id):
                    self._gc_version(document_id, v, strategy, tenant_id, chunk_repo)
                    doc_repo.delete_version(document_id, strategy, v, tenant_id)
            # 索引版本 +1：使查询缓存失效
            self._bump_version(strategy, tenant_id)
            logger.info(
                "文档回滚完成: doc=%s, strategy=%s, 活跃 %s → %s",
                document_id, strategy, active, target,
            )
            return target

    def _gc_version(self, doc_id: str, version: int, strategy: str,
                    tenant_id: str, chunk_repo) -> None:
        """删除旧版本各端数据（向量/metadata/MySQL/ES）。失败仅告警。

        **策略隔离**：只清理当前上传策略（strategy）的旧版本数据，
        不误伤其他策略的索引——同名版本化只作用于本次上传的策略。
        """
        try:
            vector_ids = chunk_repo.get_vector_ids_by_document(
                doc_id, tenant_id=tenant_id, version=version, strategy=strategy
            )
            if vector_ids:
                self._remove_vectors(vector_ids, strategy, tenant_id)
            self._remove_metadata(doc_id, version, strategy, tenant_id)
            if self.config.storage_es_enabled:
                from app.storage.es_repository import ChunkESRepository
                ChunkESRepository(
                    strategy=strategy, tenant_id=tenant_id
                ).delete_by_document_version(doc_id, version)
            # MySQL：只删当前策略该版本 chunks（策略隔离）
            chunk_repo.delete_by_document_version(
                doc_id, version, tenant_id, strategy=strategy
            )
            logger.info(
                "版本 GC 完成: doc=%s, v=%s, vectors=%d, strategy=%s",
                doc_id, version, len(vector_ids), strategy,
            )
        except Exception as e:
            logger.warning(
                "版本 GC 失败（残留数据，可重建修复）: doc=%s, v=%s, %s",
                doc_id, version, e, exc_info=True,
            )
