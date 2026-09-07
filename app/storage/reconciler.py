"""派生索引对账器：以 MySQL chunks 为事实源，比对并修复派生索引。

背景：MySQL / ES / FAISS / Milvus / metadata.json 之间无共享事务，写入失败
可能造成各端差异（缺失 / 孤儿）。本对账器以 MySQL（主记录，提交点）为基准，
差分修复派生索引，达到最终一致。

修复项（fix=True）：
  - metadata.json：补缺失条目 / 摘除孤儿条目
  - ES：补缺失文档 / 删除孤儿文档
  - 向量后端（FAISS/Milvus）：仅报告差异——补齐需重新 embed（成本高），
    差异较大时建议直接 rebuild；少量孤儿可手动清理

用法：
  from app.storage.reconciler import Reconciler
  report = Reconciler().reconcile(strategy="recursive", tenant_id="default")
"""
import time
from typing import Dict, List, Optional

from app.core.config import Config
from app.core.logger import get_logger

logger = get_logger(__name__)


class Reconciler:
    """以 MySQL chunks 为事实源的派生索引对账器。"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    # ---- 主入口 ----

    def reconcile(self, strategy: str, tenant_id: str = "default",
                  fix: bool = True) -> Dict:
        """执行对账，返回差异报告。

        fix=True 时修复 metadata/ES 差异；向量侧始终只报告。
        """
        t0 = time.time()
        if not self.config.storage_mysql_enabled:
            logger.info("对账跳过: MySQL 未启用（无事实源）")
            return {
                "status": "skipped", "reason": "storage.backends.mysql.enabled=false",
                "strategy": strategy, "tenant_id": tenant_id,
            }

        from app.storage import ChunkRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)

        # 1. 事实源：MySQL chunks
        mysql_chunks = chunk_repo.list_all()
        mysql_by_vid = {int(c["vector_id"]): c for c in mysql_chunks if c.get("vector_id")}
        mysql_ids = set(mysql_by_vid.keys())
        logger.info(
            "对账开始: strategy=%s, tenant=%s, 事实源 chunks=%d",
            strategy, tenant_id, len(mysql_ids),
        )

        report: Dict = {
            "status": "ok", "strategy": strategy, "tenant_id": tenant_id,
            "mysql_chunks": len(mysql_ids),
            "metadata_missing": 0, "metadata_orphan": 0,
            "es_missing": 0, "es_orphan": 0,
            "vector_missing": 0, "vector_orphan": 0,
            "fixed_metadata_missing": 0, "fixed_metadata_orphan": 0,
            "fixed_es_missing": 0, "fixed_es_orphan": 0,
            "elapsed_seconds": 0.0,
        }

        # 2. metadata.json 对账
        self._reconcile_metadata(
            mysql_by_vid, mysql_ids, strategy, tenant_id, fix, report,
        )

        # 3. ES 对账
        self._reconcile_es(
            mysql_by_vid, mysql_ids, strategy, tenant_id, fix, report,
        )

        # 4. 向量后端：仅报告（补齐需 embed，建议 rebuild）
        self._report_vector_drift(
            mysql_ids, strategy, tenant_id, report,
        )

        report["elapsed_seconds"] = round(time.time() - t0, 3)
        logger.info(
            "对账完成: %.3fs, status=%s, metadata(miss/orphan)=%d/%d, "
            "es(miss/orphan)=%d/%d, vector(miss/orphan)=%d/%d",
            report["elapsed_seconds"], report["status"],
            report["metadata_missing"], report["metadata_orphan"],
            report["es_missing"], report["es_orphan"],
            report["vector_missing"], report["vector_orphan"],
        )
        return report

    # ---- 各端对账 ----

    def _reconcile_metadata(self, mysql_by_vid: Dict, mysql_ids: set,
                            strategy: str, tenant_id: str, fix: bool,
                            report: Dict) -> None:
        """metadata.json：补缺失（事实源有、文件无）+ 摘孤儿（文件有、事实源无）。"""
        from app.storage.metadata_store import MetadataStore

        meta_path = self.config.index_dir_for(strategy, tenant_id) / "metadata.json"
        entries = {}
        from pathlib import Path
        if Path(meta_path).exists():
            try:
                entries = MetadataStore().load(str(meta_path)) or {}
            except Exception as e:
                logger.warning("对账: metadata.json 加载失败: %s", e)

        meta_ids = set()
        for k, e in entries.items():
            try:
                meta_ids.add(int(e.get("vector_id", k)))
            except (TypeError, ValueError):
                pass

        missing_ids = mysql_ids - meta_ids
        orphan_ids = meta_ids - mysql_ids
        report["metadata_missing"] = len(missing_ids)
        report["metadata_orphan"] = len(orphan_ids)

        if not fix:
            return
        fixed_m, fixed_o = 0, 0
        if missing_ids:
            for vid in sorted(missing_ids):
                c = mysql_by_vid[vid]
                entries[str(vid)] = {
                    "chunk_id": c.get("chunk_id"),
                    "document_id": c.get("document_id"),
                    "vector_id": vid,
                    "version": c.get("version", 1),
                    "content": c.get("content"),
                    "start_offset": c.get("start_offset", 0),
                    "end_offset": c.get("end_offset", 0),
                    "metadata": c.get("metadata") or {},
                }
                fixed_m += 1
        if orphan_ids:
            for vid in orphan_ids:
                entries.pop(str(vid), None)
                fixed_o += 1
        if fixed_m or fixed_o:
            try:
                MetadataStore().save_entries(entries, str(meta_path))
                logger.info(
                    "对账修复 metadata: 补 %d, 摘 %d, strategy=%s",
                    fixed_m, fixed_o, strategy,
                )
            except Exception as e:
                logger.warning("对账修复 metadata 失败: %s", e)
                fixed_m = fixed_o = 0
        report["fixed_metadata_missing"] = fixed_m
        report["fixed_metadata_orphan"] = fixed_o

    def _reconcile_es(self, mysql_by_vid: Dict, mysql_ids: set,
                      strategy: str, tenant_id: str, fix: bool,
                      report: Dict) -> None:
        """ES：补缺失 + 删孤儿（按 vector_id 差分）。"""
        if not self.config.storage_es_enabled:
            return
        from app.storage.es_repository import ChunkESRepository

        es_repo = ChunkESRepository(strategy=strategy, tenant_id=tenant_id)
        try:
            es_chunks = es_repo.list_all()
        except Exception as e:
            logger.warning("对账: ES 读取失败，跳过 ES 对账: %s", e)
            return
        es_ids = set()
        for c in es_chunks:
            vid = c.get("vector_id")
            if vid is not None:
                es_ids.add(int(vid))

        missing_ids = mysql_ids - es_ids
        orphan_ids = es_ids - mysql_ids
        report["es_missing"] = len(missing_ids)
        report["es_orphan"] = len(orphan_ids)

        if not fix:
            return
        fixed_m, fixed_o = 0, 0
        if missing_ids:
            docs = []
            for vid in sorted(missing_ids):
                c = mysql_by_vid[vid]
                docs.append({
                    "chunk_id": c.get("chunk_id"),
                    "document_id": c.get("document_id"),
                    "strategy": strategy,
                    "chunk_index": c.get("chunk_index", 0),
                    "vector_id": vid,
                    "version": c.get("version", 1),
                    "content": c.get("content"),
                    "start_offset": c.get("start_offset", 0),
                    "end_offset": c.get("end_offset", 0),
                    "metadata": c.get("metadata") or {},
                })
            try:
                es_repo.batch_insert(docs, strategy=strategy)
                fixed_m = len(docs)
            except Exception as e:
                logger.warning("对账修复 ES 缺失失败: %s", e)
        if orphan_ids:
            try:
                from app.storage.es_client import ESClient
                idx = ESClient(tenant_id=tenant_id)._index_name(strategy)
                es_repo._es._client.delete_by_query(
                    index=idx,
                    body={"query": {"terms": {"vector_id": sorted(orphan_ids)}}},
                    refresh=True,
                )
                fixed_o = len(orphan_ids)
            except Exception as e:
                logger.warning("对账清理 ES 孤儿失败: %s", e)
        report["fixed_es_missing"] = fixed_m
        report["fixed_es_orphan"] = fixed_o

    def _report_vector_drift(self, mysql_ids: set, strategy: str,
                             tenant_id: str, report: Dict) -> None:
        """向量后端：枚举本地索引 ids，报告差异（不自动修复，建议 rebuild）。"""
        try:
            index_path = self.config.index_dir_for(strategy, tenant_id) / "faiss.index"
            if not index_path.exists():
                report["vector_missing"] = len(mysql_ids)
                report["vector_orphan"] = 0
                return
            from app.vector import create_vector_store
            store = create_vector_store(
                backend="faiss", dimension=1,
                index_type=self.config.vector_index_type,
            )
            store.load(str(index_path))
            vector_ids = set(int(i) for i in (store.ids() or []))
            report["vector_missing"] = len(mysql_ids - vector_ids)
            report["vector_orphan"] = len(vector_ids - mysql_ids)
        except Exception as e:
            logger.warning("对账: 向量侧枚举失败（跳过报告）: %s", e)
