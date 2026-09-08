"""发布前校验（Build → Validation → Publish 的 Validation 阶段）。

确认本次写入的 chunks 已在各启用后端就绪（metadata.json / ES / 向量），
校验通过才允许 Publish（版本切换）；strict=false 时仅告警继续发布。
"""
from pathlib import Path

from app.core.logger import get_logger

logger = get_logger(__name__)


class ValidationMixin:
    """发布前校验 mixin（由 IndexWriter 组合）。"""

    # ---- 发布前校验（Validation）----
    def _validate_build(self, chunks, strategy: str, tenant_id: str) -> None:
        """发布前校验：确认本次 chunk 已写入各启用后端，通过才允许 Publish。

        校验项（对本次写入的 vector_id 集合做存在性核对）：
          - metadata.json：全部 vector_id 已落盘
          - ES（启用时）：全部 vector_id 可查（list_all 分页对比）
          - 向量后端：FAISS 可枚举 ids 时核对全部已入库；Milvus 跳过（无法枚举）

        失败策略：
          - strict（默认 true）：抛异常中止发布——版本不切换，检索不受影响，
            已写入的新 generation 成为孤儿，由对账器/重试兜底
          - strict=false：仅告警，继续发布
        """
        if not self.config.write_validation_enabled:
            return
        expected = {int(c.vector_id) for c in chunks if c.vector_id}
        failures = []

        # 1. metadata.json
        try:
            from app.storage.metadata_store import MetadataStore
            meta_path = self.config.index_dir_for(strategy, tenant_id) / "metadata.json"
            entries = MetadataStore().load(str(meta_path)) or {}
            meta_ids = set()
            for k, e in entries.items():
                try:
                    meta_ids.add(int(e.get("vector_id", k)))
                except (TypeError, ValueError):
                    pass
            missing = expected - meta_ids
            if missing:
                failures.append("metadata.json 缺失 {} 个 vector_id".format(len(missing)))
        except Exception as e:
            failures.append("metadata 校验失败: {}".format(e))

        # 2. ES（启用时）
        if self.config.storage_es_enabled:
            try:
                from app.storage.es_repository import ChunkESRepository
                es_repo = ChunkESRepository(strategy=strategy, tenant_id=tenant_id)
                es_ids = set(
                    int(c.get("vector_id")) for c in es_repo.list_all()
                    if c.get("vector_id") is not None
                )
                missing = expected - es_ids
                if missing:
                    failures.append("ES 缺失 {} 个 vector_id".format(len(missing)))
            except Exception as e:
                failures.append("ES 校验失败: {}".format(e))

        # 3. 向量后端（FAISS 可枚举；Milvus 跳过）
        if not self.config.storage_milvus_enabled:
            try:
                from app.vector import create_vector_store
                index_path = self.config.index_dir_for(strategy, tenant_id) / "faiss.index"
                if Path(index_path).exists():
                    store = create_vector_store(
                        backend="faiss", dimension=1,
                        index_type=self.config.vector_index_type,
                    )
                    store.load(str(index_path))
                    vector_ids = set(int(i) for i in store.ids())
                    missing = expected - vector_ids
                    if missing:
                        failures.append("向量后端缺失 {} 个 vector_id".format(len(missing)))
            except Exception as e:
                failures.append("向量校验失败: {}".format(e))

        if failures:
            msg = "写入校验失败: {}".format("; ".join(failures))
            if self.config.write_validation_strict:
                logger.error("%s —— 中止发布（版本不切换，构建数据待对账/重试）", msg)
                raise RuntimeError(msg)
            logger.warning("%s —— 非 strict 模式，继续发布", msg)
        else:
            logger.info(
                "写入校验通过: chunks=%d, metadata/es/vector 均已就绪, strategy=%s",
                len(expected), strategy,
            )
