"""统一索引写入器：所有写入入口的唯一收敛点（编排层）。

职责（本文件只保留编排与入口，实现细节按生命周期拆到 mixin）：
  - 对外入口：write / rebuild / incremental_rebuild_after_delete /
    remove_document / rollback_document（持写锁 + 流程编排）
  - _write_locked 全流程：clean → chunk → 版本解析 → 覆盖清理 → embed →
    向量/metadata/MySQL/ES 写入 → 发布前校验 → 版本收尾 → 索引版本 +1

模块分工：
  - writer_locks.IndexWriteLock —— 跨进程可重入写锁
  - ValidationMixin   (validation)   —— 发布前校验（Build→Validation→Publish）
  - PersistenceMixin  (persistence)  —— MySQL/ES 持久化与清理（软失败）
  - RemovalMixin      (removal)      —— 覆盖清理 / 按 vector_id 移除 / 残留校验
  - VersioningMixin   (versioning)   —— 文档版本化：解析/台账/GC/回滚/索引版本
  - LoadingMixin      (loading)      —— data/raw 文档加载

对外导入路径 `from app.ingestion.writer import IndexWriter` 保持不变（Mixin 组合）。
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import Config
from app.core.logger import get_logger
from app.ingestion.chunk import vector_id_for
from app.ingestion.writer_locks import IndexWriteLock
from app.ingestion.validation import ValidationMixin
from app.ingestion.persistence import PersistenceMixin
from app.ingestion.removal import RemovalMixin
from app.ingestion.versioning import VersioningMixin
from app.ingestion.loading import LoadingMixin

logger = get_logger(__name__)


class IndexWriter(
    ValidationMixin,
    PersistenceMixin,
    RemovalMixin,
    VersioningMixin,
    LoadingMixin,
):
    """统一索引写入器：所有写入入口的唯一收敛点。

    替代 knowledge.py 中 _run_ingestion / _do_upload / _do_rebuild / _rebuild_silent
    四处重复代码 + pipeline.py 的 _persist_to_mysql 软失败逻辑。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    # ---- 核心写入 ----
    def _write_lock(self, strategy: str, tenant_id: str) -> IndexWriteLock:
        """（strategy, tenant）粒度写锁。"""
        return IndexWriteLock(
            self.config.index_dir_for(strategy, tenant_id) / "write.lock"
        )

    def write(self, documents: List, strategy: str,
              tenant_id: str = "default",
              owner_user_id: str = "",
              index_path: Optional[str] = None,
              metadata_path: Optional[str] = None,
              version_bump: bool = True,
              overwrite_existing: bool = True) -> Dict:
        """写入文档到所有启用的存储后端。

        外层持（strategy, tenant）写锁（P1-4/P1-5：跨进程互斥、防文件竞态），
        实际逻辑在 _write_locked（可重入，rebuild 等嵌套调用复用同一把锁）。
        """
        with self._write_lock(strategy, tenant_id):
            return self._write_locked(
                documents, strategy, tenant_id, owner_user_id,
                index_path, metadata_path, version_bump, overwrite_existing,
            )

    def _write_locked(self, documents: List, strategy: str,
                      tenant_id: str = "default",
                      owner_user_id: str = "",
                      index_path: Optional[str] = None,
                      metadata_path: Optional[str] = None,
                      version_bump: bool = True,
                      overwrite_existing: bool = True) -> Dict:
        """写入文档到所有启用的存储后端。（在 write 已持锁的临界区内执行）

        流程:
          1. Clean → Chunk → Embed
          2. 同名文档处理（二选一，由版本化开关决定）:
             - 版本化开启（document_versioning.enabled=true）：新版本写入 +
               活跃指针原子切换 + GC（见 _finalize_versions）
             - 版本化关闭（默认）：**覆盖更新**（overwrite_existing=true 时）——
               同名文档先清旧内容（各端、全策略），再写新，各端一致无孤儿
          3. 写向量后端（Milvus 启用优先 Milvus，失败降级 FAISS）
          4. 写 metadata.json（始终写入，作为降级兜底）
          5. 写 MySQL（如果 storage.backends.mysql.enabled，软失败）
          6. 写 ES（如果 storage.backends.es.enabled，软失败）
          7. 版本收尾（版本化开启时）：原子切换活跃版本指针 + GC 旧版本

        租户隔离:
          - tenant_id 决定索引文件 / MySQL 行 / ES 索引 / Milvus collection 的归属
          - 'default' 租户沿用旧路径/命名（data/index/{strategy} 等），向后兼容
          - 其他租户使用 data/index/{tenant_id}/{strategy}/ 等隔离目录

        文档级 ACL:
          - owner_user_id 记录上传者（documents.owner_user_id），用于文档级授权

        参数:
            documents: app.ingestion.document.Document 列表（未 clean）
            strategy: 分块策略 'fixed'/'recursive'
            tenant_id: 归属租户（默认 'default'）
            owner_user_id: 上传者 user_id（默认 ''，表示存量/共享文档）
            index_path: FAISS 索引输出路径（None 时按租户自动构造）
            metadata_path: metadata.json 输出路径（None 时按租户自动构造）
            version_bump: 是否推进文档版本（版本化开启时；upload=True；rebuild=False）
            overwrite_existing: 版本化关闭时是否覆盖更新同名文档（upload=True；
                rebuild/incremental=False，它们已先全量清理）

        返回:
            {document_count, chunk_count, dimension, documents, chunks,
             index_path, metadata_path, vector_backend,
             mysql_persisted, es_persisted, milvus_persisted}
        """
        from app.embedding.model import EmbeddingModel
        from app.ingestion.cleaner.cleaner import DocumentCleaner
        from app.ingestion.chunker.chunker import Chunker
        from app.ingestion.chunker.recursive_chunker import RecursiveChunker
        from app.storage.metadata_store import MetadataStore

        # 路径默认值（按租户隔离）
        index_dir = self.config.index_dir_for(strategy, tenant_id)
        index_dir.mkdir(parents=True, exist_ok=True)
        if index_path is None:
            index_path = str(index_dir / "faiss.index")
        if metadata_path is None:
            metadata_path = str(index_dir / "metadata.json")

        t_total = time.time()
        logger.info(
            "IndexWriter.write 开始: strategy=%s, tenant=%s, owner=%s, docs=%d, "
            "mysql_enabled=%s, es_enabled=%s, milvus_enabled=%s",
            strategy, tenant_id, owner_user_id or "-", len(documents),
            self.config.storage_mysql_enabled, self.config.storage_es_enabled,
            self.config.storage_milvus_enabled,
        )

        # 1. Clean
        t = time.time()
        cleaned_documents = []
        skipped = 0
        cleaner = DocumentCleaner()
        for doc in documents:
            cleaned = cleaner.clean(doc)
            if not cleaned.content:
                logger.info(
                    "Clean 跳过空文档: doc_id=%s", doc.document_id
                )
                skipped += 1
                continue
            cleaned_documents.append(cleaned)
        logger.info(
            "Clean 完成: %.3fs, 输入=%d, 保留=%d, 跳过=%d",
            time.time() - t, len(documents), len(cleaned_documents), skipped,
        )

        # 2. Chunk
        t = time.time()
        if strategy == "fixed":
            chunker = Chunker(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            logger.info(
                "Chunker 创建: type=Chunker, chunk_size=%d, overlap=%d",
                self.config.chunk_size, self.config.chunk_overlap,
            )
        else:
            chunker = RecursiveChunker(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
            logger.info(
                "Chunker 创建: type=RecursiveChunker, chunk_size=%d, overlap=%d",
                self.config.chunk_size, self.config.chunk_overlap,
            )

        chunks = []
        for doc in cleaned_documents:
            document_chunks = chunker.split(doc)
            chunks.extend(document_chunks)
            logger.info(
                "Chunk 拆分: doc_id=%s, chunks=%d", doc.document_id, len(document_chunks)
            )

        if not chunks:
            logger.error("Chunk 为空，中止写入: strategy=%s", strategy)
            raise ValueError("没有可入库的 Chunk")
        logger.info(
            "Chunk 完成: %.3fs, 总 chunks=%d", time.time() - t, len(chunks)
        )

        # 2b. 文档版本化：解析每文档版本（upload bump / rebuild 保持），
        #     chunk_id 含版本 → 各端隔离（向量/metadata/MySQL/ES 零冲突追加）。
        #     版本化先确保 schema 已迁移（document_versions.strategy / token_version 等），
        #     避免在持久化阶段才 init，导致此处版本解析先报 Unknown column。
        if self._versioning_active():
            self._ensure_schema()
        version_by_doc = self._resolve_doc_versions(
            cleaned_documents, tenant_id, strategy, version_bump
        )
        self._apply_version_to_chunks(chunks, version_by_doc)

        # 2c. 覆盖更新（版本化关闭时）：同名文档先清旧内容（各端、全策略），
        #     再写新 → 各端一致、无孤儿；版本化开启时由版本切换 + GC 处理
        if not self._versioning_active() and overwrite_existing:
            self._overwrite_purge(cleaned_documents, strategy, tenant_id)

        # 2d. 分配稳定向量 ID（chunk_id 哈希派生）——作为 FAISS/Milvus 显式主键，
        #     使向量 id 稳定、删除不影响其余向量（无需重建）
        for c in chunks:
            if not c.vector_id:
                c.vector_id = vector_id_for(c.chunk_id)

        # 3. Embed
        t = time.time()
        embedding_model = EmbeddingModel(self.config.embedding_model)
        texts = [chunk.content for chunk in chunks]
        logger.info(
            "Embedding 开始: model=%s, texts=%d",
            self.config.embedding_model, len(texts),
        )
        vectors = embedding_model.encode(texts)
        logger.info(
            "Embedding 完成: %.3fs, shape=%s",
            time.time() - t, vectors.shape,
        )

        # 4. 写向量后端（Milvus 启用优先 Milvus，否则 FAISS）
        t = time.time()
        use_milvus_backend = self.config.storage_milvus_enabled
        vector_backend = "milvus" if use_milvus_backend else "faiss"
        logger.info(
            "向量写入: backend=%s, milvus_enabled=%s",
            vector_backend, use_milvus_backend,
        )

        milvus_persisted = False
        chunk_ids = [c.vector_id for c in chunks]
        if use_milvus_backend:
            try:
                from app.vector import create_vector_store

                collection_name = self.config.milvus_collection_for(strategy, tenant_id)
                vector_store = create_vector_store(
                    backend="milvus",
                    dimension=vectors.shape[1],
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                    collection_name=collection_name,
                    index_type=self.config.vector_index_type,
                    ivf_nlist=self.config.ivf_nlist,
                    ivf_nprobe=self.config.ivf_nprobe,
                    hnsw_m=self.config.hnsw_m,
                    hnsw_ef_construction=self.config.hnsw_ef_construction,
                    hnsw_ef_search=self.config.hnsw_ef_search,
                )
                # 追加语义：加载已有 collection（存在则复用，不存在则创建），
                # 用稳定 vector_id 显式主键写入，避免与旧向量冲突
                try:
                    vector_store.load(collection_name)
                except Exception as le:
                    logger.info(
                        "Milvus collection 不存在，将新建: collection=%s, error=%s",
                        collection_name, le,
                    )
                vector_store.add(vectors, ids=chunk_ids)
                # save 不写本地文件，仅确保 collection_name 生效 + flush
                vector_store.save(collection_name)
                milvus_persisted = True
                logger.info(
                    "Milvus 写入完成: %.3fs, collection=%s, dim=%d, vectors=%d",
                    time.time() - t, collection_name, vectors.shape[1], vectors.shape[0],
                )
            except Exception as me:
                logger.warning(
                    "Milvus 写入失败，降级到 FAISS: strategy=%s, error=%s: %s",
                    strategy, type(me).__name__, me, exc_info=True,
                )
                use_milvus_backend = False
                vector_backend = "faiss"
                t = time.time()

        if not use_milvus_backend:
            from app.vector import create_vector_store

            vector_store = create_vector_store(
                backend="faiss",
                dimension=vectors.shape[1],
                index_type=self.config.vector_index_type,
                ivf_nlist=self.config.ivf_nlist,
                ivf_nprobe=self.config.ivf_nprobe,
                hnsw_m=self.config.hnsw_m,
                hnsw_ef_construction=self.config.hnsw_ef_construction,
                hnsw_ef_search=self.config.hnsw_ef_search,
            )
            # 追加语义：已有索引文件则加载（保留原向量 + id），再追加本次向量
            if Path(index_path).exists():
                try:
                    vector_store.load(str(index_path))
                except Exception as le:
                    logger.warning(
                        "FAISS 索引加载失败，重建新索引: %s", le,
                    )
            vector_store.add(vectors, ids=chunk_ids)
            vector_store.save(index_path)
            logger.info(
                "FAISS 写入完成: %.3fs, path=%s, dim=%d, vectors=%d",
                time.time() - t, index_path, vectors.shape[1], vectors.shape[0],
            )

        # 5. 写 metadata.json（追加合并：已有条目按 vector_id 保留，新增/覆盖本次）
        t = time.time()
        metadata_store = MetadataStore()
        entries = {}
        if Path(metadata_path).exists():
            try:
                entries = metadata_store.load(str(metadata_path)) or {}
            except Exception as le:
                logger.warning("metadata.json 加载失败，重新写入: %s", le)
                entries = {}
        for c in chunks:
            entries[str(c.vector_id)] = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "vector_id": c.vector_id,
                "version": c.version,
                "content": c.content,
                "start_offset": c.start_offset,
                "end_offset": c.end_offset,
                "metadata": c.metadata,
            }
        metadata_store.save_entries(entries, metadata_path)
        logger.info(
            "metadata.json 写入完成: %.3fs, path=%s, 本次=%d, 累计=%d",
            time.time() - t, metadata_path, len(chunks), len(entries),
        )

        # 6. 写 MySQL（软失败）
        logger.info("MySQL 持久化开始: strategy=%s, tenant=%s", strategy, tenant_id)
        mysql_persisted = self._persist_to_mysql(
            cleaned_documents, chunks, strategy, tenant_id, owner_user_id
        )

        # 7. 写 ES（软失败）
        logger.info("ES 持久化开始: strategy=%s, tenant=%s", strategy, tenant_id)
        es_persisted = self._persist_to_es(chunks, strategy, tenant_id)

        result = {
            "document_count": len(cleaned_documents),
            "chunk_count": len(chunks),
            "dimension": vectors.shape[1],
            "documents": cleaned_documents,
            "chunks": chunks,
            "index_path": index_path,
            "metadata_path": metadata_path,
            "tenant_id": tenant_id,
            "vector_backend": vector_backend,
            "mysql_persisted": mysql_persisted,
            "es_persisted": es_persisted,
            "milvus_persisted": milvus_persisted,
        }

        logger.info(
            "IndexWriter.write 完成: %.3fs, strategy=%s, tenant=%s, vector_backend=%s, docs=%d, chunks=%d, "
            "mysql=%s, es=%s, milvus=%s",
            time.time() - t_total, strategy, tenant_id, vector_backend,
            len(cleaned_documents), len(chunks),
            mysql_persisted, es_persisted, milvus_persisted,
        )
        # 版本收尾：原子切换活跃版本指针 + GC 旧版本（retention=latest 同步 GC）
        # —— Build → Validation → Publish 分离：校验通过才允许发布版本
        self._validate_build(chunks, strategy, tenant_id)
        self._finalize_versions(version_by_doc, strategy, tenant_id)
        self._bump_version(strategy, tenant_id)
        return result

    # ---- 幂等重建 ----
    def rebuild(self, strategy: str,
                tenant_id: str = "default",
                owner_user_id: str = "",
                index_path: Optional[str] = None,
                metadata_path: Optional[str] = None) -> Dict:
        """幂等重建索引：先清理旧数据，再全量写入。

        外层持（strategy, tenant）写锁（P1-5：与 upload 等写入互斥）。
        """
        with self._write_lock(strategy, tenant_id):
            return self._rebuild_locked(
                strategy, tenant_id, owner_user_id, index_path, metadata_path,
            )

    def _rebuild_locked(self, strategy: str,
                        tenant_id: str = "default",
                        owner_user_id: str = "",
                        index_path: Optional[str] = None,
                        metadata_path: Optional[str] = None) -> Dict:
        """幂等重建索引：先清理旧数据，再全量写入。（在已持锁临界区内执行）:

        清理顺序:
          1. MySQL: chunk_repo.delete_by_strategy(strategy, tenant_id)
          2. ES: es_client.drop_index(strategy)（租户索引）
          3. Milvus: milvus_store.drop(tenant 对应的 collection)

        然后加载 data/raw/{tenant_id}/ 下所有文档，调用 write()。

        参数:
            strategy: 分块策略
            tenant_id: 归属租户（默认 'default'，使用旧目录 data/raw/）
            index_path / metadata_path: 见 write()

        返回:
            write() 的返回 + cleaned 清理计数
        """
        t_total = time.time()
        logger.info(
            "IndexWriter.rebuild 开始: strategy=%s, tenant=%s, mysql_enabled=%s, es_enabled=%s, "
            "milvus_enabled=%s",
            strategy, tenant_id, self.config.storage_mysql_enabled, self.config.storage_es_enabled,
            self.config.storage_milvus_enabled,
        )

        # 1. 幂等清理
        logger.info("rebuild 步骤1: 幂等清理旧数据, strategy=%s, tenant=%s", strategy, tenant_id)
        mysql_deleted = self._cleanup_mysql(strategy, tenant_id)
        es_dropped = self._cleanup_es(strategy, tenant_id)
        milvus_dropped = self._cleanup_milvus(strategy, tenant_id)
        # 本地 FAISS 索引 + metadata.json 也一并清掉（write 是追加语义，需从空开始）
        index_dir = self.config.index_dir_for(strategy, tenant_id)
        for fname in ("faiss.index", "metadata.json"):
            f = index_dir / fname
            try:
                if f.exists():
                    f.unlink()
            except Exception as e:
                logger.warning("清理本地索引文件失败: %s, %s", f, e)
        logger.info(
            "rebuild 清理完成: mysql_deleted=%d, es_dropped=%s, milvus_dropped=%s, local_index=cleared",
            mysql_deleted, es_dropped, milvus_dropped,
        )

        # 2. 加载所有文档（租户目录）
        logger.info("rebuild 步骤2: 加载 data/raw 下所有文档 (tenant=%s)", tenant_id)
        documents = self._load_all_documents(tenant_id)
        logger.info("rebuild 文档加载完成: %d 个", len(documents))

        # 3. 全量写入
        logger.info("rebuild 步骤3: 全量写入")
        result = self.write(
            documents=documents,
            strategy=strategy,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            index_path=index_path,
            metadata_path=metadata_path,
            version_bump=False,
            overwrite_existing=False,
        )
        result["mysql_deleted"] = mysql_deleted
        result["es_dropped"] = es_dropped
        result["milvus_dropped"] = milvus_dropped

        logger.info(
            "IndexWriter.rebuild 完成: %.3fs, strategy=%s, tenant=%s, vector_backend=%s, "
            "mysql_deleted=%d, es_dropped=%s, milvus_dropped=%s, "
            "docs=%d, chunks=%d, mysql=%s, es=%s, milvus=%s",
            time.time() - t_total, strategy, tenant_id, result.get("vector_backend", "faiss"),
            mysql_deleted, es_dropped, milvus_dropped,
            result["document_count"], result["chunk_count"],
            result["mysql_persisted"], result["es_persisted"], result.get("milvus_persisted", False),
        )
        return result

    def incremental_rebuild_after_delete(
        self,
        strategy: str,
        deleted_doc_ids: List[str],
        tenant_id: str = "default",
        owner_user_id: str = "",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> Dict:
        """删除文档后的增量重建：ES/MySQL 按文档级增量清理，向量层（FAISS/Milvus）+ metadata 重写。

        外层持（strategy, tenant）写锁（P1-5）。
        """
        with self._write_lock(strategy, tenant_id):
            return self._incremental_rebuild_locked(
                strategy, deleted_doc_ids, tenant_id, owner_user_id,
                index_path, metadata_path,
            )

    def _incremental_rebuild_locked(
        self,
        strategy: str,
        deleted_doc_ids: List[str],
        tenant_id: str = "default",
        owner_user_id: str = "",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> Dict:
        """删除文档后的增量重建（在已持锁临界区内执行）。:

        相对 rebuild() 的优化：
          - **跳过 MySQL `delete_by_strategy`**（上层 knowledge.py 已按文档级 delete_by_document 清完，
            避免对保留的文档 chunks 做无用的全表删 + 重新批量插入）
          - ES 走 `incremental_reindex(deleted_doc_ids=...)`（delete_by_query 只删被移除文档的 chunks）
          - Milvus 仍必须 drop（向量层单独删除会破坏 auto_id 与 enumerate 顺序一致性）
          - FAISS/metadata.json 仍必须全量重写（理由同上：枚举索引必须与剩余文件 0..N-1 对应）
          - 最后重新加载 data/raw/{tenant_id}/ 下剩余文件 + write（write 内部会 insert_ignore MySQL/ES，幂等）

        参数:
            strategy: 分块策略
            deleted_doc_ids: 被删除的 document_id 列表（ES 增量清理用）
            tenant_id: 归属租户（默认 'default'）
            index_path / metadata_path: 见 write()

        返回:
            write() 的返回 + cleaned 标志
        """
        if not deleted_doc_ids:
            logger.warning(
                "incremental_rebuild_after_delete: deleted_doc_ids 为空，"
                "直接走正常 rebuild 以保证一致性"
            )
            return self.rebuild(
                strategy, tenant_id=tenant_id, owner_user_id=owner_user_id,
                index_path=index_path, metadata_path=metadata_path,
            )

        t_total = time.time()
        logger.info(
            "IndexWriter.incremental_rebuild_after_delete 开始: strategy=%s, tenant=%s, "
            "deleted_docs=%s, mysql_enabled=%s, es_enabled=%s, milvus_enabled=%s",
            strategy, tenant_id, deleted_doc_ids,
            self.config.storage_mysql_enabled, self.config.storage_es_enabled,
            self.config.storage_milvus_enabled,
        )

        # 1. 增量清理
        # MySQL：外层已按 document_id 级 delete_by_document + delete(文档) 完成，**跳过** delete_by_strategy
        mysql_deleted: int = 0  # 仅用于日志对照，knowledge.py 里已经做过
        es_deleted = self._cleanup_es_incremental(strategy, deleted_doc_ids, tenant_id)
        milvus_dropped = self._cleanup_milvus(strategy, tenant_id)
        logger.info(
            "增量清理完成: mysql(跳过,外层已清)=0, es_incremental_deleted=%s, milvus_dropped=%s",
            es_deleted, milvus_dropped,
        )

        # 2. 加载剩余文档（data/raw/{tenant_id}/ 中已没有被删除的文件，knowledge.py 已同步删）
        logger.info("增量重建 步骤2: 加载 data/raw 下剩余文档 (tenant=%s)", tenant_id)
        try:
            documents = self._load_all_documents(tenant_id)
        except FileNotFoundError:
            # 全部文档被删光：保持空索引（不抛错），返回空结果
            logger.warning("增量重建: data/raw 已无文档，生成空索引 (tenant=%s)", tenant_id)
            documents = []
        logger.info("增量重建 文档加载完成: 剩余 %d 个", len(documents))

        # 3. 全量写入（剩余文件 → 新 FAISS/Milvus + metadata；MySQL/ES insert_ignore 幂等）
        if documents:
            logger.info("增量重建 步骤3: 全量写入剩余文档")
            result = self.write(
                documents=documents,
                strategy=strategy,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                index_path=index_path,
                metadata_path=metadata_path,
            )
        else:
            # 无文档：删除已有向量索引文件 / ES 索引 / Milvus collection，保证零残留
            logger.warning("增量重建: 无剩余文档，清理空索引残留")
            self._cleanup_es(strategy, tenant_id)
            self._cleanup_milvus(strategy, tenant_id)
            index_dir = self.config.index_dir_for(strategy, tenant_id)
            if index_path is None:
                index_path = str(index_dir / "faiss.index")
            if metadata_path is None:
                metadata_path = str(index_dir / "metadata.json")
            try:
                if Path(index_path).exists():
                    Path(index_path).unlink()
            except Exception as e:
                logger.warning("清理空 FAISS 索引失败: %s", e)
            # metadata.json 写空
            from app.storage.metadata_store import MetadataStore
            MetadataStore().save([], metadata_path)
            result = {
                "document_count": 0,
                "chunk_count": 0,
                "dimension": 0,
                "documents": [],
                "chunks": [],
                "index_path": index_path,
                "metadata_path": metadata_path,
                "tenant_id": tenant_id,
                "vector_backend": "none",
                "mysql_persisted": False,
                "es_persisted": False,
                "milvus_persisted": False,
            }
            # 全文档删除也属于索引变更：登记版本（未走 write 的 bump）
            self._bump_version(strategy, tenant_id)

        result["mysql_deleted"] = mysql_deleted
        result["es_deleted_incremental"] = es_deleted
        result["milvus_dropped"] = milvus_dropped
        result["deleted_doc_ids"] = list(deleted_doc_ids)

        logger.info(
            "IndexWriter.incremental_rebuild_after_delete 完成: %.3fs, strategy=%s, tenant=%s, "
            "vector_backend=%s, es_incremental=%s, milvus_dropped=%s, "
            "remaining_docs=%d, chunks=%d",
            time.time() - t_total, strategy, tenant_id, result.get("vector_backend", "none"),
            es_deleted, milvus_dropped,
            result["document_count"], result["chunk_count"],
        )
        return result

    def remove_document(self, strategy: str, tenant_id: str,
                        document_id: str, vector_ids: List[int],
                        raise_on_orphan: bool = True) -> List[str]:
        """删除文档后，从向量后端 + metadata.json + ES 移除对应数据（无需重建索引）。

        外层持（strategy, tenant）写锁（P1-4/P1-5）。
        """
        with self._write_lock(strategy, tenant_id):
            return self._remove_document_locked(
                strategy, tenant_id, document_id, vector_ids, raise_on_orphan,
            )

    def _remove_document_locked(self, strategy: str, tenant_id: str,
                                document_id: str, vector_ids: List[int],
                                raise_on_orphan: bool = True) -> List[str]:
        """删除文档后，从向量后端 + metadata.json + ES 移除对应数据（无需重建索引）。

        稳定 ID 索引：向量按显式 vector_id 删除，其余向量 id 不变；
        metadata.json 按 str(vector_id) 摘除对应条目。

        **孤儿可观测**（P1-1）：各后端移除后做校验，任何移除失败/确认仍残留
        都会收集进失败列表并返回——不再静默 warning。为避免孤儿，向量未移除时
        metadata 与 ES 仍继续清理（尽量清干净能清的），最后汇总失败。

        参数:
            strategy: 分块策略
            tenant_id: 租户
            document_id: 文档 ID（ES 按文档删除用）
            vector_ids: 该文档 chunk 的稳定向量 ID 列表
            raise_on_orphan: 存在孤儿清理失败时是否抛异常（默认 True 使调用方可观测）

        返回:
            失败描述列表（空字符串列表表示清理完整、无孤儿）。
        """
        vector_ids = [int(v) for v in vector_ids if v]
        logger.info(
            "remove_document: strategy=%s, tenant=%s, doc=%s, vector_ids=%d",
            strategy, tenant_id, document_id, len(vector_ids),
        )
        failures: List[str] = []

        # 1. 向量后端（Milvus 优先，否则 FAISS）——移除后校验，确认无残留
        if vector_ids:
            failures.extend(self._remove_vectors_verified(vector_ids, strategy, tenant_id))

        # 2. metadata.json：按 vector_id 摘除条目
        if vector_ids:
            try:
                self._remove_metadata_by_ids(vector_ids, strategy, tenant_id)
            except Exception as e:
                failures.append("metadata.json 移除条目失败: {}".format(e))

        # 3. ES：按文档删除 chunks
        if self.config.storage_es_enabled:
            try:
                from app.storage.es_repository import ChunkESRepository

                es_repo = ChunkESRepository(strategy=strategy, tenant_id=tenant_id)
                es_repo.incremental_reindex(chunks=[], deleted_doc_ids=[document_id])
                logger.info(
                    "ES 删除文档 chunks: strategy=%s, doc=%s",
                    strategy, document_id,
                )
            except Exception as e:
                failures.append("ES 删除文档 chunks 失败: {}".format(e))

        # 4. 索引版本 +1（数据库唯一权威源；strict：登记失败则删除操作失败）
        self._bump_version(strategy, tenant_id)

        if failures:
            msg = "remove_document 清理不完整（存在孤儿，需对账/重建修复）:" \
                  "[{}] {}".format(strategy, "; ".join(failures))
            logger.error("%s —— doc=%s, tenant=%s", msg, document_id, tenant_id)
            if raise_on_orphan:
                raise RuntimeError(msg)
        else:
            logger.info(
                "remove_document 清理完整（无孤儿）: doc=%s, vector_ids=%d, strategy=%s",
                document_id, len(vector_ids), strategy,
            )
        return failures
