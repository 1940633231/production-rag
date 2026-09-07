"""统一索引写入器：封装 chunk → vector store（FAISS / Milvus）+ MySQL + ES 三路分发。

职责:
  - write(documents, strategy): clean → chunk → embed → 写入 FAISS/Milvus + metadata.json + MySQL + ES
  - rebuild(strategy): 幂等重建（先清理旧数据，再全量写入）
  - incremental_rebuild_after_delete(strategy, deleted_doc_ids): 删除文档后的增量重建
  - persist_to_mysql() / persist_to_es(): 软失败持久化（内部方法）

设计原则:
  - metadata.json 始终写入（降级兜底）
  - 向量后端：storage_milvus_enabled=true 时用 Milvus，失败降级到 FAISS
  - MySQL / ES 软失败：写入异常时记 warning 不中断流程
  - 重建时先 delete_by_strategy + es drop_index + milvus drop 保证幂等
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import Config
from app.core.logger import get_logger
from app.ingestion.chunk import vector_id_for

logger = get_logger(__name__)


class IndexWriter:
    """统一索引写入器：所有写入入口的唯一收敛点。

    替代 knowledge.py 中 _run_ingestion / _do_upload / _do_rebuild / _rebuild_silent
    四处重复代码 + pipeline.py 的 _persist_to_mysql 软失败逻辑。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    # ---- 核心写入 ----

    def write(self, documents: List, strategy: str,
              tenant_id: str = "default",
              owner_user_id: str = "",
              index_path: Optional[str] = None,
              metadata_path: Optional[str] = None,
              version_bump: bool = True,
              overwrite_existing: bool = True) -> Dict:
        """写入文档到所有启用的存储后端。

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
        #     chunk_id 含版本 → 各端隔离（向量/metadata/MySQL/ES 零冲突追加）
        version_by_doc = self._resolve_doc_versions(
            cleaned_documents, tenant_id, version_bump
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
                        document_id: str, vector_ids: List[int]) -> None:
        """删除文档后，从向量后端 + metadata.json + ES 移除对应数据（无需重建索引）。

        稳定 ID 索引：向量按显式 vector_id 删除，其余向量 id 不变；
        metadata.json 按 str(vector_id) 摘除对应条目。

        参数:
            strategy: 分块策略
            tenant_id: 租户
            document_id: 文档 ID（ES 按文档删除用）
            vector_ids: 该文档 chunk 的稳定向量 ID 列表
        """
        vector_ids = [int(v) for v in vector_ids if v]
        logger.info(
            "remove_document: strategy=%s, tenant=%s, doc=%s, vector_ids=%d",
            strategy, tenant_id, document_id, len(vector_ids),
        )

        # 1. 向量后端（Milvus 优先，否则 FAISS）——按 vector_id 删除
        if vector_ids and self.config.storage_milvus_enabled:
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
            except Exception as e:
                logger.warning(
                    "Milvus 移除向量失败（可稍后重建索引修复）: %s", e,
                )
        elif vector_ids:
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
                    "FAISS 移除向量失败（可稍后重建索引修复）: %s", e,
                )

        # 2. metadata.json：按 vector_id 摘除条目
        if vector_ids:
            try:
                from app.storage.metadata_store import MetadataStore

                meta_path = self.config.index_dir_for(strategy, tenant_id) / "metadata.json"
                if Path(meta_path).exists():
                    ms = MetadataStore()
                    entries = ms.load(str(meta_path)) or {}
                    for vid in vector_ids:
                        entries.pop(str(vid), None)
                    ms.save_entries(entries, str(meta_path))
                    logger.info(
                        "metadata.json 移除条目: path=%s, 剩余=%d",
                        meta_path, len(entries),
                    )
            except Exception as e:
                logger.warning(
                    "metadata.json 移除条目失败（可稍后重建索引修复）: %s", e,
                )

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
                logger.warning(
                    "ES 删除文档 chunks 失败（可稍后重建索引修复）: %s", e,
                )

        # 4. 索引版本 +1（数据库唯一权威源；strict：登记失败则删除操作失败）
        self._bump_version(strategy, tenant_id)

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

    def _resolve_doc_versions(self, documents, tenant_id: str,
                              bump: bool) -> Dict:
        """解析每文档目标版本：{document_id: version}。

        - bump=True（upload）：已存在文档 → current_version+1（新版本）；
          新文档 → 1
        - bump=False（rebuild/incremental）：保持当前版本（不 bump，
          保证 rebuild 后 vector_id/chunk_id 稳定，ACL 不丢）
        - 版本化未生效：全部返回 1（chunk_id 保持旧格式，行为零变化）
        """
        versions = {d.document_id: 1 for d in documents}
        if not self._versioning_active():
            return versions
        from app.storage import DocumentRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        doc_repo = DocumentRepository(mgr)
        for doc in documents:
            cur = doc_repo.get_current_version(doc.document_id, tenant_id)
            if cur is None:
                versions[doc.document_id] = 1
            elif bump:
                versions[doc.document_id] = cur + 1
            else:
                versions[doc.document_id] = cur
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

    def _finalize_versions(self, version_by_doc: Dict,
                           strategy: str, tenant_id: str) -> None:
        """写入完成后收尾：原子切换活跃版本指针 + GC 旧版本。

        P1 仅支持 retention=latest：切换后同步 GC 旧版本各端数据，
        保证"上传完成 = 只剩活跃版本"，检索无需版本过滤。
        """
        if not self._versioning_active():
            return
        if self.config.document_versioning_retention != "latest":
            logger.warning(
                "document_versioning.retention=%s 暂不支持（P1 仅 latest），跳过版本收尾",
                self.config.document_versioning_retention,
            )
            return
        from app.storage import ChunkRepository, DocumentRepository
        from app.storage.mysql import MySQLManager

        mgr = MySQLManager(pool_size=self.config.storage_pool_size)
        doc_repo = DocumentRepository(mgr)
        chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)
        for doc_id, new_ver in version_by_doc.items():
            old_ver = new_ver - 1
            if old_ver < 1:
                continue  # 新文档：documents.current_version 已默认 1
            # 原子切换（乐观锁：current_version 仍为旧值才切，防并发覆盖）
            ok = doc_repo.set_current_version(
                doc_id, new_ver, expected_version=old_ver, tenant_id=tenant_id
            )
            if not ok:
                logger.warning(
                    "版本切换冲突（并发上传），跳过 GC: doc=%s, v=%s",
                    doc_id, new_ver,
                )
                continue
            self._gc_version(doc_id, old_ver, strategy, tenant_id, chunk_repo)

    def _gc_version(self, doc_id: str, version: int, strategy: str,
                    tenant_id: str, chunk_repo) -> None:
        """删除旧版本各端数据（向量/metadata/MySQL/ES）。失败仅告警。

        遍历 fixed + recursive 两个分块策略清理（同一逻辑文档可能同时存在于
        两种策略索引，且跨策略的 chunk_id/vector_id 相同）：
          - MySQL：delete_by_document_version 无 strategy 条件 → 一次全删
          - 向量/metadata/ES：按策略逐个清理（remove 对不存在 id 幂等安全）
        """
        try:
            vector_ids = chunk_repo.get_vector_ids_by_document(
                doc_id, tenant_id=tenant_id, version=version
            )
            for strat in ("fixed", "recursive"):
                try:
                    if vector_ids:
                        self._remove_vectors(vector_ids, strat, tenant_id)
                    self._remove_metadata(doc_id, version, strat, tenant_id)
                    if self.config.storage_es_enabled:
                        from app.storage.es_repository import ChunkESRepository
                        ChunkESRepository(
                            strategy=strat, tenant_id=tenant_id
                        ).delete_by_document_version(doc_id, version)
                except Exception as se:
                    logger.warning(
                        "版本 GC 失败（strategy=%s，残留可重建修复）: doc=%s, v=%s, %s",
                        strat, doc_id, version, se, exc_info=True,
                    )
            # MySQL：无 strategy 条件，一次删除该文档该版本的全部 chunks
            chunk_repo.delete_by_document_version(doc_id, version, tenant_id)
            logger.info(
                "版本 GC 完成: doc=%s, v=%s, vectors=%d, strategies=fixed,recursive",
                doc_id, version, len(vector_ids),
            )
        except Exception as e:
            logger.warning(
                "版本 GC 失败（残留数据，可重建修复）: doc=%s, v=%s, %s",
                doc_id, version, e, exc_info=True,
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
        """清空某文档各端旧内容（全策略、全版本），documents 行保留（ACL 不丢）。"""
        try:
            vector_ids = chunk_repo.get_vector_ids_by_document(
                doc_id, tenant_id=tenant_id
            )
            for strat in ("fixed", "recursive"):
                try:
                    if vector_ids:
                        self._remove_vectors(vector_ids, strat, tenant_id)
                    self._remove_metadata_document(doc_id, strat, tenant_id)
                    if self.config.storage_es_enabled:
                        from app.storage.es_repository import ChunkESRepository
                        ChunkESRepository(
                            strategy=strat, tenant_id=tenant_id
                        ).incremental_reindex(
                            chunks=[], deleted_doc_ids=[doc_id]
                        )
                except Exception as se:
                    logger.warning(
                        "覆盖清理失败（strategy=%s，孤儿可重建修复）: doc=%s, %s",
                        strat, doc_id, se, exc_info=True,
                    )
            # MySQL chunks：删除失败抛异常（中止写入，避免半覆盖不一致）
            chunk_repo.delete_by_document(doc_id, tenant_id)
            logger.info(
                "覆盖更新清理完成: doc=%s, vectors=%d, strategies=fixed,recursive",
                doc_id, len(vector_ids),
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
        """将文档和 chunks 写入 MySQL（如果启用）。

        租户隔离：documents/chunks 均写入 tenant_id。
        文档级 ACL：documents 写入 owner_user_id（上传者）。
        软失败：异常时记 warning 返回 False，不中断写入流程。
        """
        if not self.config.storage_mysql_enabled:
            logger.info("MySQL 持久化跳过: storage_mysql_enabled=false")
            return False

        t = time.time()
        try:
            from app.storage import DocumentRepository, ChunkRepository
            from app.storage.mysql import MySQLManager

            logger.info(
                "MySQL 连接: pool_size=%d, host=%s",
                self.config.storage_pool_size,
                "from env",
            )
            mgr = MySQLManager(pool_size=self.config.storage_pool_size)
            try:
                mgr.init_schema()
            except Exception as e:
                logger.info("MySQL 连接失败: {}。请确认 MySQL 服务运行中且环境变量已配置。".format(e))
                return False

            doc_repo = DocumentRepository(mgr)
            chunk_repo = ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)

            # 逐个插入文档，记录每个结果
            doc_inserted = 0
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
                    doc_inserted += 1
                except Exception as de:
                    logger.warning(
                        "MySQL 文档插入失败: doc_id=%s, tenant=%s, %s",
                        doc.document_id, tenant_id, de, exc_info=True,
                    )
            logger.info(
                "MySQL 文档插入: 成功=%d/%d", doc_inserted, len(documents)
            )

            # 批量插入 chunks
            try:
                chunk_repo.batch_insert(chunks)
                logger.info(
                    "MySQL chunks 批量插入成功: strategy=%s, tenant=%s, chunks=%d",
                    strategy, tenant_id, len(chunks),
                )
            except Exception as ce:
                logger.warning(
                    "MySQL chunks 批量插入失败: strategy=%s, tenant=%s, chunks=%d, %s",
                    strategy, tenant_id, len(chunks), ce, exc_info=True,
                )
                raise

            logger.info(
                "MySQL 持久化完成: %.3fs, strategy=%s, tenant=%s, docs=%d, chunks=%d",
                time.time() - t, strategy, tenant_id, doc_inserted, len(chunks),
            )
            return True
        except Exception as e:
            logger.warning(
                "MySQL 持久化失败（不影响索引）: strategy=%s, tenant=%s, docs=%d, chunks=%d, "
                "error=%s: %s",
                strategy, tenant_id, len(documents), len(chunks),
                type(e).__name__, e, exc_info=True,
            )
            return False

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

    # ---- 内部：文档加载 ----

    def _load_all_documents(self, tenant_id: str = "default") -> List:
        """加载某租户 data/raw/ 目录下所有支持的文档。"""
        import importlib

        raw_dir = self.config.raw_dir_for(tenant_id)
        if not raw_dir.exists():
            logger.error("data/raw 目录不存在 (tenant=%s): %s", tenant_id, raw_dir)
            raise FileNotFoundError("data/raw 目录不存在 (tenant={})".format(tenant_id))

        loader_map = {
            ".txt": "app.ingestion.loader.txt_loader.TxtLoader",
            ".html": "app.ingestion.loader.html_loader.HtmlLoader",
            ".pdf": "app.ingestion.loader.pdf_loader.PdfLoader",
            ".docx": "app.ingestion.loader.word_loader.WordLoader",
        }

        files = [f for f in raw_dir.iterdir() if f.suffix.lower() in loader_map]
        logger.info(
            "扫描 data/raw/: 找到 %d 个可处理文件: %s",
            len(files), [f.name for f in files],
        )
        if not files:
            logger.error("data/raw/ 下无可处理的文件")
            raise FileNotFoundError("data/raw/ 下无可处理的文件")

        documents = []
        for f in files:
            loader_cls_path = loader_map[f.suffix.lower()]
            module_path, cls_name = loader_cls_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            loader_cls = getattr(module, cls_name)
            loader = loader_cls()
            doc = loader.load(str(f))
            documents.append(doc)
            logger.info(
                "文档加载: file=%s, doc_id=%s, content_len=%d",
                f.name, doc.document_id, len(doc.content),
            )

        logger.info("全部文档加载完成: %d 个", len(documents))
        return documents

    def _load_single_document(self, file_path: Path):
        """加载单个文档。"""
        import importlib

        loader_map = {
            ".txt": "app.ingestion.loader.txt_loader.TxtLoader",
            ".html": "app.ingestion.loader.html_loader.HtmlLoader",
            ".pdf": "app.ingestion.loader.pdf_loader.PdfLoader",
            ".docx": "app.ingestion.loader.word_loader.WordLoader",
        }

        loader_cls_path = loader_map.get(file_path.suffix.lower())
        if not loader_cls_path:
            logger.error("不支持的文件类型: %s", file_path.suffix)
            raise ValueError("不支持的文件类型: {}".format(file_path.suffix))
        module_path, cls_name = loader_cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        loader_cls = getattr(module, cls_name)
        loader = loader_cls()
        doc = loader.load(str(file_path))
        logger.info(
            "单文档加载: file=%s, doc_id=%s, content_len=%d",
            file_path.name, doc.document_id, len(doc.content),
        )
        return doc
