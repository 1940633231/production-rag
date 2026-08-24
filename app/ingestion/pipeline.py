import time

from app.embedding.model import EmbeddingModel
from app.ingestion.chunker.base import BaseChunker
from app.vector import create_vector_store
from app.storage.metadata_store import MetadataStore
from app.core.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """文档入库流水线：clean → chunk → embed → 持久化。

    持久化通过 IndexWriter 统一分发到 MySQL / ES（软失败），
    FAISS 索引 + metadata.json 始终写入。
    """

    def __init__(self, embedding_model, metadata_store, chunker, cleaner,
                 strategy="recursive", config=None):

        self.embedding_model = embedding_model

        self.metadata_store = metadata_store

        self.chunker = chunker

        self.cleaner = cleaner

        self.strategy = strategy

        self.config = config

    def ingest(self, documents, index_path, metadata_path):
        """执行入库流程并持久化到所有后端。

        参数:
            documents: 未 clean 的 Document 列表
            index_path: FAISS 索引输出路径
            metadata_path: metadata.json 输出路径

        返回:
            {document_count, chunk_count, dimension, documents, chunks,
             mysql_persisted, es_persisted}
        """
        t_total = time.time()
        logger.info(
            "IngestionPipeline.ingest 开始: strategy=%s, docs=%d, "
            "index=%s, metadata=%s",
            self.strategy, len(documents), index_path, metadata_path,
        )

        # ==========================
        # 1. Clean
        # ==========================
        t = time.time()
        cleaned_documents = []
        skipped = 0

        for document in documents:
            cleaned = self.cleaner.clean(document)
            if not cleaned.content:
                logger.info("Clean 跳过空文档: doc_id=%s", document.document_id)
                skipped += 1
                continue
            cleaned_documents.append(cleaned)

        logger.info(
            "Clean 完成: %.3fs, 输入=%d, 保留=%d, 跳过=%d",
            time.time() - t, len(documents), len(cleaned_documents), skipped,
        )

        # ==========================
        # 2. Chunk
        # ==========================
        t = time.time()
        chunks = []

        for document in cleaned_documents:
            document_chunks = self.chunker.split(document)
            chunks.extend(document_chunks)
            logger.info(
                "Chunk 拆分: doc_id=%s, chunks=%d",
                document.document_id, len(document_chunks),
            )

        if not chunks:
            logger.error("Chunk 为空，中止入库: strategy=%s", self.strategy)
            raise ValueError("没有可入库的 Chunk")

        logger.info(
            "Chunk 完成: %.3fs, 总 chunks=%d", time.time() - t, len(chunks)
        )

        # ==========================
        # 3. Embedding
        # ==========================
        t = time.time()
        texts = [chunk.content for chunk in chunks]
        logger.info("Embedding 开始: texts=%d", len(texts))
        vectors = self.embedding_model.encode(texts)
        logger.info(
            "Embedding 完成: %.3fs, shape=%s", time.time() - t, vectors.shape
        )

        # ==========================
        # 4. Vector Store（FAISS + metadata.json）
        # ==========================
        t = time.time()
        vector_store = create_vector_store(
            backend="faiss", dimension=vectors.shape[1]
        )
        vector_store.add(vectors)
        vector_store.save(index_path)
        logger.info(
            "FAISS 写入完成: %.3fs, path=%s, dim=%d, vectors=%d",
            time.time() - t, index_path, vectors.shape[1], vectors.shape[0],
        )

        t = time.time()
        self.metadata_store.save(chunks, metadata_path)
        logger.info(
            "metadata.json 写入完成: %.3fs, path=%s, chunks=%d",
            time.time() - t, metadata_path, len(chunks),
        )

        # ==========================
        # 5. 持久化到 MySQL / ES（通过 IndexWriter 统一分发，软失败）
        # ==========================
        mysql_persisted = False
        es_persisted = False
        try:
            from app.ingestion.writer import IndexWriter

            writer = IndexWriter(self.config)
            logger.info("MySQL 持久化开始: strategy=%s", self.strategy)
            mysql_persisted = writer._persist_to_mysql(
                cleaned_documents, chunks, self.strategy
            )
            logger.info("ES 持久化开始: strategy=%s", self.strategy)
            es_persisted = writer._persist_to_es(chunks, self.strategy)
        except Exception as e:
            logger.warning(
                "IndexWriter 持久化失败（不影响索引）: strategy=%s, "
                "error=%s: %s",
                self.strategy, type(e).__name__, e, exc_info=True,
            )

        logger.info(
            "IngestionPipeline.ingest 完成: %.3fs, strategy=%s, docs=%d, "
            "chunks=%d, dim=%d, mysql=%s, es=%s",
            time.time() - t_total, self.strategy,
            len(cleaned_documents), len(chunks), vectors.shape[1],
            mysql_persisted, es_persisted,
        )

        return {
            "document_count": len(cleaned_documents),
            "chunk_count": len(chunks),
            "dimension": vectors.shape[1],
            "documents": cleaned_documents,
            "chunks": chunks,
            "mysql_persisted": mysql_persisted,
            "es_persisted": es_persisted,
        }
