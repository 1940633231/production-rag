"""RAG Service：从 config 初始化所有组件，暴露高层 query() 接口。

职责：
- 从 Config 读取参数，初始化 Retriever/Reranker/ContextManager 并组装 RAGPipeline
- 暴露 query(query) → RAGResponse 高层接口，供 API/脚本调用
- 模块级缓存（get_service）复用模型加载（embedding/cross-encoder 加载耗时）

import 策略：
- 顶部只 import 标准库 + 不依赖外部库的 app 模块（Config/RAGPipeline/RAGResponse）
- 检索器/reranker 的 import 延迟到 _build_pipeline() 内，让 bm25 模式不需要 numpy/faiss
"""
import threading
import time
from typing import Iterator, List, Optional

from app.core.config import Config
from app.core.logger import get_logger
from app.rag.pipeline import RAGPipeline
from app.rag.result import RAGResponse

logger = get_logger(__name__)


class RAGService:
    """RAG 服务层：从 config 初始化组件，暴露 query() 接口。

    tenant_id：租户隔离维度。决定索引文件 / Milvus collection / chunk 仓储
    的数据归属，使不同租户的问答互不可见（permission-aware retrieval）。
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        strategy: str = "recursive",
        mode: str = "vector",
        use_rerank: bool = True,
        tenant_id: str = "default",
    ):
        logger.info(
            "RAGService 初始化: strategy=%s, mode=%s, use_rerank=%s, tenant=%s",
            strategy, mode, use_rerank, tenant_id,
        )
        self.config = config or Config()
        self.strategy = strategy
        self.mode = mode
        self.use_rerank = use_rerank
        self.tenant_id = tenant_id
        self._pipeline: RAGPipeline = self._build_pipeline()
        logger.info("RAGService 初始化完成")

    def _build_pipeline(self) -> RAGPipeline:
        """从 config 初始化所有组件并构建 pipeline。"""
        logger.info("开始构建 pipeline (tenant=%s)", self.tenant_id)
        build_start = time.time()

        # ---- 向量后端判定 & 路径（按租户隔离）----
        use_milvus = self.config.storage_milvus_enabled
        vector_backend = "milvus" if use_milvus else "faiss"
        index_dir = self.config.index_dir_for(self.strategy, self.tenant_id)
        faiss_index_path = index_dir / "faiss.index"
        metadata_path = index_dir / "metadata.json"
        milvus_collection = (
            self.config.milvus_collection_for(self.strategy, self.tenant_id)
            if use_milvus else None
        )
        logger.info(
            "向量后端: backend=%s, strategy=%s, tenant=%s, milvus_collection=%s, faiss_index=%s",
            vector_backend, self.strategy, self.tenant_id, milvus_collection, faiss_index_path,
        )

        # ---- Chunk Repository（多后端：MySQL 优先，降级到 metadata.json）----
        chunk_repo = None
        try:
            from app.storage import create_chunk_repo
            chunk_repo = create_chunk_repo(
                self.config, strategy=self.strategy, tenant_id=self.tenant_id
            )
            if chunk_repo is not None:
                t = time.time()
                chunk_count = chunk_repo.count()
                logger.info(
                    "ChunkRepository 初始化完成: %.3fs, chunks=%d, backend=%s, tenant=%s",
                    time.time() - t, chunk_count, type(chunk_repo).__name__,
                    self.tenant_id,
                )
        except Exception as e:
            logger.warning("存储后端不可用，将降级到 metadata.json: %s", e)
            chunk_repo = None

        # 降级：从 metadata.json 加载并包装为 MetadataChunkRepository
        if chunk_repo is None:
            if not metadata_path.exists():
                logger.error("Metadata 文件不存在: %s", metadata_path)
                raise FileNotFoundError(
                    "Metadata not found: {}".format(metadata_path)
                )
            from app.storage.metadata_store import MetadataStore
            from app.storage.base import MetadataChunkRepository
            t = time.time()
            metadata = MetadataStore().load(str(metadata_path))
            chunk_repo = MetadataChunkRepository(metadata)
            logger.info(
                "Metadata 降级加载完成: %.3fs, chunks=%d, path=%s",
                time.time() - t, len(metadata), metadata_path,
            )

        # ---- Retriever（按 mode 切换，延迟 import 避免强制依赖）----
        if self.mode in ("vector", "hybrid"):
            from app.embedding.model import EmbeddingModel
            from app.vector import create_vector_store
            from app.rag.retriever import Retriever

            t = time.time()
            model_name = self.config.embedding_model
            if model_name not in _embedding_cache:
                logger.info("Embedding 模型加载开始: %s", model_name)
                _embedding_cache[model_name] = EmbeddingModel(model_name)
            embedding_model = _embedding_cache[model_name]

            # Milvus/FAISS 分支
            if use_milvus:
                logger.info(
                    "Milvus 向量检索加载: host=%s, port=%s, collection=%s",
                    self.config.milvus_host, self.config.milvus_port, milvus_collection,
                )
                vector_store = create_vector_store(
                    backend="milvus",
                    dimension=embedding_model.dimension,
                    host=self.config.milvus_host,
                    port=self.config.milvus_port,
                    collection_name=milvus_collection,
                    index_type=self.config.vector_index_type,
                    ivf_nlist=self.config.ivf_nlist,
                    ivf_nprobe=self.config.ivf_nprobe,
                    hnsw_m=self.config.hnsw_m,
                    hnsw_ef_construction=self.config.hnsw_ef_construction,
                    hnsw_ef_search=self.config.hnsw_ef_search,
                )
                # Milvus.load(path) 用 path 作 collection name 连接并加载
                vector_store.load(milvus_collection)
                index_label = "milvus:" + milvus_collection
            else:
                if not faiss_index_path.exists():
                    logger.error("FAISS 索引文件不存在: %s", faiss_index_path)
                    raise FileNotFoundError(
                        "FAISS index not found: {}".format(faiss_index_path)
                    )
                vector_store = create_vector_store(
                    backend="faiss",
                    dimension=embedding_model.dimension,
                    index_type=self.config.vector_index_type,
                    ivf_nlist=self.config.ivf_nlist,
                    ivf_nprobe=self.config.ivf_nprobe,
                    hnsw_m=self.config.hnsw_m,
                    hnsw_ef_construction=self.config.hnsw_ef_construction,
                    hnsw_ef_search=self.config.hnsw_ef_search,
                )
                vector_store.load(str(faiss_index_path))
                index_label = str(faiss_index_path)

            dense = Retriever(embedding_model, vector_store, chunk_repo)
            dense.strategy = self.strategy  # 供分阶段指标打标签
            logger.info(
                "Vector 检索器初始化完成: %.3fs, dim=%d, index=%s",
                time.time() - t, embedding_model.dimension, index_label,
            )

        if self.mode in ("bm25", "hybrid"):
            t = time.time()
            sparse = None
            sparse_backend = "bm25_local"
            # 优先：ES 全文检索（storage.backends.es.enabled 时）
            if self.config.storage_es_enabled:
                try:
                    from app.search.es_fulltext_search import ESFulltextSearch

                    sparse = ESFulltextSearch(strategy=self.strategy)
                    sparse_backend = "es_fulltext"
                except Exception as es_err:
                    logger.warning(
                        "ES 全文检索初始化失败，降级到本地 BM25Search: %s",
                        es_err,
                    )
                    sparse = None
            # 兜底：本地 BM25（rank_bm25 + jieba）
            if sparse is None:
                from app.search.bm25_search import BM25Search

                sparse = BM25Search(chunk_repo)
                sparse_backend = "bm25_local"
            logger.info(
                "Sparse 检索器初始化完成: %.3fs, backend=%s",
                time.time() - t, sparse_backend,
            )

        if self.mode == "vector":
            retriever = dense
        elif self.mode == "bm25":
            retriever = sparse
            sparse.strategy = self.strategy  # 供 BM25/ES 打分阶段打标签
        else:  # hybrid
            from app.search.hybrid_search import HybridSearch
            retriever = HybridSearch(dense, sparse)
            sparse.strategy = self.strategy  # 供 BM25/ES 打分阶段打标签
        retriever.strategy = self.strategy
        logger.info("检索器已就绪: mode=%s, type=%s", self.mode, type(retriever).__name__)

        # ---- Reranker（可选）----
        reranker = None
        if self.use_rerank:
            from app.rerank.reranker import Reranker
            model_name = self.config.rerank_model
            if model_name not in _reranker_cache:
                t = time.time()
                logger.info("Reranker 加载开始: %s", model_name)
                _reranker_cache[model_name] = Reranker(model_name)
                logger.info("Reranker 加载完成: %.3fs", time.time() - t)
            reranker = _reranker_cache[model_name]

        # ---- ContextManager（从 config 注入全部参数）----
        from app.ingestion.tokenizer import create_token_counter
        from app.context.builder import ContextBuilder
        from app.context.compressor import ContextCompressor
        from app.context.manager import ContextManager
        token_counter = create_token_counter(self.config.tokenizer_backend)
        builder = ContextBuilder(
            dedup_span_overlap=self.config.dedup_span_overlap,
            dedup_jaccard=self.config.dedup_jaccard,
            merge_span_gap=self.config.merge_span_gap,
        )
        compressor = ContextCompressor(token_counter)
        context_manager = ContextManager(
            token_counter=token_counter,
            max_context_tokens=self.config.max_context_tokens,
            reserved_tokens=self.config.reserved_tokens,
            builder=builder,
            compressor=compressor,
            order_strategy=self.config.context_order_strategy,
            budget_temperature=self.config.context_budget_temperature,
        )
        logger.info(
            "ContextManager 初始化完成: max_tokens=%d, reserved=%d, order=%s",
            self.config.max_context_tokens,
            self.config.reserved_tokens,
            self.config.context_order_strategy,
        )

        # ---- Generator（从 config 切换 backend：stub 零依赖 / qwen DashScope / openai 兼容）----
        from app.generation.generator import create_generator
        t = time.time()
        logger.info(
            "Generator 初始化开始: backend=%s, model=%s",
            self.config.generation_backend, self.config.generation_model,
        )
        # 按后端传参：openai 用 generation.openai.*（base_url + 独立 api_key_env），
        # 其余（qwen）沿用顶层 generation.*
        if self.config.generation_backend == "openai":
            generator = create_generator(
                "openai",
                model=self.config.generation_openai_model,
                base_url=self.config.generation_openai_base_url,
                api_key_env=self.config.generation_openai_api_key_env,
                temperature=self.config.generation_temperature,
                max_tokens=self.config.generation_max_tokens,
                timeout=self.config.generation_timeout,
                retry_times=self.config.generation_retry_times,
                retry_backoff=self.config.generation_retry_backoff,
                max_concurrency=self.config.generation_max_concurrency,
            )
        else:
            generator = create_generator(
                self.config.generation_backend,
                model=self.config.generation_model,
                api_key_env=self.config.generation_api_key_env,
                temperature=self.config.generation_temperature,
                max_tokens=self.config.generation_max_tokens,
                timeout=self.config.generation_timeout,
                retry_times=self.config.generation_retry_times,
                retry_backoff=self.config.generation_retry_backoff,
                max_concurrency=self.config.generation_max_concurrency,
            )
        logger.info(
            "Generator 初始化完成: %.3fs, type=%s",
            time.time() - t, type(generator).__name__,
        )

        # ---- Query 改写器（复用 generator；stub 时自动跳过改写）----
        query_rewriter = None
        if generator is not None:
            from app.generation.query_rewriter import QueryRewriter
            query_rewriter = QueryRewriter(generator)

        # ---- Multi-Query 扩展器（多路召回；stub/关闭时回退单路）----
        multi_query_expander = None
        if generator is not None and self.config.retrieval_multi_query > 1:
            from app.generation.multi_query import MultiQueryExpander
            multi_query_expander = MultiQueryExpander(
                generator, num_queries=self.config.retrieval_multi_query
            )

        logger.info("pipeline 构建完成: 总耗时=%.3fs", time.time() - build_start)
        return RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            context_manager=context_manager,
            generator=generator,
            query_rewriter=query_rewriter,
            multi_query_expander=multi_query_expander,
            top_k=self.config.retrieval_top_k,
            rerank_candidate_pool=self.config.rerank_candidate_pool,
        )

    def query(self, query: str, history: Optional[List] = None,
              document_ids: Optional[set] = None) -> RAGResponse:
        """高层查询接口，返回 RAGResponse。

        history: 多轮对话历史 [{role, content}, ...]，透传给 LLM 理解指代（可选）
        document_ids: 文档级 ACL 可读文档集合（None 表示不设文档级过滤）
        """
        logger.info(
            "收到查询: query=%r, history_turns=%d, readable_docs=%s",
            query, len(history or []),
            "all" if document_ids is None else len(document_ids),
        )
        start = time.time()
        try:
            response = self._pipeline.run(query, history=history, document_ids=document_ids)
        except Exception as e:
            logger.error(
                "查询失败: %.3fs, error=%s", time.time() - start, e, exc_info=True
            )
            raise
        # 查询总耗时（P50/P95/P99）
        from app.core.metrics import metrics
        metrics.record_query_latency(self.strategy, self.mode, time.time() - start)
        logger.info(
            "查询返回: %.3fs, chunks=%d, answer_len=%d",
            time.time() - start,
            len(response.chunks),
            len(response.answer or ""),
        )
        return response

    def query_stream(self, query: str, history: Optional[List] = None,
                     document_ids: Optional[set] = None) -> Iterator[dict]:
        """流式查询接口，逐事件 yield。

        透传 RAGPipeline.run_stream 的事件流，供 API SSE 端点直接消费。
        事件结构见 RAGPipeline.run_stream 文档。

        history: 多轮对话历史 [{role, content}, ...]，透传给 LLM 理解指代（可选）
        document_ids: 文档级 ACL 可读文档集合（None 表示不设文档级过滤）
        """
        logger.info(
            "收到流式查询: query=%r, history_turns=%d, readable_docs=%s",
            query, len(history or []),
            "all" if document_ids is None else len(document_ids),
        )
        start = time.time()
        try:
            for event in self._pipeline.run_stream(
                query, history=history, document_ids=document_ids
            ):
                # 流式查询总耗时在 done/error 事件处记录（P50/P95/P99）
                if event.get("type") in ("done", "error"):
                    from app.core.metrics import metrics
                    metrics.record_query_latency(
                        self.strategy, self.mode, time.time() - start
                    )
                yield event
        except Exception as e:
            logger.error(
                "流式查询失败: %.3fs, error=%s", time.time() - start, e, exc_info=True
            )
            yield {"type": "error", "error": "流式查询失败: {}".format(e)}
        logger.info("流式查询结束: %.3fs", time.time() - start)


# ---- 模块级缓存 ----
# _service_cache: 按 (strategy, mode, use_rerank, milvus_enabled, index_version) 缓存 RAGService
# _embedding_cache / _reranker_cache: 按模型名缓存模型实例，索引变更时复用，避免重复加载权重
_service_cache: dict = {}
_embedding_cache: dict = {}
_reranker_cache: dict = {}
_cache_lock = threading.Lock()


def _index_version(strategy: str, tenant_id: str = "default") -> str:
    """计算指定 (tenant, strategy) 的索引版本号（metadata.json 的 mtime + size）。

    上传/删除/重建都会重写 metadata.json，版本随之变化，
    使 get_service 能自动感知索引更新，无需清空整个缓存。
    """
    config = Config()
    metadata_path = config.index_dir_for(strategy, tenant_id) / "metadata.json"
    try:
        stat = metadata_path.stat()
        return "{}:{}".format(stat.st_mtime_ns, stat.st_size)
    except OSError:
        return "missing"


def get_service(
    config: Optional[Config] = None,
    strategy: str = "recursive",
    mode: str = "vector",
    use_rerank: bool = True,
    tenant_id: str = "default",
) -> RAGService:
    """获取缓存的 RAGService 单例。

    缓存 key 含索引版本号（metadata.json 的 mtime+size）与 tenant_id：
      - 配置不变、索引未变：直接复用（模型 + 索引均在内存）
      - 索引变更（上传/删除/重建）：自动构建新 service，
        模型从 _embedding_cache/_reranker_cache 复用，不重复加载权重
      - 租户隔离：不同 tenant 使用独立的 service 与索引（permission-aware cache）

    注意：auth 关闭或默认租户时 tenant_id='default'，缓存行为与旧版一致。
    """
    _config = config or Config()
    version = _index_version(strategy, tenant_id)
    key = (strategy, mode, use_rerank, _config.storage_milvus_enabled, tenant_id, version)
    with _cache_lock:
        if key not in _service_cache:
            _service_cache[key] = RAGService(
                config=_config, strategy=strategy, mode=mode, use_rerank=use_rerank,
                tenant_id=tenant_id,
            )
        # 索引版本变化后，淘汰该 (strategy, tenant) 的旧版本 service（模型缓存保留复用）
        for stale in [
            k for k in _service_cache
            if k[0] == strategy and k[4] == tenant_id and k != key
        ]:
            _service_cache.pop(stale, None)
        return _service_cache[key]


def reset_service_cache():
    """清空全部缓存（service + embedding/reranker 模型），测试或切换配置时用。"""
    with _cache_lock:
        _service_cache.clear()
        _embedding_cache.clear()
        _reranker_cache.clear()
