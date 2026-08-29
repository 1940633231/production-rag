from pathlib import Path
import yaml


class Config:

    def __init__(self, config_path="configs/config.yaml"):

        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:

            self.data = yaml.safe_load(f)

    @property
    def embedding_model(self):
        return self.data["embedding"]["model_name"]

    @property
    def index_path(self):
        return self.data["vector"]["index_path"]

    @property
    def metadata_path(self):
        return self.data["vector"]["metadata_path"]

    # ---- 向量索引类型与参数 ----

    @property
    def vector_config(self):
        """vector 段整体。"""
        return self.data.get("vector", {})

    @property
    def vector_index_type(self):
        """索引类型: flat / ivf / hnsw，默认 flat。"""
        return self.vector_config.get("index_type", "flat")

    @property
    def ivf_nlist(self):
        return self.vector_config.get("ivf_nlist", 128)

    @property
    def ivf_nprobe(self):
        return self.vector_config.get("ivf_nprobe", 16)

    @property
    def hnsw_m(self):
        return self.vector_config.get("hnsw_m", 16)

    @property
    def hnsw_ef_construction(self):
        return self.vector_config.get("hnsw_ef_construction", 200)

    @property
    def hnsw_ef_search(self):
        return self.vector_config.get("hnsw_ef_search", 64)

    @property
    def chunk_size(self):
        return self.data["chunk"]["chunk_size"]

    @property
    def chunk_overlap(self):
        return self.data["chunk"]["overlap"]

    @property
    def retrieval_top_k(self):
        return self.data["retrieval"]["top_k"]

    @property
    def retrieval_multi_query(self):
        """多路召回总路数（含原始 query），1 或缺失表示关闭。"""
        return self.data.get("retrieval", {}).get("multi_query", 1)

    @property
    def rerank_model(self):
        return self.data["rerank"]["model_name"]

    @property
    def rerank_candidate_pool(self):
        return self.data["rerank"]["candidate_pool"]

    # ---- context 段（ContextManager 配置）----

    @property
    def context_config(self):
        """context 段整体，缺失时返回空 dict 保证默认值可用。"""
        return self.data.get("context", {})

    @property
    def max_context_tokens(self):
        return self.context_config.get("max_context_tokens", 4096)

    @property
    def reserved_tokens(self):
        return self.context_config.get("reserved_tokens", 1024)

    @property
    def tokenizer_backend(self):
        return self.context_config.get("tokenizer_backend", "char")

    @property
    def context_order_strategy(self):
        return self.context_config.get("order_strategy", "score")

    @property
    def context_budget_temperature(self):
        return self.context_config.get("budget_temperature", 1.0)

    @property
    def dedup_span_overlap(self):
        return self.context_config.get("dedup_span_overlap", 0.5)

    @property
    def dedup_jaccard(self):
        return self.context_config.get("dedup_jaccard", 0.85)

    @property
    def merge_span_gap(self):
        return self.context_config.get("merge_span_gap", 5)

    # ---- generation 段（LLM 生成配置）----

    @property
    def generation_config(self):
        """generation 段整体，缺失时返回空 dict 保证默认值可用。"""
        return self.data.get("generation", {})

    @property
    def generation_backend(self):
        return self.generation_config.get("backend", "stub")

    @property
    def generation_model(self):
        return self.generation_config.get("model_name", "qwen-turbo")

    @property
    def generation_api_key_env(self):
        return self.generation_config.get("api_key_env", "DASHSCOPE_API_KEY")

    @property
    def generation_temperature(self):
        return self.generation_config.get("temperature", 0.3)

    @property
    def generation_max_tokens(self):
        return self.generation_config.get("max_tokens", 1024)

    @property
    def generation_timeout(self):
        return self.generation_config.get("timeout", 60)

    @property
    def generation_retry_times(self):
        return self.generation_config.get("retry_times", 2)

    @property
    def generation_retry_backoff(self):
        return self.generation_config.get("retry_backoff", 1.0)

    @property
    def generation_max_concurrency(self):
        """最大并发 LLM 调用数（0=不限制）。"""
        return self.generation_config.get("max_concurrency", 4)

    # ---- storage 段（MySQL / ES / Milvus 持久化配置）----

    @property
    def storage_config(self):
        """storage 段整体，缺失时返回空 dict 保证默认值可用。"""
        return self.data.get("storage", {})

    @property
    def storage_enabled(self):
        """向后兼容总开关：等同 backends.mysql.enabled。

        旧配置仅有 storage.enabled=true（无 backends 段）时仍生效。
        """
        backends = self.storage_config.get("backends", {})
        mysql_cfg = backends.get("mysql", {})
        if "enabled" in mysql_cfg:
            return mysql_cfg["enabled"]
        return self.storage_config.get("enabled", False)

    @property
    def storage_backends(self):
        """storage.backends 段整体，缺失时返回空 dict。"""
        return self.storage_config.get("backends", {})

    @property
    def storage_mysql_enabled(self):
        """MySQL 后端开关：优先读 backends.mysql.enabled，回退到 storage.enabled。"""
        backends = self.storage_backends
        mysql_cfg = backends.get("mysql", {})
        if "enabled" in mysql_cfg:
            return mysql_cfg["enabled"]
        return self.storage_config.get("enabled", False)

    @property
    def storage_es_enabled(self):
        """Elasticsearch 后端开关。"""
        return self.storage_backends.get("es", {}).get("enabled", False)

    @property
    def storage_milvus_enabled(self):
        """Milvus 后端开关。

        环境变量 MILVUS_ENABLED（1/true/yes/on）可覆盖 config.yaml，
        便于 CI/测试临时切到 FAISS 本地索引。
        """
        import os
        env_val = os.getenv("MILVUS_ENABLED")
        if env_val is not None:
            return env_val.strip().lower() in ("1", "true", "yes", "on")
        return self.storage_backends.get("milvus", {}).get("enabled", False)

    # ---- milvus 连接参数 ----

    @property
    def _milvus_cfg(self):
        """storage.backends.milvus 配置段（缺失时返回空 dict）。"""
        return self.storage_backends.get("milvus", {})

    @property
    def milvus_host(self):
        """Milvus 服务地址：环境变量 MILVUS_HOST > config.yaml > 默认 127.0.0.1"""
        import os
        env_val = os.getenv("MILVUS_HOST")
        if env_val:
            return env_val
        return self._milvus_cfg.get("host", "127.0.0.1")

    @property
    def milvus_port(self):
        """Milvus 服务端口：环境变量 MILVUS_PORT > config.yaml > 默认 19530"""
        import os
        env_val = os.getenv("MILVUS_PORT")
        if env_val:
            return int(env_val)
        return int(self._milvus_cfg.get("port", 19530))

    @property
    def milvus_collection_prefix(self):
        """Milvus collection 前缀：环境变量 MILVUS_COLLECTION_PREFIX > config.yaml > 默认 production_rag

        实际 collection 名称为 {prefix}_{strategy}，例如 production_rag_recursive。
        """
        import os
        env_val = os.getenv("MILVUS_COLLECTION_PREFIX")
        if env_val:
            return env_val
        return self._milvus_cfg.get("collection_prefix", "production_rag")

    def milvus_collection_name(self, strategy: str) -> str:
        """按 strategy 生成实际的 Milvus collection 名称。

        约定: {collection_prefix}_{strategy}
        例: milvus_collection_name("recursive") → "production_rag_recursive"
        """
        return "{}_{}".format(self.milvus_collection_prefix, strategy)

    # ---- 通用 ----

    @property
    def storage_pool_size(self):
        return self.storage_config.get("pool_size", 5)

    @property
    def mysql_host(self):
        import os
        return os.getenv("MYSQL_HOST", "127.0.0.1")

    @property
    def mysql_port(self):
        import os
        return int(os.getenv("MYSQL_PORT", "3306"))

    @property
    def mysql_user(self):
        import os
        return os.getenv("MYSQL_USER", "root")

    @property
    def mysql_password(self):
        import os
        return os.getenv("MYSQL_PASSWORD", "root")

    @property
    def mysql_database(self):
        import os
        return os.getenv("MYSQL_DATABASE", "production_rag")
