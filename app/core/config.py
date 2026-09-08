from pathlib import Path
import yaml


class Config:

    def __init__(self, config_path="configs/config.yaml"):

        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:

            self.data = yaml.safe_load(f)

    # ---- 运行环境与生产安全护栏 ----

    # 已知默认值：生产模式不允许原样使用（fail-fast）
    DEFAULT_JWT_SECRET = "dev-secret-change-me"
    DEFAULT_ADMIN_PASSWORD = "admin123"

    @property
    def run_mode(self):
        """运行模式：环境变量 APP_ENV（dev/production，默认 dev/dev_local）。"""
        import os
        return os.getenv("APP_ENV", "dev").strip().lower() or "dev"

    @property
    def is_production(self):
        """是否生产模式（APP_ENV=production / prod）。"""
        return self.run_mode in ("production", "prod")

    def validate_production_security(self):
        """生产模式安全校验：发现已知默认密钥/默认管理员密码则抛错，拒绝以弱配置启动。

        目的：不依赖 deployment operator 自觉修改。任何一次启动若仍带着
        dev-secret-change-me / admin123 跑生产，直接启动失败，而不是默默上线。
        """
        if not self.is_production:
            return
        errors = []
        if self.auth_config.get("enabled", True):
            secret = self.auth_jwt_secret
            if secret == self.DEFAULT_JWT_SECRET:
                errors.append("JWT 密钥仍为默认占位 'dev-secret-change-me'")
            elif len(secret.encode("utf-8")) < 32:
                errors.append("JWT 密钥过短（<32 字节），易被暴力破解")
            if self.auth_seed_password == self.DEFAULT_ADMIN_PASSWORD:
                errors.append("管理员密码仍为默认 'admin123'")
        if errors:
            raise RuntimeError(
                "生产模式（APP_ENV=production）安全校验失败："
                + "; ".join(errors)
                + "。请设置强密钥：export JWT_SECRET=<强随机值>；"
                  "并以 seed_users.py --password <强密码> 初始化管理员后更新 auth.seed_password。"
            )

    def warn_default_credentials(self):
        """非生产模式下的善意提醒（不阻断启动）。"""
        if self.is_production:
            return
        import logging
        if self.auth_config.get("enabled", True):
            secret = self.auth_jwt_secret
            if secret == self.DEFAULT_JWT_SECRET or len(secret.encode("utf-8")) < 32:
                logging.getLogger(__name__).warning(
                    "JWT 密钥仍为默认/过短（%s...）。生产环境请务必通过环境变量 %s 覆盖强密钥。",
                    secret[:8], self.auth_config.get("jwt_secret_env", "JWT_SECRET"),
                )
            if self.auth_seed_password == self.DEFAULT_ADMIN_PASSWORD:
                logging.getLogger(__name__).warning(
                    "管理员密码仍为默认 'admin123'。请用 scripts/seed_users.py --password <强密码> "
                    "重设并在部署配置中覆盖 auth.seed_password。",
                )

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
    def retrieval_multi_query_auto(self):
        """多路召回按需启用：true 时由 LLM 判断该 query 是否需要多路
        （需要才扩展，省 LLM 调用/减噪音路；默认 false 保持固定路数）。"""
        return bool(self.data.get("retrieval", {}).get("multi_query_auto", False))

    # ---- Query Scope（可选能力：业务驱动 RAG 过滤）----

    @property
    def retrieval_scope_config(self):
        """retrieval.scope 段整体，缺失时返回空 dict。"""
        return self.data.get("retrieval", {}).get("scope", {})

    @property
    def scope_enabled(self):
        """Query Scope 可选能力开关（默认 false：检索不做业务范围过滤）。"""
        return bool(self.retrieval_scope_config.get("enabled", False))

    @property
    def scope_mode(self):
        """scope 实体来源：auto（LLM 从 query 提取）/ explicit（仅显式传入）。"""
        return str(self.retrieval_scope_config.get("mode", "auto")).lower()

    @property
    def scope_require_entity(self):
        """提取不到实体时：true 拒绝 / false 降级为不过滤（默认 false）。"""
        return bool(self.retrieval_scope_config.get("require_entity", False))

    @property
    def scope_match_top_k(self):
        """content 匹配时生成过滤集的文档召回数。"""
        return int(self.retrieval_scope_config.get("match_top_k", 10))

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

    @property
    def generation_openai_config(self):
        """generation.openai 段（OpenAI 兼容后端），缺失时返回空 dict。"""
        return self.generation_config.get("openai", {})

    @property
    def generation_openai_base_url(self):
        """OpenAI 兼容端点 base_url（OpenAI / DeepSeek / Kimi / GLM / 百炼等）。"""
        return self.generation_openai_config.get(
            "base_url", "https://api.openai.com/v1"
        )

    @property
    def generation_openai_api_key_env(self):
        """OpenAI 兼容后端 API key 的环境变量名。"""
        return self.generation_openai_config.get("api_key_env", "OPENAI_API_KEY")

    @property
    def generation_openai_model(self):
        """OpenAI 兼容后端模型名；未单独配置时复用 generation.model_name。"""
        return self.generation_openai_config.get(
            "model_name"
        ) or self.generation_model

    @property
    def ingestion_config(self):
        """ingestion 段整体，缺失时返回空 dict。"""
        return self.data.get("ingestion", {})

    @property
    def document_versioning_enabled(self):
        """文档版本化开关（默认 false：同名上传保持现状，行为零变化）。

        开启后同名文档每次上传 = 新版本（各端隔离写入 + 活跃指针原子切换 +
        旧版本 GC），检索只命中活跃版本。
        """
        vc = self.ingestion_config.get("document_versioning", {})
        return bool(vc.get("enabled", False))

    @property
    def document_versioning_retention(self):
        """版本保留策略：latest（只留最新，感知为"更新"）——P1 仅支持 latest。"""
        vc = self.ingestion_config.get("document_versioning", {})
        return str(vc.get("retention", "latest")).lower()

    @property
    def write_validation_enabled(self):
        """发布前校验开关（默认 true）：确认本次 chunks 已写入各启用后端。"""
        return bool(self.ingestion_config.get("write_validation", {}).get(
            "enabled", True
        ))

    @property
    def write_validation_strict(self):
        """校验失败策略（默认 true）：strict=抛异常中止发布（版本不切换）；
        false=仅告警继续发布。"""
        return bool(self.ingestion_config.get("write_validation", {}).get(
            "strict", True
        ))

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

    def milvus_collection_for(self, strategy: str, tenant_id: str = "default") -> str:
        """按 (tenant, strategy) 生成租户隔离的 Milvus collection 名称。

        - default 租户: 保持旧命名 {prefix}_{strategy}（向后兼容存量数据）
        - 其他租户: {prefix}_{tenant_id}_{strategy}，保证租户间向量完全隔离
        """
        if tenant_id == "default":
            return self.milvus_collection_name(strategy)
        return "{}_{}_{}".format(self.milvus_collection_prefix, tenant_id, strategy)

    # ---- 租户隔离路径助手 ----

    def raw_dir_for(self, tenant_id: str = "default") -> Path:
        """某租户的原始文档目录。

        - default 租户: data/raw（旧布局，向后兼容）
        - 其他租户: data/raw/{tenant_id}
        """
        if tenant_id == "default":
            return Path("data/raw")
        return Path("data/raw") / tenant_id

    def index_dir_for(self, strategy: str, tenant_id: str = "default") -> Path:
        """某租户指定策略的索引目录。

        - default 租户: data/index/{strategy}（旧布局，向后兼容）
        - 其他租户: data/index/{tenant_id}/{strategy}
        """
        if tenant_id == "default":
            return Path("data/index") / strategy
        return Path("data/index") / tenant_id / strategy

    def es_index_name_for(self, strategy: str, tenant_id: str = "default") -> str:
        """某租户指定策略的 ES 索引名。

        - default 租户: {prefix}_{strategy}（旧命名，向后兼容）
        - 其他租户: {prefix}_{tenant_id}_{strategy}
        """
        prefix = self.storage_backends.get("es", {}).get(
            "index_prefix", "production_rag"
        )
        if tenant_id == "default":
            return "{}_{}".format(prefix, strategy)
        return "{}_{}_{}".format(prefix, tenant_id, strategy)

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

    # ---- auth 段（认证 / RBAC）----

    @property
    def auth_config(self):
        """auth 段整体，缺失时返回空 dict 保证默认值可用。"""
        return self.data.get("auth", {})

    @property
    def auth_enabled(self):
        """鉴权总开关（默认 true）。"""
        return self.auth_config.get("enabled", True)

    @property
    def auth_jwt_secret(self):
        """JWT 密钥：环境变量 auth.jwt_secret_env > config.yaml auth.jwt_secret > 默认值。"""
        import os
        env_name = self.auth_config.get("jwt_secret_env", "JWT_SECRET")
        env_val = os.getenv(env_name)
        if env_val:
            return env_val
        return self.auth_config.get("jwt_secret", "dev-secret-change-me")

    @property
    def auth_token_expire_hours(self):
        return int(self.auth_config.get("token_expire_hours", 24))

    @property
    def auth_token_version_check(self):
        """是否启用 token 吊销版本校验（权限变更后旧 token 即时失效，fail-closed）。"""
        return bool(self.auth_config.get("token_version_check", True))

    @property
    def auth_algorithm(self):
        return self.auth_config.get("algorithm", "HS256")

    @property
    def auth_seed_username(self):
        """种子账号用户名（scripts/seed_users.py 使用）。"""
        return self.auth_config.get("seed_username", "admin")

    @property
    def auth_seed_password(self):
        """种子账号密码（scripts/seed_users.py 使用）。"""
        return self.auth_config.get("seed_password", "admin123")

    # ---- cache 段（权限感知查询缓存）----

    @property
    def cache_config(self):
        """cache 段整体，缺失时返回空 dict 保证默认值可用。"""
        return self.data.get("cache", {})

    @property
    def cache_enabled(self):
        """权限感知查询缓存总开关（默认 true）。"""
        return self.cache_config.get("enabled", True)

    @property
    def cache_ttl_seconds(self):
        """缓存有效期（秒）。0 表示立即过期。"""
        return int(self.cache_config.get("ttl_seconds", 300))

    @property
    def cache_max_entries(self):
        """最大缓存条目数（超出按 LRU 淘汰）。"""
        return int(self.cache_config.get("max_entries", 2000))

    @property
    def cache_backend(self):
        """查询缓存后端：redis / memory（默认 memory，Redis 不可用时自动降级）。"""
        return str(self.cache_config.get("backend", "memory")).lower()

    @property
    def cache_redis_config(self):
        """cache.redis 段整体，缺失时返回空 dict。"""
        return self.cache_config.get("redis", {})

    @property
    def cache_redis_host(self):
        """Redis 地址：优先环境变量 REDIS_HOST，回退 config.yaml cache.redis.host。"""
        import os
        return os.getenv("REDIS_HOST") or self.cache_redis_config.get("host", "127.0.0.1")

    @property
    def cache_redis_port(self):
        """Redis 端口：优先环境变量 REDIS_PORT，回退 config.yaml cache.redis.port。"""
        import os
        return int(os.getenv("REDIS_PORT") or self.cache_redis_config.get("port", 6379))

    @property
    def cache_redis_db(self):
        """Redis 逻辑库：优先环境变量 REDIS_DB，回退 config.yaml cache.redis.db。"""
        import os
        return int(os.getenv("REDIS_DB") or self.cache_redis_config.get("db", 0))

    @property
    def cache_redis_prefix(self):
        """Redis key 前缀：优先环境变量 REDIS_PREFIX，回退 config.yaml cache.redis.prefix。"""
        import os
        return os.getenv("REDIS_PREFIX") or self.cache_redis_config.get("prefix", "rag:qcache:")

    @property
    def cache_redis_password(self):
        """Redis 密码：从 cache.redis.password_env 指定的环境变量读取，未设置则无密码。"""
        import os
        env_name = self.cache_redis_config.get("password_env")
        if not env_name:
            return None
        return os.getenv(env_name) or None

    # ---- audit 段（审计日志）----

    @property
    def audit_config(self):
        """audit 段整体，缺失时返回空 dict 保证默认值可用。"""
        return self.data.get("audit", {})

    @property
    def audit_enabled(self):
        """审计总开关（默认 true）。"""
        return self.audit_config.get("enabled", True)

    @property
    def audit_record_types(self):
        """审计事件类型白名单。

        "*" 表示全部；否则返回 set，如 {"login", "user", "document"}。
        """
        raw = self.audit_config.get("record", "*")
        if isinstance(raw, str) and raw.strip() == "*":
            return "*"
        if isinstance(raw, str):
            return set(t.strip() for t in raw.split(",") if t.strip())
        return set(raw) if isinstance(raw, (list, set)) else "*"
