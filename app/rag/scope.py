"""Query Scope（可选能力）：业务驱动 RAG 检索范围过滤。

定位：不是 RAG 标配，而是按需引入的能力（retrieval.scope.enabled=false 默认关闭）。
适用场景：语料存在"实体竞争"——不同实体（公司/品牌/业务主体）的语义相近内容
（如多家公司的财报）互相污染检索结果时，把检索范围限定到目标实体的文档集内。
无实体竞争（单实体语料）保持默认关闭，即为纯语义检索。

流程:
  1. scope 未启用 → None（完全保持现有行为）
  2. 实体确定：显式 scope 参数 > LLM 自动提取（mode=auto）
  3. content 匹配（零 schema）：实体名 → 本地 BM25 检索 → 聚合 document_id 集
  4. 提取不到实体且 require_entity=false → 降级为不过滤（None）

与检索管道衔接:
  - 输出的 document_id 过滤集与 ACL 可读集取交集后传入现有 document_ids
    预过滤管道（FAISS IDSelector / Milvus expr / BM25 / ES terms 均已支持）
  - 交集结果参与查询缓存 key，保证不同 scope 不串缓存
"""
import threading
from typing import Dict, Optional, Set

from app.core.config import Config
from app.core.logger import get_logger

logger = get_logger(__name__)


class QueryScopeResolver:
    """业务范围过滤器：把"目标实体"解析为文档过滤集。"""

    SYSTEM_PROMPT = (
        "你是一个业务范围识别助手。从用户问题中提取检索限定的业务主体"
        "（公司/品牌/组织/业务范围）。要求：\n"
        "1. 只输出主体名称本身（如\"宝钢股份\"），不要修饰语、不要引号\n"
        "2. 如果问题不限定任何具体主体（泛问行业/通用概念），输出\"无\"\n"
        "3. 只输出一个名称，不要任何解释"
    )

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._generator = None
        self._generator_lock = threading.Lock()
        # BM25 实例缓存：(strategy, tenant_id, index_version) -> BM25Search
        self._bm25: Dict[tuple, object] = {}
        self._bm25_lock = threading.Lock()
        self._last_entity: Optional[str] = None

    # ---- 对外 ----

    def resolve(self, query: str, scope: Optional[str] = None,
                strategy: str = "recursive",
                tenant_id: str = "default") -> Optional[Set[str]]:
        """返回目标实体对应的文档过滤集；None = 不过滤（未启用或降级）。

        返回空集表示实体匹配不到任何文档（业务范围无内容）。
        """
        self._last_entity = None
        if not self.config.scope_enabled:
            return None
        entity = scope or self._extract_entity(query)
        if not entity:
            # require_entity=false：提取不到实体 → 降级为不过滤
            return None
        self._last_entity = entity
        return self._match_docs(entity, strategy, tenant_id)

    @property
    def last_entity(self) -> Optional[str]:
        """最近一次解析出的实体（供 stats 提示使用）。"""
        return self._last_entity

    # ---- 实体提取 ----

    def _extract_entity(self, query: str) -> Optional[str]:
        """LLM 从 query 提取业务主体；mode=explicit / stub / 失败 → None。"""
        if self.config.scope_mode == "explicit":
            return None
        generator = self._get_generator()
        if generator is None:
            return None
        from app.generation.generator import StubGenerator
        if isinstance(generator, StubGenerator):
            logger.info("scope 实体提取跳过: StubGenerator")
            return None
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "问题：{}\n\n实体：".format(query)},
        ]
        try:
            entity = (generator.generate(messages) or "").strip().strip("。")
        except Exception as e:
            logger.warning("scope 实体提取失败（降级不过滤）: %s", e)
            return None
        if not entity or entity.lower() in ("无", "none", "未知", "不清楚"):
            return None
        return entity

    # ---- content 匹配（实体 → 文档过滤集）----

    def _match_docs(self, entity: str, strategy: str,
                    tenant_id: str) -> Set[str]:
        """用实体名在 chunk 内容里做 BM25 匹配，聚合命中的 document_id。"""
        bm25 = self._get_bm25(strategy, tenant_id)
        if bm25 is None:
            return set()
        top_k = self.config.scope_match_top_k
        hits = bm25.search(entity, top_k=top_k, document_ids=None)
        doc_ids = {h.get("document_id") for h in hits if h.get("document_id")}
        logger.info(
            "scope content 匹配: entity=%r, top_k=%d, hits=%d, docs=%d",
            entity, top_k, len(hits), len(doc_ids),
        )
        return doc_ids

    # ---- 资源（generator / BM25 缓存）----

    def _get_generator(self):
        if self._generator is None:
            with self._generator_lock:
                if self._generator is None:
                    from app.generation.generator import create_generator

                    cfg = self.config
                    if cfg.generation_backend == "openai":
                        self._generator = create_generator(
                            "openai",
                            model=cfg.generation_openai_model,
                            base_url=cfg.generation_openai_base_url,
                            api_key_env=cfg.generation_openai_api_key_env,
                            temperature=cfg.generation_temperature,
                            max_tokens=cfg.generation_max_tokens,
                            timeout=cfg.generation_timeout,
                            retry_times=cfg.generation_retry_times,
                            retry_backoff=cfg.generation_retry_backoff,
                            max_concurrency=cfg.generation_max_concurrency,
                        )
                    else:
                        self._generator = create_generator(
                            cfg.generation_backend,
                            model=cfg.generation_model,
                            api_key_env=cfg.generation_api_key_env,
                            temperature=cfg.generation_temperature,
                            max_tokens=cfg.generation_max_tokens,
                            timeout=cfg.generation_timeout,
                            retry_times=cfg.generation_retry_times,
                            retry_backoff=cfg.generation_retry_backoff,
                            max_concurrency=cfg.generation_max_concurrency,
                        )
        return self._generator

    def _get_bm25(self, strategy: str, tenant_id: str):
        """按 (strategy, tenant, index_version) 缓存 BM25 实例。"""
        from app.rag.service import _index_version

        version = _index_version(strategy, tenant_id)
        key = (strategy, tenant_id, version)
        with self._bm25_lock:
            bm25 = self._bm25.get(key)
            if bm25 is None:
                bm25 = self._build_bm25(strategy, tenant_id)
                # 只保留当前 (strategy, tenant) 的实例，旧版本淘汰
                self._bm25 = {
                    k: v for k, v in self._bm25.items()
                    if k[0] != strategy or k[1] != tenant_id
                }
                self._bm25[key] = bm25
            return bm25

    def _build_bm25(self, strategy: str, tenant_id: str):
        """从 metadata.json 构建本地 BM25（chunk 含 document_id，零 schema）。"""
        metadata_path = self.config.index_dir_for(strategy, tenant_id) / "metadata.json"
        if not metadata_path.exists():
            logger.warning("scope 匹配: metadata 不存在，返回空过滤集: %s", metadata_path)
            return None
        from app.search.bm25_search import BM25Search
        from app.storage.base import MetadataChunkRepository
        from app.storage.metadata_store import MetadataStore

        metadata = MetadataStore().load(str(metadata_path))
        return BM25Search(MetadataChunkRepository(metadata))
