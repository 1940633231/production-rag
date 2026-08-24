"""Elasticsearch 客户端封装：索引管理 + 文档写入 + 搜索。

当前为占位实现：ES 未配置时所有操作抛 RuntimeError，
触发上层降级到 MetadataStore / MySQL。

接入真实 ES 时需:
    pip install elasticsearch
并在 config.yaml 的 storage.backends.es 段配置 hosts / index_prefix。
"""
import os

from typing import Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    from elasticsearch import Elasticsearch
    _ES_AVAILABLE = True
except ImportError:
    _ES_AVAILABLE = False
    logger.info("elasticsearch-py 未安装，ES 后端不可用")


class ESClient:
    """Elasticsearch 客户端封装。

    索引命名: {index_prefix}_{strategy}（如 production_rag_recursive）
    每个分块策略对应一个独立 ES 索引。
    """

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        basic_auth: Optional[tuple[str, str]] = None,
        index_prefix: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        if not _ES_AVAILABLE:
            raise RuntimeError(
                "elasticsearch-py 未安装，请运行: pip install elasticsearch"
            )

        # 1. hosts
        if hosts is None:
            es_hosts_env = os.getenv("ES_HOSTS")
            if es_hosts_env:
                hosts = [h.strip() for h in es_hosts_env.split(",")]
            else:
                hosts = ["http://localhost:9200"]

        # 2. basic_auth
        if basic_auth is None:
            user = os.getenv("ES_USER")
            pwd = os.getenv("ES_PASSWORD")
            if user and pwd:
                basic_auth = (user, pwd)

        # 3. index_prefix
        if index_prefix is None:
            index_prefix = os.getenv("ES_INDEX_PREFIX", "production_rag")

        # 4. timeout
        if timeout is None:
            timeout = int(os.getenv("ES_TIMEOUT", "30"))

        self.index_prefix = index_prefix

        client_kwargs = {"hosts": hosts}

        if basic_auth:
            client_kwargs["basic_auth"] = basic_auth

        client_kwargs["request_timeout"] = timeout

        self._client = Elasticsearch(**client_kwargs)

    def _index_name(self, strategy: str) -> str:
        return "{}_{}".format(self.index_prefix, strategy)

    def create_index(self, strategy: str, mappings: Optional[Dict] = None):
        """创建 ES 索引（幂等）。"""
        idx = self._index_name(strategy)
        default_mappings = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "strategy": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "vector_id": {"type": "integer"},
                    "content": {"type": "text", "analyzer": "ik_max_word"},
                    "start_offset": {"type": "integer"},
                    "end_offset": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": False},
                }
            }
        }
        if not self._client.indices.exists(index=idx):
            self._client.indices.create(
                index=idx, body=mappings or default_mappings
            )
            logger.info("ES 索引已创建: %s", idx)

    def drop_index(self, strategy: str):
        """删除 ES 索引（幂等，用于重建前清理）。"""
        idx = self._index_name(strategy)
        if self._client.indices.exists(index=idx):
            self._client.indices.delete(index=idx)
            logger.info("ES 索引已删除: %s", idx)

    def bulk_index(self, strategy: str, chunks: List[Dict]):
        """批量写入 chunks 到 ES 索引。"""
        idx = self._index_name(strategy)
        if not self._client.indices.exists(index=idx):
            self.create_index(strategy)
        body = []
        for i, chunk in enumerate(chunks):
            body.append({"index": {"_index": idx, "_id": chunk.get("chunk_id", i)}})
            body.append(chunk)
        if body:
            self._client.bulk(body=body)
            logger.info("ES 批量写入: strategy=%s, chunks=%d", strategy, len(chunks))

    def search(self, strategy: str, query: str, top_k: int = 10,
               sort_by_vector_id: bool = False) -> List[Dict]:
        """全文检索。

        - query 为 "*" 或空串时使用 match_all（用于全量列表/计数场景）
        - 否则对 content 字段做 match 查询
        - sort_by_vector_id=True 时按 vector_id 升序返回（用于 list_all 保证顺序）
        """
        idx = self._index_name(strategy)
        if query in ("*", ""):
            query_clause: Dict = {"match_all": {}}
        else:
            query_clause = {"match": {"content": query}}
        body: Dict = {
            "query": query_clause,
            "size": top_k,
        }
        if sort_by_vector_id:
            body["sort"] = [{"vector_id": {"order": "asc"}}]
        result = self._client.search(index=idx, body=body)
        hits = result.get("hits", {}).get("hits", [])
        return [
            {
                "chunk_id": h["_source"].get("chunk_id"),
                "document_id": h["_source"].get("document_id"),
                "strategy": h["_source"].get("strategy"),
                "chunk_index": h["_source"].get("chunk_index", 0),
                "vector_id": h["_source"].get("vector_id"),
                "content": h["_source"].get("content"),
                "start_offset": h["_source"].get("start_offset", 0),
                "end_offset": h["_source"].get("end_offset", 0),
                "metadata": h["_source"].get("metadata", {}),
                "score": h.get("_score", 0),
            }
            for h in hits
        ]

    def count(self, strategy: str) -> int:
        """返回索引中文档总数。"""
        idx = self._index_name(strategy)
        if not self._client.indices.exists(index=idx):
            return 0
        result = self._client.count(index=idx)
        return result.get("count", 0)

    def ping(self) -> bool:
        """检查 ES 连接是否可用。"""
        try:
            return self._client.ping()
        except Exception:
            return False
