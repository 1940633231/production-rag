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
    租户隔离: tenant_id != 'default' 时索引为 {index_prefix}_{tenant_id}_{strategy}
    每个分块策略对应一个独立 ES 索引。
    """

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        basic_auth: Optional[tuple[str, str]] = None,
        index_prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        tenant_id: str = "default"
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
        self.tenant_id = tenant_id

        client_kwargs = {"hosts": hosts}

        if basic_auth:
            client_kwargs["basic_auth"] = basic_auth

        client_kwargs["request_timeout"] = timeout

        self._client = Elasticsearch(**client_kwargs)

    def _index_name(self, strategy: str) -> str:
        if self.tenant_id == "default":
            return "{}_{}".format(self.index_prefix, strategy)
        return "{}_{}_{}".format(self.index_prefix, self.tenant_id, strategy)

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
                    "vector_id": {"type": "long"},
                    "version": {"type": "integer"},
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
               sort_by_vector_id: bool = False,
               document_ids: Optional[List] = None) -> List[Dict]:
        """全文检索。

        - query 为 "*" 或空串时使用 match_all（用于全量列表/计数场景）
        - 否则对 content 字段做 match 查询
        - document_ids 提供时用 terms 在 document_id 上先过滤后检索
          （permission-aware：不可读文档不参与打分，不占用 top_k）
        - sort_by_vector_id=True 时按 vector_id 升序返回（用于 list_all 保证顺序）
        """
        idx = self._index_name(strategy)

        # 无可读文档：直接返回空（避免查询后置过滤浪费 top_k）
        if document_ids is not None and not document_ids:
            return []

        if query in ("*", ""):
            query_clause: Dict = {"match_all": {}}
        else:
            query_clause = {"match": {"content": query}}

        if document_ids is not None:
            # 先过滤后检索：bool must=词条匹配 + filter=terms 文档过滤
            body: Dict = {
                "query": {
                    "bool": {
                        "must": [query_clause],
                        "filter": [
                            {"terms": {"document_id": list(document_ids)}}
                        ],
                    }
                },
                "size": top_k,
            }
        else:
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

    def refresh_index(self, strategy: str):
        """立即刷新索引，使刚批量写入/删除的数据可被搜索（供写入后/校验前调用）。

        ES 默认约 1s 才把 index 结果刷到可搜索；写入后立即搜索可能读不到，
        故在写入路径显式 refresh。刷新失败仅降级（下次自动刷新仍会追上）。
        """
        try:
            self._client.indices.refresh(index=self._index_name(strategy))
        except Exception as e:
            logger.warning("ES 索引 refresh 失败（可稍后自动追上）: strategy=%s, %s", strategy, e)

    # ---- 按需读取（避免全量加载到内存）----

    @staticmethod
    def _to_chunk_dict(hit) -> Dict:
        src = hit.get("_source", {})
        return {
            "chunk_id": src.get("chunk_id"),
            "document_id": src.get("document_id"),
            "strategy": src.get("strategy"),
            "chunk_index": src.get("chunk_index", 0),
            "vector_id": src.get("vector_id"),
            "version": src.get("version"),
            "content": src.get("content"),
            "start_offset": src.get("start_offset", 0),
            "end_offset": src.get("end_offset", 0),
            "metadata": src.get("metadata", {}),
            "score": hit.get("_score", 0),
        }

    def get_by_vector_id(self, strategy: str, vector_id: int) -> Optional[Dict]:
        """按稳定 vector_id 单查 chunk（检索命中时按需取元数据）。"""
        idx = self._index_name(strategy)
        body = {"query": {"term": {"vector_id": int(vector_id)}}, "size": 1}
        resp = self._client.search(index=idx, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return None
        return self._to_chunk_dict(hits[0])

    def search_all(self, strategy: str, query: str = "*",
                   document_ids: Optional[List] = None,
                   page_size: int = 500) -> List[Dict]:
        """分页拉取全部命中（search_after 游标，避免单次 size 上限截断）。

        供 BM25 全量语料 / list_all 使用；返回按 vector_id 升序（与向量位置对齐）。
        """
        idx = self._index_name(strategy)
        if query in ("*", ""):
            query_clause: Dict = {"match_all": {}}
        else:
            query_clause = {"match": {"content": query}}
        if document_ids is not None:
            body: Dict = {
                "query": {
                    "bool": {
                        "must": [query_clause],
                        "filter": [{"terms": {"document_id": list(document_ids)}}],
                    }
                },
            }
        else:
            body = {"query": query_clause}
        body["sort"] = [{"vector_id": {"order": "asc", "unmapped_type": "long"}}]
        body["size"] = page_size

        results: List[Dict] = []
        search_after = None
        while True:
            if search_after is not None:
                body["search_after"] = search_after
            resp = self._client.search(index=idx, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            results.extend(self._to_chunk_dict(h) for h in hits)
            if len(hits) < page_size:
                break
            search_after = hits[-1].get("sort")
        return results

    def vector_ids_by_documents(self, strategy: str,
                                document_ids: List) -> Dict[str, set]:
        """按可读文档集合返回 {document_id: {vector_id}}（先过滤后检索）。

        只取 document_id/vector_id 两字段分页拉取，避免全量文档进内存。
        """
        if not document_ids:
            return {}
        idx = self._index_name(strategy)
        body = {
            "query": {
                "bool": {"filter": [{"terms": {"document_id": list(document_ids)}}]}
            },
            "_source": ["document_id", "vector_id"],
            "sort": [{"vector_id": {"order": "asc", "unmapped_type": "long"}}],
            "size": 1000,
        }
        m: Dict[str, set] = {}
        search_after = None
        while True:
            if search_after is not None:
                body["search_after"] = search_after
            resp = self._client.search(index=idx, body=body)
            hits = resp.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                src = h.get("_source", {})
                doc = src.get("document_id")
                vid = src.get("vector_id")
                if doc and vid is not None:
                    m.setdefault(doc, set()).add(int(vid))
            if len(hits) < 1000:
                break
            search_after = hits[-1].get("sort")
        return m

    def ping(self) -> bool:
        """检查 ES 连接是否可用。"""
        try:
            return self._client.ping()
        except Exception:
            return False
