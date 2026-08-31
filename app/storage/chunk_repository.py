"""分块仓库：chunks 表的 CRUD 操作 + BaseChunkRepository 读接口实现（租户感知）。

策略感知：每个 ChunkRepository 实例绑定一个 strategy（如 'fixed'/'recursive'），
所有读写操作自动按 strategy 过滤，保证不同分块策略的 chunks 互不干扰。
租户感知：可选绑定 tenant_id；绑定后读写均按租户隔离，跨租户不可见。

读路径的向量位置语义：
  - 租户 FAISS 索引的位置 0..N-1 对应「该租户 metadata.json 的 enumerate 顺序」
  - MySQL 侧按 tenant_id + strategy 过滤后按 chunks.id 升序（全局自增，
    在租户内部与写入时间序一致），enumerate 即向量位置 —— 与单租户时代一致
"""
import json
import time
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.storage.base import BaseChunkRepository
from app.storage.mysql import MySQLManager

logger = get_logger(__name__)


class ChunkRepository(BaseChunkRepository):
    """chunks 表 CRUD + 向量位置读接口（策略 + 租户感知）。

    通过 chunks.id 自增列保证 list_all 返回顺序与向量入库顺序一致。
    通过 chunks.strategy 列隔离不同分块策略的 chunks。
    通过 chunks.tenant_id 列隔离不同租户的 chunks（tenant_id=None 时不隔离）。

    内部懒加载缓存：首次调用读接口时按 strategy(+tenant) 全量加载到内存，
    后续查询零 MySQL 开销。缓存生命周期与 ChunkRepository 实例绑定。
    """

    TABLE = "chunks"

    def __init__(self, manager: Optional[MySQLManager] = None,
                 strategy: str = "recursive",
                 tenant_id: Optional[str] = None):
        self.manager = manager or MySQLManager()
        self.strategy = strategy
        self.tenant_id = tenant_id
        # 懒加载缓存：position(int) → chunk_dict
        self._cache_list: Optional[List[Dict]] = None
        self._cache_map: Optional[Dict[int, Dict]] = None

    @staticmethod
    def _tenant_clause(tenant_id: Optional[str]):
        """返回 (SQL 片段, 参数)。tenant_id=None 表示不按租户过滤。"""
        if tenant_id is None:
            return "", []
        return " AND tenant_id = %s", [tenant_id]

    # ---- BaseChunkRepository 读接口（按 strategy + tenant 过滤，按 vector_id 索引）----

    def _ensure_loaded(self):
        """首次访问时按 strategy(+tenant) 全量加载 chunks 到内存缓存。

        _cache_map 以 vector_id（稳定 ID）为 key，而非 enumerate 位置；
        get_by_id(int) 即按 vector_id 查询，与 FAISS/Milvus 显式主键一致。
        """
        if self._cache_list is not None:
            return
        t_load = time.time()
        tenant_clause, tenant_params = self._tenant_clause(self.tenant_id)
        sql = (
            "SELECT chunk_id, document_id, vector_id, content, start_offset, end_offset, metadata "
            "FROM {} WHERE strategy = %s{} ORDER BY id ASC"
        ).format(self.TABLE, tenant_clause)
        rows = []
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (self.strategy, *tenant_params))
                rows = cur.fetchall()
        for r in rows:
            if r.get("metadata"):
                r["metadata"] = json.loads(r["metadata"])
        self._cache_list = rows
        self._cache_map = {int(r.get("vector_id", 0)): r for r in rows}
        logger.info(
            "ChunkRepository 全量加载: strategy=%s, tenant=%s, %.3fs, chunks=%d",
            self.strategy, self.tenant_id if self.tenant_id is not None else "*",
            time.time() - t_load, len(rows),
        )

    def get_by_id(self, id: int) -> Optional[Dict]:
        """按向量 ID（稳定 vector_id）查询单个 chunk。"""
        self._ensure_loaded()
        return self._cache_map.get(int(id))

    def batch_get_by_ids(self, ids: List[int]) -> List[Dict]:
        """批量按向量位置 ID 查询 chunks。"""
        self._ensure_loaded()
        result = []
        for i in ids:
            doc = self._cache_map.get(i)
            if doc is not None:
                result.append(doc)
        return result

    def list_all(self) -> List[Dict]:
        """返回当前 strategy(+tenant) 的所有 chunks，按向量位置顺序排列。"""
        self._ensure_loaded()
        return self._cache_list

    def count(self) -> int:
        """当前 strategy(+tenant) 的 chunk 总数。"""
        if self._cache_list is not None:
            return len(self._cache_list)
        tenant_clause, tenant_params = self._tenant_clause(self.tenant_id)
        sql = "SELECT COUNT(*) AS cnt FROM {} WHERE strategy = %s{}".format(
            self.TABLE, tenant_clause
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (self.strategy, *tenant_params))
                return cur.fetchone()["cnt"]

    # ---- 写接口 ----

    def insert(self, chunk_id: str, document_id: str, chunk_index: int,
               content: str, start_offset: int, end_offset: int,
               metadata: Optional[Dict] = None,
               strategy: Optional[str] = None,
               tenant_id: Optional[str] = None,
               vector_id: int = 0) -> int:
        """插入单个 chunk（INSERT IGNORE 避免重复）。

        strategy 默认取构造函数的 self.strategy；
        tenant_id 默认取 self.tenant_id（None 时写入 'default'）；
        vector_id 为稳定向量 ID（FAISS/Milvus 显式主键），默认 0。
        """
        strat = strategy or self.strategy
        tnt = tenant_id or self.tenant_id or "default"
        sql = (
            "INSERT IGNORE INTO {} "
            "(chunk_id, document_id, tenant_id, strategy, vector_id, chunk_index, content, "
            "start_offset, end_offset, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        ).format(self.TABLE)
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(
                    sql, (chunk_id, document_id, tnt, strat, int(vector_id),
                          chunk_index, content, start_offset, end_offset, meta_json)
                )
        return rows

    def batch_insert(self, chunks: List,
                     strategy: Optional[str] = None,
                     tenant_id: Optional[str] = None) -> int:
        """批量插入 chunks。

        参数 chunks 是 app.ingestion.chunk.Chunk 对象列表。
        strategy 默认取构造函数的 self.strategy；
        tenant_id 默认取 self.tenant_id（None 时写入 'default'）；
        vector_id 取各 chunk 的 vector_id（稳定 ID）。
        """
        if not chunks:
            return 0

        strat = strategy or self.strategy
        tnt = tenant_id or self.tenant_id or "default"
        sql = (
            "INSERT IGNORE INTO {} "
            "(chunk_id, document_id, tenant_id, strategy, vector_id, chunk_index, content, "
            "start_offset, end_offset, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        ).format(self.TABLE)

        rows_data = [
            (
                c.chunk_id, c.document_id, tnt, strat, int(getattr(c, "vector_id", 0) or 0),
                c.chunk_index, c.content, c.start_offset, c.end_offset,
                json.dumps(c.metadata, ensure_ascii=False) if c.metadata else None,
            )
            for c in chunks
        ]

        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.executemany(sql, rows_data)
        logger.info(
            "批量插入 chunks: strategy=%s, tenant=%s, %d 条, affected=%d",
            strat, tnt, len(chunks), rows,
        )
        return rows

    # ---- 查询接口 ----

    def get(self, chunk_id: str, strategy: Optional[str] = None,
            tenant_id: Optional[str] = None) -> Optional[Dict]:
        """根据 chunk_id 查询单个 chunk。

        strategy 默认取 self.strategy；传 None 时返回任意 strategy 的首条匹配。
        tenant_id 默认取 self.tenant_id（None 时不按租户过滤）。
        """
        strat = strategy or self.strategy
        tnt = tenant_id if tenant_id is not None else self.tenant_id
        tenant_clause, tenant_params = self._tenant_clause(tnt)
        if strat:
            sql = "SELECT * FROM {} WHERE chunk_id = %s AND strategy = %s{}".format(
                self.TABLE, tenant_clause
            )
            params = (chunk_id, strat, *tenant_params)
        else:
            sql = "SELECT * FROM {} WHERE chunk_id = %s{}".format(
                self.TABLE, tenant_clause
            )
            params = (chunk_id, *tenant_params)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if row and row.get("metadata"):
                    row["metadata"] = json.loads(row["metadata"])
                return row

    def get_by_document(self, document_id: str,
                        strategy: Optional[str] = None,
                        tenant_id: Optional[str] = None) -> List[Dict]:
        """查询某文档的所有 chunks。

        strategy 默认取 self.strategy；tenant_id 默认取 self.tenant_id。
        """
        strat = strategy or self.strategy
        tnt = tenant_id if tenant_id is not None else self.tenant_id
        tenant_clause, tenant_params = self._tenant_clause(tnt)
        sql = (
            "SELECT * FROM {} WHERE document_id = %s AND strategy = %s{} "
            "ORDER BY chunk_index"
        ).format(self.TABLE, tenant_clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, strat, *tenant_params))
                rows = cur.fetchall()
        for r in rows:
            if r.get("metadata"):
                r["metadata"] = json.loads(r["metadata"])
        return rows

    def get_vector_ids_by_document(self, document_id: str,
                                   tenant_id: Optional[str] = None) -> List[int]:
        """返回某文档的全部稳定向量 ID（跨 strategy 去重）。

        删除文档时据此从向量后端 / metadata.json 移除，无需重建索引。
        """
        tnt = tenant_id if tenant_id is not None else self.tenant_id
        tenant_clause, tenant_params = self._tenant_clause(tnt)
        sql = (
            "SELECT DISTINCT vector_id FROM {} "
            "WHERE document_id = %s AND vector_id > 0{}"
        ).format(self.TABLE, tenant_clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, *tenant_params))
                return [int(r["vector_id"]) for r in cur.fetchall()]

    # ---- 删除接口 ----

    def delete_by_document(self, document_id: str,
                           tenant_id: Optional[str] = None) -> int:
        """删除某文档的所有 chunks（所有 strategy）。

        tenant_id 默认取 self.tenant_id（None 时不按租户过滤）。
        """
        tnt = tenant_id if tenant_id is not None else self.tenant_id
        tenant_clause, tenant_params = self._tenant_clause(tnt)
        sql = "DELETE FROM {} WHERE document_id = %s{}".format(
            self.TABLE, tenant_clause
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(sql, (document_id, *tenant_params))
        logger.info(
            "删除文档 chunks: doc_id=%s, tenant=%s, affected=%d",
            document_id, tnt if tnt is not None else "*", rows,
        )
        # 失效缓存
        self._cache_list = None
        self._cache_map = None
        return rows

    def delete_by_strategy(self, strategy: Optional[str] = None,
                           tenant_id: Optional[str] = None) -> int:
        """删除指定 strategy（+tenant）的所有 chunks（用于重建索引前的幂等清理）。

        strategy 默认取 self.strategy；tenant_id 默认取 self.tenant_id。
        """
        strat = strategy or self.strategy
        tnt = tenant_id if tenant_id is not None else self.tenant_id
        tenant_clause, tenant_params = self._tenant_clause(tnt)
        sql = "DELETE FROM {} WHERE strategy = %s{}".format(
            self.TABLE, tenant_clause
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(sql, (strat, *tenant_params))
        logger.info(
            "删除 strategy chunks: strategy=%s, tenant=%s, affected=%d",
            strat, tnt if tnt is not None else "*", rows,
        )
        # 失效缓存
        self._cache_list = None
        self._cache_map = None
        return rows
