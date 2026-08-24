"""分块仓库：chunks 表的 CRUD 操作 + BaseChunkRepository 读接口实现。

策略感知：每个 ChunkRepository 实例绑定一个 strategy（如 'fixed'/'recursive'），
所有读写操作自动按 strategy 过滤，保证不同分块策略的 chunks 互不干扰。
"""
import json
import time
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.storage.base import BaseChunkRepository
from app.storage.mysql import MySQLManager

logger = get_logger(__name__)


class ChunkRepository(BaseChunkRepository):
    """chunks 表 CRUD + 向量位置读接口（策略感知）。

    通过 chunks.id 自增列保证 list_all 返回顺序与向量入库顺序一致。
    通过 chunks.strategy 列隔离不同分块策略的 chunks。

    内部懒加载缓存：首次调用读接口时按 strategy 全量加载到内存，
    后续查询零 MySQL 开销。缓存生命周期与 ChunkRepository 实例绑定。
    """

    TABLE = "chunks"

    def __init__(self, manager: Optional[MySQLManager] = None,
                 strategy: str = "recursive"):
        self.manager = manager or MySQLManager()
        self.strategy = strategy
        # 懒加载缓存：position(int) → chunk_dict
        self._cache_list: Optional[List[Dict]] = None
        self._cache_map: Optional[Dict[int, Dict]] = None

    # ---- BaseChunkRepository 读接口（按 strategy 过滤）----

    def _ensure_loaded(self):
        """首次访问时按 strategy 全量加载 chunks 到内存缓存。"""
        if self._cache_list is not None:
            return
        t_load = time.time()
        sql = (
            "SELECT chunk_id, content, start_offset, end_offset, metadata "
            "FROM {} WHERE strategy = %s ORDER BY id ASC"
        ).format(self.TABLE)
        rows = []
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (self.strategy,))
                rows = cur.fetchall()
        for r in rows:
            if r.get("metadata"):
                r["metadata"] = json.loads(r["metadata"])
        self._cache_list = rows
        self._cache_map = {i: r for i, r in enumerate(rows)}
        logger.info(
            "ChunkRepository 全量加载: strategy=%s, %.3fs, chunks=%d",
            self.strategy, time.time() - t_load, len(rows),
        )

    def get_by_id(self, id: int) -> Optional[Dict]:
        """按向量位置 ID 查询单个 chunk。"""
        self._ensure_loaded()
        return self._cache_map.get(id)

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
        """返回当前 strategy 的所有 chunks，按向量位置顺序排列。"""
        self._ensure_loaded()
        return self._cache_list

    def count(self) -> int:
        """当前 strategy 的 chunk 总数。"""
        if self._cache_list is not None:
            return len(self._cache_list)
        sql = "SELECT COUNT(*) AS cnt FROM {} WHERE strategy = %s".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (self.strategy,))
                return cur.fetchone()["cnt"]

    # ---- 写接口 ----

    def insert(self, chunk_id: str, document_id: str, chunk_index: int,
               content: str, start_offset: int, end_offset: int,
               metadata: Optional[Dict] = None,
               strategy: Optional[str] = None) -> int:
        """插入单个 chunk（INSERT IGNORE 避免重复）。

        strategy 默认取构造函数的 self.strategy。
        """
        strat = strategy or self.strategy
        sql = (
            "INSERT IGNORE INTO {} "
            "(chunk_id, document_id, strategy, chunk_index, content, "
            "start_offset, end_offset, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        ).format(self.TABLE)
        meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(
                    sql, (chunk_id, document_id, strat, chunk_index, content,
                          start_offset, end_offset, meta_json)
                )
        return rows

    def batch_insert(self, chunks: List,
                     strategy: Optional[str] = None) -> int:
        """批量插入 chunks。

        参数 chunks 是 app.ingestion.chunk.Chunk 对象列表。
        strategy 默认取构造函数的 self.strategy。
        """
        if not chunks:
            return 0

        strat = strategy or self.strategy
        sql = (
            "INSERT IGNORE INTO {} "
            "(chunk_id, document_id, strategy, chunk_index, content, "
            "start_offset, end_offset, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        ).format(self.TABLE)

        rows_data = [
            (
                c.chunk_id, c.document_id, strat, c.chunk_index, c.content,
                c.start_offset, c.end_offset,
                json.dumps(c.metadata, ensure_ascii=False) if c.metadata else None,
            )
            for c in chunks
        ]

        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.executemany(sql, rows_data)
        logger.info(
            "批量插入 chunks: strategy=%s, %d 条, affected=%d",
            strat, len(chunks), rows,
        )
        return rows

    # ---- 查询接口 ----

    def get(self, chunk_id: str, strategy: Optional[str] = None) -> Optional[Dict]:
        """根据 chunk_id 查询单个 chunk。

        strategy 默认取 self.strategy；传 None 时返回任意 strategy 的首条匹配。
        """
        strat = strategy or self.strategy
        if strat:
            sql = "SELECT * FROM {} WHERE chunk_id = %s AND strategy = %s".format(self.TABLE)
            params = (chunk_id, strat)
        else:
            sql = "SELECT * FROM {} WHERE chunk_id = %s".format(self.TABLE)
            params = (chunk_id,)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if row and row.get("metadata"):
                    row["metadata"] = json.loads(row["metadata"])
                return row

    def get_by_document(self, document_id: str,
                        strategy: Optional[str] = None) -> List[Dict]:
        """查询某文档的所有 chunks。

        strategy 默认取 self.strategy。
        """
        strat = strategy or self.strategy
        sql = (
            "SELECT * FROM {} WHERE document_id = %s AND strategy = %s "
            "ORDER BY chunk_index"
        ).format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, strat))
                rows = cur.fetchall()
        for r in rows:
            if r.get("metadata"):
                r["metadata"] = json.loads(r["metadata"])
        return rows

    # ---- 删除接口 ----

    def delete_by_document(self, document_id: str) -> int:
        """删除某文档的所有 chunks（所有 strategy）。"""
        sql = "DELETE FROM {} WHERE document_id = %s".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(sql, (document_id,))
        logger.info("删除文档 chunks: doc_id=%s, affected=%d", document_id, rows)
        return rows

    def delete_by_strategy(self, strategy: Optional[str] = None) -> int:
        """删除指定 strategy 的所有 chunks（用于重建索引前的幂等清理）。

        strategy 默认取 self.strategy。
        """
        strat = strategy or self.strategy
        sql = "DELETE FROM {} WHERE strategy = %s".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(sql, (strat,))
        logger.info(
            "删除 strategy chunks: strategy=%s, affected=%d", strat, rows
        )
        # 失效缓存
        self._cache_list = None
        self._cache_map = None
        return rows
