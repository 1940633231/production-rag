"""文档仓库：documents 表的 CRUD 操作（租户感知）。

tenant_id 语义:
  - insert 时写入租户归属
  - 查询/删除时传入 tenant_id 做严格隔离（None 表示不按租户过滤，供管理/测试用）
"""
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.storage.mysql import MySQLManager

logger = get_logger(__name__)


class DocumentRepository:
    """documents 表 CRUD。"""

    TABLE = "documents"

    def __init__(self, manager: Optional[MySQLManager] = None):
        self.manager = manager or MySQLManager()

    def insert(self, document_id: str, file_name: str,
               content_length: int, source: Optional[str] = None,
               tenant_id: str = "default",
               owner_user_id: str = "") -> int:
        """插入文档记录（INSERT IGNORE 避免重复）。

        owner_user_id: 文档归属人（用于文档级 ACL）。空表示存量/共享文档。
        """
        sql = (
            "INSERT IGNORE INTO {} "
            "(document_id, tenant_id, owner_user_id, file_name, content_length, source) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        ).format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(
                    sql,
                    (document_id, tenant_id, owner_user_id, file_name,
                     content_length, source),
                )
        logger.info(
            "插入文档: id=%s, tenant=%s, owner=%s, file=%s, affected=%d",
            document_id, tenant_id, owner_user_id or "-", file_name, rows,
        )
        return rows

    @staticmethod
    def _tenant_clause(tenant_id: Optional[str]):
        """返回 (SQL 片段, 参数列表)。tenant_id=None 表示不按租户过滤。"""
        if tenant_id is None:
            return "", []
        return " AND tenant_id = %s", [tenant_id]

    def get(self, document_id: str,
            tenant_id: Optional[str] = None) -> Optional[Dict]:
        """根据 document_id 查询单个文档。

        tenant_id 提供时强制校验归属，避免跨租户读取。
        """
        clause, params = self._tenant_clause(tenant_id)
        sql = "SELECT * FROM {} WHERE document_id = %s{}".format(self.TABLE, clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, *params))
                return cur.fetchone()

    def list_all(self, limit: int = 100, offset: int = 0,
                 tenant_id: Optional[str] = None) -> List[Dict]:
        """列出文档（分页）。tenant_id 提供时仅返回该租户文档。"""
        clause, params = self._tenant_clause(tenant_id)
        sql = (
            "SELECT * FROM {} WHERE 1=1{} ORDER BY created_at DESC "
            "LIMIT %s OFFSET %s"
        ).format(self.TABLE, clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (*params, limit, offset))
                return cur.fetchall()

    def get_by_file_name(self, file_name: str,
                         tenant_id: Optional[str] = None) -> Optional[Dict]:
        """根据文件名查询文档。tenant_id 提供时仅匹配该租户。"""
        clause, params = self._tenant_clause(tenant_id)
        sql = "SELECT * FROM {} WHERE file_name = %s{}".format(self.TABLE, clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (file_name, *params))
                return cur.fetchone()

    def delete(self, document_id: str,
               tenant_id: Optional[str] = None) -> int:
        """删除文档（外键 ON DELETE CASCADE 会自动删除关联 chunks）。

        tenant_id 提供时仅删除该租户下的文档，防止跨租户误删。
        """
        clause, params = self._tenant_clause(tenant_id)
        sql = "DELETE FROM {} WHERE document_id = %s{}".format(self.TABLE, clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(sql, (document_id, *params))
        logger.info(
            "删除文档: id=%s, tenant=%s, affected=%d",
            document_id, tenant_id if tenant_id is not None else "*", rows,
        )
        return rows

    def count(self, tenant_id: Optional[str] = None) -> int:
        """文档总数。tenant_id 提供时仅统计该租户。"""
        clause, params = self._tenant_clause(tenant_id)
        sql = "SELECT COUNT(*) AS cnt FROM {} WHERE 1=1{}".format(self.TABLE, clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()["cnt"]

    # ---- 版本台账（document_versions 表）：版本按 (document, strategy) 维度 ----

    def get_active_versions(self, document_ids, strategy: str,
                            tenant_id: Optional[str] = None) -> Dict[str, int]:
        """批量读取 (document, strategy) 的活跃版本 = 该对下现存版本的最大值。

        供检索活跃过滤用；无版本台账行的文档不返回（视为不可过滤，交由上层放行）。
        """
        if not document_ids:
            return {}
        clause, params = self._tenant_clause(tenant_id)
        placeholders = ",".join(["%s"] * len(document_ids))
        sql = (
            "SELECT document_id, MAX(version) AS mv FROM document_versions "
            "WHERE strategy = %s AND document_id IN ({}){} "
            "GROUP BY document_id"
        ).format(placeholders, clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (strategy, *document_ids, *params))
                return {r["document_id"]: int(r["mv"]) for r in cur.fetchall()}

    def next_version(self, document_id: str, strategy: str,
                     tenant_id: Optional[str] = None) -> int:
        """该 (document, strategy) 的下一个版本号（现存最大 + 1，无则 1）。"""
        active = self.get_active_versions([document_id], strategy, tenant_id)
        ver = active.get(document_id, 0)
        return int(ver) + 1 if ver else 1

    def list_versions(self, document_id: str, strategy: str,
                      tenant_id: Optional[str] = None) -> List[Dict]:
        """列出某 (document, strategy) 的历史版本（新→旧）。"""
        clause, params = self._tenant_clause(tenant_id)
        sql = (
            "SELECT version, chunk_count, created_at FROM document_versions "
            "WHERE document_id = %s AND strategy = %s{} ORDER BY version DESC".format(clause)
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, strategy, *params))
                rows = cur.fetchall()
        for r in rows:
            r["created_at"] = str(r.get("created_at", ""))
        return rows

    def insert_version(self, document_id: str, strategy: str, version: int,
                       tenant_id: str = "default", chunk_count: int = 0) -> None:
        """记录 (document, strategy) 的一个版本到台账（INSERT IGNORE 幂等）。"""
        sql = (
            "INSERT IGNORE INTO document_versions "
            "(document_id, strategy, version, tenant_id, chunk_count) "
            "VALUES (%s, %s, %s, %s, %s)"
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql, (document_id, strategy, int(version), tenant_id, int(chunk_count))
                )

    def delete_version(self, document_id: str, strategy: str, version: int,
                       tenant_id: Optional[str] = None) -> int:
        """从台账删除某 (document, strategy, version) 行。"""
        clause, params = self._tenant_clause(tenant_id)
        sql = (
            "DELETE FROM document_versions "
            "WHERE document_id = %s AND strategy = %s AND version = %s{}".format(clause)
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return cur.execute(sql, (document_id, strategy, int(version), *params))

    def has_version(self, document_id: str, strategy: str, version: int,
                    tenant_id: Optional[str] = None) -> bool:
        """台账中是否存在该 (document, strategy, version)。"""
        clause, params = self._tenant_clause(tenant_id)
        sql = (
            "SELECT 1 FROM document_versions "
            "WHERE document_id = %s AND strategy = %s AND version = %s{} LIMIT 1"
        ).format(clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, strategy, int(version), *params))
                return cur.fetchone() is not None
