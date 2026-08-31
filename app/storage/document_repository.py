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
