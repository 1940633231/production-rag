"""文档仓库：documents 表的 CRUD 操作。"""
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
               content_length: int, source: Optional[str] = None) -> int:
        """插入文档记录（INSERT IGNORE 避免重复）。"""
        sql = (
            "INSERT IGNORE INTO {} "
            "(document_id, file_name, content_length, source) "
            "VALUES (%s, %s, %s, %s)"
        ).format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(sql, (document_id, file_name, content_length, source))
        logger.info("插入文档: id=%s, file=%s, affected=%d", document_id, file_name, rows)
        return rows

    def get(self, document_id: str) -> Optional[Dict]:
        """根据 document_id 查询单个文档。"""
        sql = "SELECT * FROM {} WHERE document_id = %s".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                return cur.fetchone()

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """列出所有文档（分页）。"""
        sql = "SELECT * FROM {} ORDER BY created_at DESC LIMIT %s OFFSET %s".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit, offset))
                return cur.fetchall()

    def get_by_file_name(self, file_name: str) -> Optional[Dict]:
        """根据文件名查询文档。"""
        sql = "SELECT * FROM {} WHERE file_name = %s".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (file_name,))
                return cur.fetchone()

    def delete(self, document_id: str) -> int:
        """删除文档（外键 ON DELETE CASCADE 会自动删除关联 chunks）。"""
        sql = "DELETE FROM {} WHERE document_id = %s".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.execute(sql, (document_id,))
        logger.info("删除文档: id=%s, affected=%d", document_id, rows)
        return rows

    def count(self) -> int:
        """文档总数。"""
        sql = "SELECT COUNT(*) AS cnt FROM {}".format(self.TABLE)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()["cnt"]
