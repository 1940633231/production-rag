"""文档级 ACL 仓储（Document-level Access Control）。

模型：
  - documents.owner_user_id 记录文档归属人（上传者）
  - document_acl 表按「用户/角色」对单个文档授权 read/write/delete
  - 可见性规则（租户内）：
      * superadmin：全部可见（文档级不设限）
      * 文档 owner：全部可见/可写/可删
      * 显式授权：principal_type=user 匹配 user_id；principal_type=role 匹配角色
      * 存量文档（owner 为空，无 ACL）：租户内共享（向后兼容）

事件 action 约定（供审计使用）:
  - document.grant / document.revoke
"""
from typing import Dict, List, Optional, Set

from app.core.logger import get_logger
from app.storage.mysql import MySQLManager

logger = get_logger(__name__)

_DDL_DOCUMENT_ACL = """
CREATE TABLE IF NOT EXISTS document_acl (
    document_id    VARCHAR(128) NOT NULL,
    principal_type VARCHAR(16)  NOT NULL,   -- 'user' / 'role'
    principal_id   VARCHAR(128) NOT NULL,   -- user_id 或 role_code
    permission     VARCHAR(16)  NOT NULL,   -- 'read' / 'write' / 'delete'
    created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, principal_type, principal_id, permission),
    KEY idx_acl_principal (principal_type, principal_id),
    KEY idx_acl_document (document_id),
    -- 文档删除时级联清理授权记录：防止孤儿 ACL 在 document_id 复用
    -- （同名文件重新上传）时静默挂到新文档造成越权
    CONSTRAINT document_acl_documents_FK
        FOREIGN KEY (document_id) REFERENCES documents(document_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# 供 MySQLManager.init_schema 统一建表
DDL_STATEMENTS = [_DDL_DOCUMENT_ACL]


class ACLRepository:
    """文档级授权仓储。"""

    def __init__(self, manager: Optional[MySQLManager] = None):
        self.manager = manager or MySQLManager()

    # ---------------- 授权 ----------------

    def grant(self, document_id: str, principal_type: str, principal_id: str,
              permission: str) -> None:
        """授予某用户/角色对文档的某项权限（幂等）。"""
        if principal_type not in ("user", "role"):
            raise ValueError("principal_type 必须为 user 或 role")
        if permission not in ("read", "write", "delete"):
            raise ValueError("permission 必须为 read/write/delete")
        sql = (
            "INSERT IGNORE INTO document_acl "
            "(document_id, principal_type, principal_id, permission) "
            "VALUES (%s, %s, %s, %s)"
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql, (document_id, principal_type, principal_id, permission)
                )

    def revoke(self, document_id: str, principal_type: Optional[str] = None,
               principal_id: Optional[str] = None,
               permission: Optional[str] = None) -> int:
        """撤销授权（可按条件部分撤销）。返回影响行数。"""
        clauses, params = ["document_id = %s"], [document_id]
        if principal_type:
            clauses.append("principal_type = %s")
            params.append(principal_type)
        if principal_id:
            clauses.append("principal_id = %s")
            params.append(principal_id)
        if permission:
            clauses.append("permission = %s")
            params.append(permission)
        sql = "DELETE FROM document_acl WHERE {}".format(" AND ".join(clauses))
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return cur.execute(sql, tuple(params))

    def delete_by_document(self, document_id: str) -> int:
        """删除某文档的全部授权记录（文档删除时级联清理）。

        防止孤儿 ACL 残留：document_id 复用（同名文件重新上传）时，
        旧授权若不清除会静默挂到新文档上造成越权。
        """
        return self.revoke(document_id)

    def list_grants(self, document_id: str) -> List[Dict]:
        """列出某文档的全部授权。"""
        sql = (
            "SELECT principal_type, principal_id, permission, created_at "
            "FROM document_acl WHERE document_id = %s "
            "ORDER BY permission, principal_type, principal_id"
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                rows = list(cur.fetchall())
        for r in rows:
            r["created_at"] = str(r.get("created_at", ""))
        return rows

    # ---------------- 权限判定 ----------------

    def has_permission(self, user, document_id: str, permission: str,
                       tenant_id: str = "default") -> bool:
        """判断用户对某文档是否拥有指定权限。

        user 为 AuthUser（含 user_id/roles/is_superadmin）；None 视为放行（鉴权关闭）。
        """
        if user is None or user.is_superadmin:
            return True

        from app.storage.document_repository import DocumentRepository
        doc = DocumentRepository(self.manager).get(document_id, tenant_id=tenant_id)
        if doc is None:
            return False
        owner = doc.get("owner_user_id") or ""
        # 归属人拥有全部权限
        if owner == user.user_id:
            return True
        # 存量文档（无归属、无 ACL）：仅可读
        if not owner:
            return permission == "read"

        roles = list(user.roles or [])
        if not roles:
            # 无角色时只看直接用户授权
            placeholders = ""
            role_clause = ""
            role_params = []
        else:
            placeholders = ",".join(["%s"] * len(roles))
            role_clause = " OR (principal_type = 'role' AND principal_id IN ({}))".format(
                placeholders
            )
            role_params = roles

        sql = (
            "SELECT 1 FROM document_acl WHERE document_id = %s "
            "AND permission = %s AND ("
            "(principal_type = 'user' AND principal_id = %s)"
            + role_clause +
            ") LIMIT 1"
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, permission, user.user_id, *role_params))
                return cur.fetchone() is not None

    def get_readable_document_ids(self, user, tenant_id: str = "default") -> Optional[Set[str]]:
        """返回用户在该租户内可读的 document_id 集合。

        返回 None 表示不设文档级过滤（superadmin / 鉴权关闭）。
        """
        if user is None or user.is_superadmin:
            return None

        roles = list(user.roles or [])
        if roles:
            placeholders = ",".join(["%s"] * len(roles))
            role_clause = (
                "OR (principal_type = 'role' AND principal_id IN ({}))".format(placeholders)
            )
            role_params = roles
        else:
            role_clause = ""
            role_params = []

        sql = (
            "SELECT DISTINCT d.document_id FROM documents d "
            "WHERE d.tenant_id = %s AND ("
            "  d.owner_user_id = %s"
            "  OR d.owner_user_id = ''"
            "  OR EXISTS ("
            "    SELECT 1 FROM document_acl a "
            "    WHERE a.document_id = d.document_id AND a.permission = 'read' AND ("
            "      (a.principal_type = 'user' AND a.principal_id = %s)"
            "      {}"
            "    )"
            "  )"
            ")"
        ).format(role_clause)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (tenant_id, user.user_id, user.user_id, *role_params))
                return {r["document_id"] for r in cur.fetchall()}
