"""RBAC 仓储：users / roles / permissions / user_roles / role_permissions 表 CRUD。

DDL 说明:
  - users.user_id 由调用方生成（uuid）；username 唯一
  - user_roles / role_permissions 外键 ON DELETE CASCADE
  - superadmin 角色不绑定具体权限，权限解析时特殊处理（拥有全部）
"""
from typing import Dict, List, Optional, Set

from app.auth.rbac import ALL_PERMISSIONS, BUILTIN_ROLES, SUPERADMIN_ROLE
from app.core.logger import get_logger
from app.storage.mysql import MySQLManager

logger = get_logger(__name__)

_DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id       VARCHAR(64)   NOT NULL,
    username      VARCHAR(128)  NOT NULL,
    password_hash VARCHAR(256)  NOT NULL,
    display_name  VARCHAR(128)  NOT NULL DEFAULT '',
    tenant_id     VARCHAR(64)   NOT NULL DEFAULT 'default',
    is_active     TINYINT(1)    NOT NULL DEFAULT 1,
    created_at    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id),
    UNIQUE KEY uk_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_DDL_ROLES = """
CREATE TABLE IF NOT EXISTS roles (
    role_code VARCHAR(64)  NOT NULL,
    name      VARCHAR(128) NOT NULL,
    builtin   TINYINT(1)   NOT NULL DEFAULT 0,
    PRIMARY KEY (role_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_DDL_PERMISSIONS = """
CREATE TABLE IF NOT EXISTS permissions (
    permission_code VARCHAR(128) NOT NULL,
    name            VARCHAR(128) NOT NULL,
    PRIMARY KEY (permission_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_DDL_USER_ROLES = """
CREATE TABLE IF NOT EXISTS user_roles (
    user_id   VARCHAR(64) NOT NULL,
    role_code VARCHAR(64) NOT NULL,
    PRIMARY KEY (user_id, role_code),
    CONSTRAINT fk_ur_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    CONSTRAINT fk_ur_role FOREIGN KEY (role_code) REFERENCES roles(role_code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_DDL_ROLE_PERMISSIONS = """
CREATE TABLE IF NOT EXISTS role_permissions (
    role_code       VARCHAR(64)  NOT NULL,
    permission_code VARCHAR(128) NOT NULL,
    PRIMARY KEY (role_code, permission_code),
    CONSTRAINT fk_rp_role FOREIGN KEY (role_code) REFERENCES roles(role_code) ON DELETE CASCADE,
    CONSTRAINT fk_rp_perm FOREIGN KEY (permission_code) REFERENCES permissions(permission_code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# 供 MySQLManager.init_schema 统一建表
DDL_STATEMENTS = [
    _DDL_USERS,
    _DDL_ROLES,
    _DDL_PERMISSIONS,
    _DDL_USER_ROLES,
    _DDL_ROLE_PERMISSIONS,
]


class RBACRepository:
    """用户 / 角色 / 权限仓储。"""

    def __init__(self, manager: Optional[MySQLManager] = None):
        self.manager = manager or MySQLManager()

    # ---------------- 权限 ----------------

    def ensure_permissions(self, permissions: Dict[str, str]) -> int:
        """批量写入权限点（INSERT IGNORE，幂等），返回影响行数。"""
        sql = "INSERT IGNORE INTO permissions (permission_code, name) VALUES (%s, %s)"
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                rows = cur.executemany(sql, list(permissions.items()))
        return rows or 0

    def list_permissions(self) -> List[Dict]:
        """列出全部权限点。"""
        sql = "SELECT permission_code, name FROM permissions ORDER BY permission_code"
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchall()

    # ---------------- 角色 ----------------

    def create_role(
        self,
        role_code: str,
        name: str,
        permission_codes: Optional[List[str]] = None,
        builtin: bool = False,
    ) -> None:
        """创建角色并绑定权限（幂等）。"""
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT IGNORE INTO roles (role_code, name, builtin) VALUES (%s, %s, %s)",
                    (role_code, name, 1 if builtin else 0),
                )
                if permission_codes:
                    cur.executemany(
                        "INSERT IGNORE INTO role_permissions (role_code, permission_code) VALUES (%s, %s)",
                        [(role_code, p) for p in permission_codes],
                    )

    def ensure_builtin_roles(self) -> None:
        """幂等写入全部权限点 + 内置角色（superadmin 不绑定具体权限）。"""
        self.ensure_permissions(ALL_PERMISSIONS)
        for code, (name, perms) in BUILTIN_ROLES.items():
            perms_to_bind = [] if perms == "*" else perms
            self.create_role(code, name, perms_to_bind, builtin=True)

    def get_role_by_code(self, role_code: str) -> Optional[Dict]:
        """按 code 查角色。"""
        sql = "SELECT role_code, name, builtin FROM roles WHERE role_code = %s"
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (role_code,))
                return cur.fetchone()

    def list_roles(self) -> List[Dict]:
        """列出全部角色（含权限列表）。"""
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT role_code, name, builtin FROM roles ORDER BY role_code")
                roles = cur.fetchall()
                for role in roles:
                    role["permissions"] = self._get_role_permission_codes(
                        cur, role["role_code"]
                    )
                return roles

    @staticmethod
    def _get_role_permission_codes(cur, role_code: str) -> List[str]:
        cur.execute(
            "SELECT permission_code FROM role_permissions WHERE role_code = %s ORDER BY permission_code",
            (role_code,),
        )
        return [r["permission_code"] for r in cur.fetchall()]

    def get_role_permissions(self, role_code: str) -> List[str]:
        """按 code 查角色的权限列表。"""
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return self._get_role_permission_codes(cur, role_code)

    def set_role_permissions(self, role_code: str, permission_codes: List[str]) -> None:
        """整体替换角色权限。"""
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM role_permissions WHERE role_code = %s", (role_code,)
                )
                if permission_codes:
                    cur.executemany(
                        "INSERT IGNORE INTO role_permissions (role_code, permission_code) VALUES (%s, %s)",
                        [(role_code, p) for p in permission_codes],
                    )

    def delete_role(self, role_code: str) -> int:
        """删除角色（级联清理 role_permissions / user_roles / 文档级 ACL 角色授权）。"""
        # 角色删除时清理其全部文档授权（principal_id 多态无法用外键约束，代码级清理）
        try:
            from app.acl.repository import ACLRepository
            ACLRepository(self.manager).delete_by_role(role_code)
        except Exception as e:
            logger.warning("清理角色文档 ACL 失败: %s", e)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return cur.execute("DELETE FROM roles WHERE role_code = %s", (role_code,))

    # ---------------- 用户 ----------------

    def create_user(
        self,
        user_id: str,
        username: str,
        password_hash: str,
        display_name: str = "",
        tenant_id: str = "default",
        is_active: bool = True,
    ) -> int:
        """创建用户（用户名冲突时抛异常）。"""
        sql = (
            "INSERT INTO users "
            "(user_id, username, password_hash, display_name, tenant_id, is_active) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return cur.execute(
                    sql,
                    (user_id, username, password_hash, display_name, tenant_id, int(is_active)),
                )

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """按用户名查用户。"""
        sql = "SELECT * FROM users WHERE username = %s"
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (username,))
                return cur.fetchone()

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """按 user_id 查用户。"""
        sql = "SELECT * FROM users WHERE user_id = %s"
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchone()

    def list_users(self) -> List[Dict]:
        """列出全部用户（含角色）。"""
        sql = "SELECT user_id, username, display_name, tenant_id, is_active, created_at FROM users ORDER BY created_at DESC"
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                users = cur.fetchall()
                for user in users:
                    user["roles"] = self._get_user_role_codes(cur, user["user_id"])
                return users

    @staticmethod
    def _get_user_role_codes(cur, user_id: str) -> List[str]:
        cur.execute(
            "SELECT role_code FROM user_roles WHERE user_id = %s ORDER BY role_code",
            (user_id,),
        )
        return [r["role_code"] for r in cur.fetchall()]

    def get_user_roles(self, user_id: str) -> List[str]:
        """按 user_id 查角色编码列表。"""
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return self._get_user_role_codes(cur, user_id)

    def set_user_roles(self, user_id: str, role_codes: List[str]) -> None:
        """整体替换用户的角色绑定。"""
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_roles WHERE user_id = %s", (user_id,))
                if role_codes:
                    cur.executemany(
                        "INSERT IGNORE INTO user_roles (user_id, role_code) VALUES (%s, %s)",
                        [(user_id, r) for r in role_codes],
                    )

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """用户权限集合。superadmin 拥有全部权限点。"""
        roles = self.get_user_roles(user_id)
        if SUPERADMIN_ROLE in roles:
            return set(ALL_PERMISSIONS.keys())
        if not roles:
            return set()
        placeholders = ",".join(["%s"] * len(roles))
        sql = (
            "SELECT DISTINCT rp.permission_code FROM role_permissions rp "
            "JOIN user_roles ur ON ur.role_code = rp.role_code "
            "WHERE ur.user_id = %s AND ur.role_code IN ({})"
        ).format(placeholders)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, *roles))
                return {r["permission_code"] for r in cur.fetchall()}

    def update_user(
        self, user_id: str, password_hash: Optional[str] = None,
        display_name: Optional[str] = None, is_active: Optional[bool] = None,
    ) -> int:
        """更新用户字段（仅更新非 None 字段）。"""
        updates, params = [], []
        if password_hash is not None:
            updates.append("password_hash = %s")
            params.append(password_hash)
        if display_name is not None:
            updates.append("display_name = %s")
            params.append(display_name)
        if is_active is not None:
            updates.append("is_active = %s")
            params.append(int(is_active))
        if not updates:
            return 0
        sql = "UPDATE users SET {} WHERE user_id = %s".format(", ".join(updates))
        params.append(user_id)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return cur.execute(sql, tuple(params))

    def delete_user(self, user_id: str) -> int:
        """删除用户（级联清理 user_roles 与文档级 ACL 授权）。"""
        # 用户删除时清理其全部文档授权（principal_id 多态无法用外键约束，代码级清理）
        try:
            from app.acl.repository import ACLRepository
            ACLRepository(self.manager).delete_by_user(user_id)
        except Exception as e:
            logger.warning("清理用户文档 ACL 失败: %s", e)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                return cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
