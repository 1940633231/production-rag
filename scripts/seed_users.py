"""种子账号脚本：初始化 RBAC 表 + 内置角色/权限 + 创建管理员账号。

用途:
  - 首次部署后运行，或在 MySQL 认证表缺失/被清空后修复
  - 幂等：重复运行不会重复创建/报错

用法:
  .venv/Scripts/python.exe scripts/seed_users.py [--username admin] [--password admin123]

说明:
  - 账号密码默认取 config.yaml auth.seed_username / auth.seed_password
  - 种子账号固定绑定 superadmin 角色（拥有全部权限点）
  - 依赖 MySQL 可连通（storage.backends.mysql.enabled 或环境变量 MYSQL_*）
"""
import argparse
import sys
import uuid

from app.core.config import Config
from app.core.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="初始化 RBAC 并创建种子管理员")
    parser.add_argument("--username", help="种子账号用户名（默认取 config auth.seed_username）")
    parser.add_argument("--password", help="种子账号密码（默认取 config auth.seed_password）")
    args = parser.parse_args()

    config = Config()
    username = args.username or config.auth_seed_username
    password = args.password or config.auth_seed_password

    from app.auth.rbac_repository import RBACRepository
    from app.auth.rbac import SUPERADMIN_ROLE
    from app.auth.security import hash_password
    from app.storage.mysql import MySQLManager

    # 1. 建表（幂等）：文档表 + 认证/RBAC 表
    MySQLManager().init_schema()
    logger.info("MySQL 表结构初始化完成")

    # 1.5 索引版本迁移：为存量索引初始化版本记录（数据库为唯一权威源）
    try:
        from app.storage.index_version_repository import IndexVersionRepository
        IndexVersionRepository().init_from_disk()
    except Exception as e:
        logger.warning("存量索引版本初始化失败（可稍后重跑）: %s", e)

    # 2. 内置权限点 + 内置角色（幂等）
    repo = RBACRepository()
    repo.ensure_builtin_roles()
    logger.info("内置权限点与角色已就绪")

    # 3. 创建/更新种子管理员（superadmin）
    existing = repo.get_user_by_username(username)
    if existing is not None:
        logger.info("种子账号已存在，跳过创建: %s (user_id=%s)", username, existing["user_id"])
        return

    user_id = "u-" + uuid.uuid4().hex[:12]
    repo.create_user(
        user_id=user_id,
        username=username,
        password_hash=hash_password(password),
        display_name="超级管理员",
        tenant_id="default",
    )
    repo.set_user_roles(user_id, [SUPERADMIN_ROLE])
    logger.info(
        "种子管理员已创建: username=%s, user_id=%s, role=%s",
        username, user_id, SUPERADMIN_ROLE,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("种子账号初始化失败: %s", e, exc_info=True)
        sys.exit(1)
