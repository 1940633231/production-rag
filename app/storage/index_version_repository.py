"""索引版本仓储（Index Version Repository）。

索引版本以**数据库为唯一权威源**（生产多活部署下各实例共享同一 MySQL，
保证版本全局一致；不再回退本地 metadata.json 的 stat——本地文件状态在
多实例下不一致，违背"数据库唯一权威源"原则）。

表结构:
  index_versions(tenant_id, strategy, version, updated_at)
    - version 单调递增，每次索引变更（上传/删除/重建）bump +1
    - 主键 (tenant_id, strategy)，按租户 + 分块策略隔离

严格模式（strict）:
  - 读取: 无记录或 DB 异常返回 None → 调用方标记版本 "unknown" 并禁用缓存
    （宁可慢、不可错，避免用旧版本号命中过期缓存）
  - 写入: bump 失败抛异常，本次索引变更整体视为失败——保证版本登记与
    数据写入强一致，避免 DB 恢复后仍命中旧版本缓存
"""
from typing import Optional

from app.core.logger import get_logger
from app.storage.mysql import MySQLManager

logger = get_logger(__name__)

_DDL_INDEX_VERSIONS = """
CREATE TABLE IF NOT EXISTS index_versions (
    tenant_id    VARCHAR(64) NOT NULL DEFAULT 'default',
    strategy     VARCHAR(32) NOT NULL DEFAULT 'recursive',
    version      BIGINT      NOT NULL DEFAULT 0,
    updated_at   TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, strategy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# 供 MySQLManager.init_schema 统一建表
DDL_STATEMENTS = [_DDL_INDEX_VERSIONS]


class IndexVersionRepository:
    """索引版本仓储：以数据库为唯一权威源。"""

    def __init__(self, manager: Optional[MySQLManager] = None):
        self.manager = manager or MySQLManager()

    def get_version(self, tenant_id: str = "default",
                    strategy: str = "recursive") -> Optional[int]:
        """读取索引版本。无记录或 DB 异常返回 None（strict：视为 unknown）。"""
        sql = (
            "SELECT version FROM index_versions "
            "WHERE tenant_id = %s AND strategy = %s"
        )
        try:
            with self.manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (tenant_id, strategy))
                    row = cur.fetchone()
        except Exception as e:
            logger.warning("索引版本读取失败（视为 unknown）: %s", e)
            return None
        if row is None:
            return None
        return int(row.get("version", 0))

    def bump(self, tenant_id: str = "default", strategy: str = "recursive") -> int:
        """版本 +1（首次写入创建记录）。失败抛异常（strict 模式）。"""
        sql = (
            "INSERT INTO index_versions (tenant_id, strategy, version) "
            "VALUES (%s, %s, 1) "
            "ON DUPLICATE KEY UPDATE version = version + 1"
        )
        try:
            with self.manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (tenant_id, strategy))
        except Exception as e:
            logger.error("索引版本 bump 失败（strict：本次索引变更视为失败）: %s", e)
            raise

    def init_from_disk(self, root=None) -> int:
        """存量迁移：扫描 data/index 下已有的 metadata.json，为缺失的
        (tenant, strategy) 插入初始版本 0（INSERT IGNORE，幂等）。

        一次性迁移用：首次启用数据库版本权威后，存量索引立即获得版本号，
        避免"从未 bump 过"的索引一直处于 unknown（缓存永久禁用）。

        root: 索引根目录，默认 "data/index"（测试可注入临时目录）。
        """
        from pathlib import Path

        root = Path(root) if root is not None else Path("data/index")
        if not root.exists():
            return 0
        pairs = set()
        for strategy in ("fixed", "recursive"):
            # default 租户（旧布局）：data/index/{strategy}/metadata.json
            if (root / strategy / "metadata.json").exists():
                pairs.add(("default", strategy))
        for tenant_dir in root.iterdir():
            if not tenant_dir.is_dir():
                continue
            for strategy in ("fixed", "recursive"):
                # 其他租户：data/index/{tenant}/{strategy}/metadata.json
                if (tenant_dir / strategy / "metadata.json").exists():
                    pairs.add((tenant_dir.name, strategy))
        if not pairs:
            return 0
        sql = (
            "INSERT IGNORE INTO index_versions (tenant_id, strategy, version) "
            "VALUES (%s, %s, 0)"
        )
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, list(pairs))
        logger.info("存量索引版本初始化完成: %d 个 (tenant, strategy)", len(pairs))
        return len(pairs)
