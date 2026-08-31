"""MySQL 连接管理器：连接池 + DDL 建表。

采用 pymysql 作为底层驱动（纯 Python，跨平台，无编译依赖）。
连接池复用 connection，避免每次请求新建连接。

表结构:
  documents - 文档元信息
    - document_id (PK)
    - file_name, content_length, source, created_at, updated_at
  chunks - 文档分块
    - chunk_id (PK)
    - document_id (FK → documents)
    - chunk_index, content, start_offset, end_offset, metadata (JSON)

使用示例:
    from app.storage.mysql import MySQLManager

    mgr = MySQLManager()
    mgr.init_schema()  # 建表

    with mgr.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            print(cur.fetchone())
"""
import os
from contextlib import contextmanager
from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    import pymysql
    from pymysql.cursors import DictCursor
    from dbutils.pooled_db import PooledDB
    _MYSQL_AVAILABLE = True
except ImportError:
    _MYSQL_AVAILABLE = False
    logger.warning("pymysql 或 dbutils 未安装，MySQL 存储层不可用")


_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    document_id    VARCHAR(128)  NOT NULL,
    tenant_id      VARCHAR(64)   NOT NULL DEFAULT 'default',
    owner_user_id  VARCHAR(64)   NOT NULL DEFAULT '',
    file_name      VARCHAR(512)  NOT NULL,
    content_length INT           NOT NULL DEFAULT 0,
    source         VARCHAR(512)  DEFAULT NULL,
    created_at     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id),
    KEY idx_documents_tenant (tenant_id),
    KEY idx_documents_owner (owner_user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGINT        NOT NULL AUTO_INCREMENT,
    chunk_id      VARCHAR(128)  NOT NULL,
    document_id   VARCHAR(128)  NOT NULL,
    tenant_id     VARCHAR(64)   NOT NULL DEFAULT 'default',
    strategy      VARCHAR(32)   NOT NULL DEFAULT 'recursive',
    chunk_index   INT           NOT NULL,
    content       MEDIUMTEXT    NOT NULL,
    start_offset  INT           NOT NULL DEFAULT 0,
    end_offset    INT           NOT NULL DEFAULT 0,
    metadata      JSON          DEFAULT NULL,
    created_at    TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, strategy),
    KEY idx_id (id),
    KEY idx_document_id (document_id),
    KEY idx_strategy (strategy),
    KEY idx_chunks_tenant (tenant_id, strategy),
    CONSTRAINT fk_chunk_document FOREIGN KEY (document_id)
        REFERENCES documents(document_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# 旧表迁移：为已存在的 chunks 表补加 id 自增列（保证向量位置顺序可回溯）
_MIGRATE_ADD_ID = """
ALTER TABLE chunks
    ADD COLUMN id BIGINT NOT NULL AUTO_INCREMENT,
    ADD KEY idx_id (id);
"""

# 旧表迁移：为已存在的 chunks 表补加 strategy 列 + 复合主键
_MIGRATE_ADD_STRATEGY = [
    "ALTER TABLE chunks ADD COLUMN strategy VARCHAR(32) NOT NULL DEFAULT 'recursive'",
    "ALTER TABLE chunks ADD KEY idx_strategy (strategy)",
    "ALTER TABLE chunks DROP PRIMARY KEY",
    "ALTER TABLE chunks ADD PRIMARY KEY (chunk_id, strategy)",
]


class MySQLManager:
    """MySQL 连接管理器：封装连接池和建表逻辑。

    连接参数优先级:
      1. 构造函数显式传参
      2. 环境变量 (MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE)
      3. 默认值
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        pool_size: int = 5,
    ):
        if not _MYSQL_AVAILABLE:
            raise RuntimeError(
                "pymysql/dbutils 未安装，请运行: pip install pymysql dbutils"
            )

        self.host = host or os.getenv("MYSQL_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("MYSQL_PORT", "3306"))
        self.user = user or os.getenv("MYSQL_USER", "root")
        self.password = password or os.getenv("MYSQL_PASSWORD", "root")
        self.database = database or os.getenv("MYSQL_DATABASE", "production_rag")
        self.pool_size = pool_size

        self._pool: Optional["PooledDB"] = None

    def _ensure_pool(self):
        """延迟创建连接池。"""
        if self._pool is None:
            logger.info(
                "创建 MySQL 连接池: %s@%s:%s/%s (pool_size=%d)",
                self.user, self.host, self.port, self.database, self.pool_size,
            )
            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=self.pool_size,
                mincached=1,
                maxcached=self.pool_size,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=DictCursor,
                autocommit=False,
                connect_timeout=5,
                read_timeout=10,
                write_timeout=10,
            )
            logger.info("MySQL 连接池已创建")

    @contextmanager
    def get_connection(self):
        """获取连接（上下文管理器，自动提交/回滚/归还）。

        使用方式:
            with mgr.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        """
        self._ensure_pool()
        conn = self._pool.connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self):
        """初始化表结构（幂等）。

        对已存在的旧表，检测并迁移缺失的列（id / strategy）。
        同时创建认证/RBAC 相关表（users/roles/permissions/...）。
        """
        logger.info("初始化 MySQL 表结构")
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL_DOCUMENTS)
                cur.execute(_DDL_CHUNKS)
                # 迁移：旧表无 id 列时补加
                self._migrate_chunks_id(cur)
                # 迁移：旧表无 strategy 列时补加 + 切换复合主键
                self._migrate_chunks_strategy(cur)
                # 迁移：旧表无 tenant_id 列时补加（租户隔离）
                self._migrate_tenant_id(cur)
                # 迁移：旧表无 owner_user_id 列时补加（文档级 ACL）
                self._migrate_owner_user_id(cur)
                # 认证/RBAC 表（软失败：不影响文档表）
                try:
                    from app.auth.rbac_repository import DDL_STATEMENTS as AUTH_DDL
                    for ddl in AUTH_DDL:
                        cur.execute(ddl)
                    logger.info("认证/RBAC 表初始化完成: users, roles, permissions, user_roles, role_permissions")
                except Exception as e:
                    logger.warning("认证/RBAC 表初始化失败（可稍后运行 scripts/seed_users.py）: %s", e)
                # 审计日志表（软失败：不影响文档表）
                try:
                    from app.audit.logger import DDL_STATEMENTS as AUDIT_DDL
                    for ddl in AUDIT_DDL:
                        cur.execute(ddl)
                    logger.info("审计日志表初始化完成: audit_logs")
                except Exception as e:
                    logger.warning("审计日志表初始化失败（可稍后运行 scripts/seed_users.py）: %s", e)
                # 文档级 ACL 表（软失败：不影响文档表）
                try:
                    from app.acl.repository import DDL_STATEMENTS as ACL_DDL
                    for ddl in ACL_DDL:
                        cur.execute(ddl)
                    logger.info("文档级 ACL 表初始化完成: document_acl")
                except Exception as e:
                    logger.warning("文档级 ACL 表初始化失败（可稍后运行 scripts/seed_users.py）: %s", e)
        logger.info("MySQL 表结构初始化完成: documents, chunks")

    def _migrate_chunks_id(self, cur):
        """检测 chunks 表是否有 id 列，缺失则执行 ALTER TABLE 补加。"""
        try:
            cur.execute("SELECT id FROM chunks LIMIT 1")
        except Exception:
            logger.info("chunks 表缺少 id 列，执行迁移: ALTER TABLE ADD id")
            cur.execute(_MIGRATE_ADD_ID)
            logger.info("chunks 表 id 列迁移完成")

    def _migrate_chunks_strategy(self, cur):
        """检测 chunks 表是否有 strategy 列，缺失则补加列 + 切换复合主键。"""
        try:
            cur.execute("SELECT strategy FROM chunks LIMIT 1")
        except Exception:
            logger.info("chunks 表缺少 strategy 列，执行迁移")
            for stmt in _MIGRATE_ADD_STRATEGY:
                cur.execute(stmt)
            logger.info("chunks 表 strategy 列 + 复合主键迁移完成")

    def _migrate_tenant_id(self, cur):
        """检测 documents / chunks 表是否有 tenant_id 列，缺失则补加 + 建索引。

        老库升级为租户隔离时执行；既有数据统一归属 default 租户。
        """
        for table, index_name in (
            ("documents", "idx_documents_tenant"),
            ("chunks", "idx_chunks_tenant"),
        ):
            try:
                cur.execute("SELECT tenant_id FROM {} LIMIT 1".format(table))
            except Exception:
                logger.info("%s 表缺少 tenant_id 列，执行迁移", table)
                cur.execute(
                    "ALTER TABLE {} ADD COLUMN tenant_id VARCHAR(64) "
                    "NOT NULL DEFAULT 'default'".format(table)
                )
                try:
                    cur.execute(
                        "ALTER TABLE {} ADD KEY {} (tenant_id)".format(table, index_name)
                    )
                except Exception as ie:
                    logger.info("tenant_id 索引已存在或创建失败（忽略）: %s", ie)
                logger.info("%s 表 tenant_id 列迁移完成", table)

    def _migrate_owner_user_id(self, cur):
        """检测 documents 表是否有 owner_user_id 列，缺失则补加 + 建索引。

        老库升级为文档级 ACL 时执行；存量文档 owner 为空（视为租户内共享，向后兼容）。
        """
        try:
            cur.execute("SELECT owner_user_id FROM documents LIMIT 1")
        except Exception:
            logger.info("documents 表缺少 owner_user_id 列，执行迁移")
            cur.execute(
                "ALTER TABLE documents ADD COLUMN owner_user_id VARCHAR(64) "
                "NOT NULL DEFAULT ''"
            )
            try:
                cur.execute(
                    "ALTER TABLE documents ADD KEY idx_documents_owner (owner_user_id)"
                )
            except Exception as ie:
                logger.info("owner_user_id 索引已存在或创建失败（忽略）: %s", ie)
            logger.info("documents 表 owner_user_id 列迁移完成")

    def ping(self) -> bool:
        """检查连接是否可用。"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone() is not None
        except Exception as e:
            logger.error("MySQL ping 失败: %s", e)
            return False
