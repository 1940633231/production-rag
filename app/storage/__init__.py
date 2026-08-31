"""存储层：MySQL 持久化（documents + chunks 表 CRUD）+ 多后端抽象。

依赖:
    pip install pymysql dbutils

使用:
    from app.storage import MySQLManager, DocumentRepository, ChunkRepository

    mgr = MySQLManager()
    mgr.init_schema()
    doc_repo = DocumentRepository(mgr)
    chunk_repo = ChunkRepository(mgr, strategy="recursive")

多后端读接口:
    from app.storage import create_chunk_repo, BaseChunkRepository
    repo = create_chunk_repo(config, strategy="recursive")
"""
from app.storage.base import (
    BaseChunkRepository,
    MetadataChunkRepository,
)
from app.storage.mysql import MySQLManager, _MYSQL_AVAILABLE
from app.storage.document_repository import DocumentRepository
from app.storage.chunk_repository import ChunkRepository
from app.storage.es_repository import ChunkESRepository

__all__ = [
    "MySQLManager",
    "DocumentRepository",
    "ChunkRepository",
    "BaseChunkRepository",
    "MetadataChunkRepository",
    "ChunkESRepository",
    "create_chunk_repo",
    "_MYSQL_AVAILABLE",
]


def create_chunk_repo(config=None, manager=None, strategy="recursive",
                      tenant_id: str = "default"):
    """工厂函数：按 config.storage.backends 创建 chunk 仓库实例。

    后端选择优先级:
      1. storage.backends.es.enabled=true → ChunkESRepository
      2. storage.backends.mysql.enabled=true（或 storage.enabled=true 向后兼容）→ ChunkRepository
      3. 均未启用 → 返回 None（由调用方降级到 MetadataStore）

    参数:
        config: Config 实例（为 None 时新建）
        manager: 可选的 MySQLManager 实例（复用连接池）
        strategy: 分块策略（'fixed'/'recursive'），用于按策略隔离 chunks
        tenant_id: 租户 ID，用于按租户隔离 chunks（'default' 为单租户旧行为）

    返回:
        BaseChunkRepository 子类实例，或 None

    异常:
        后端初始化失败时抛出（由调用方 try/except 降级到 MetadataStore）
    """
    if config is None:
        from app.core.config import Config
        config = Config()

    # ES 后端（全文检索）
    if config.storage_es_enabled:
        from app.storage.es_repository import ChunkESRepository
        return ChunkESRepository(strategy=strategy, tenant_id=tenant_id)

    # MySQL 后端（结构化存储）
    if config.storage_mysql_enabled:
        from app.storage.chunk_repository import ChunkRepository
        from app.storage.mysql import MySQLManager
        mgr = manager or MySQLManager(pool_size=config.storage_pool_size)
        return ChunkRepository(mgr, strategy=strategy, tenant_id=tenant_id)

    return None

