r"""MySQL 存储层 CRUD 单元测试。

验证 MySQLManager / DocumentRepository / ChunkRepository 的全部 CRUD 操作，
包括外键 CASCADE 删除行为。

前置条件:
  1. MySQL 服务运行中（默认 127.0.0.1:3306）
  2. 环境变量已配置（MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE）
  3. 目标数据库已创建：CREATE DATABASE IF NOT EXISTS production_rag CHARACTER SET utf8mb4;

运行方式:
  # 确保环境变量
  set MYSQL_HOST=127.0.0.1
  set MYSQL_USER=root
  set MYSQL_PASSWORD=root
  set MYSQL_DATABASE=production_rag

  # 运行测试
  .venv\Scripts\python.exe -m pytest tests\storage\test_mysql_crud.py -v

  # 如果 MySQL 不可用，测试会自动 skip
"""
import os
import sys
import uuid
from pathlib import Path

import pytest

# 注入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env（pytest 不走 FastAPI 启动流程，需手动加载）
from app.core.env import load_env
load_env()

from app.storage.mysql import MySQLManager, _MYSQL_AVAILABLE
from app.storage.document_repository import DocumentRepository
from app.storage.chunk_repository import ChunkRepository
from app.ingestion.chunk import Chunk


# ---- fixtures ----

@pytest.fixture(scope="module")
def manager():
    """创建 MySQLManager 并初始化表结构（模块级共享）。

    如果 MySQL 不可用或连接失败，跳过全部测试。
    """
    if not _MYSQL_AVAILABLE:
        pytest.skip("pymysql/dbutils 未安装")

    mgr = MySQLManager()
    try:
        mgr.init_schema()
    except Exception as e:
        pytest.skip("MySQL 连接失败: {}。请确认 MySQL 服务运行中且环境变量已配置。".format(e))
    return mgr


@pytest.fixture(autouse=True)
def cleanup(manager):
    """每个测试前后清理测试数据，避免残留。

    使用 test_ 前缀标识测试数据，测试后统一清理。
    """
    yield
    # 测试后清理：删除所有 test_ 前缀的数据
    try:
        with manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE chunk_id LIKE 'test_%'")
                cur.execute("DELETE FROM documents WHERE document_id LIKE 'test_%'")
    except Exception:
        pass


# ---- 工具函数 ----

def _gen_id(prefix="test"):
    """生成唯一测试 ID。"""
    return "{}_{}".format(prefix, uuid.uuid4().hex[:8])


def _make_chunk(document_id, index=0, content="测试内容", prefix="test"):
    """构造 Chunk 对象。"""
    return Chunk(
        chunk_id="{}_chunk_{}_{}".format(prefix, document_id, index),
        document_id=document_id,
        chunk_index=index,
        content=content,
        start_offset=index * 100,
        end_offset=index * 100 + len(content),
        metadata={"source": "test", "page": index + 1},
    )


# ============================================================
# 1. MySQLManager 测试
# ============================================================

class TestMySQLManager:

    def test_ping(self, manager):
        """ping 应返回 True。"""
        assert manager.ping() is True

    def test_get_connection_auto_commit(self, manager):
        """get_connection 上下文正常退出时应自动 commit。"""
        doc_id = _gen_id()
        with manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (document_id, file_name, content_length) "
                    "VALUES (%s, %s, %s)",
                    (doc_id, "commit_test.txt", 100),
                )

        # 在新连接中验证数据已提交
        doc_repo = DocumentRepository(manager)
        doc = doc_repo.get(doc_id)
        assert doc is not None
        assert doc["file_name"] == "commit_test.txt"

    def test_get_connection_rollback_on_error(self, manager):
        """get_connection 上下文中抛异常时应自动 rollback。"""
        doc_id = _gen_id()
        try:
            with manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO documents (document_id, file_name, content_length) "
                        "VALUES (%s, %s, %s)",
                        (doc_id, "rollback_test.txt", 100),
                    )
                # 故意抛异常触发 rollback
                raise ValueError("故意触发回滚")
        except ValueError:
            pass

        # 验证数据已回滚
        doc_repo = DocumentRepository(manager)
        doc = doc_repo.get(doc_id)
        assert doc is None


# ============================================================
# 2. DocumentRepository 测试
# ============================================================

class TestDocumentRepository:

    def test_insert_and_get(self, manager):
        """插入文档后应能通过 get 查询到。"""
        repo = DocumentRepository(manager)
        doc_id = _gen_id()
        rows = repo.insert(doc_id, "test_doc.txt", 1024, source="data/raw/test_doc.txt")
        assert rows == 1

        doc = repo.get(doc_id)
        assert doc is not None
        assert doc["document_id"] == doc_id
        assert doc["file_name"] == "test_doc.txt"
        assert doc["content_length"] == 1024
        assert doc["source"] == "data/raw/test_doc.txt"

    def test_insert_ignore_duplicate(self, manager):
        """INSERT IGNORE 重复插入不应报错且不影响行数。"""
        repo = DocumentRepository(manager)
        doc_id = _gen_id()
        repo.insert(doc_id, "test_dup.txt", 100)

        # 重复插入
        rows = repo.insert(doc_id, "test_dup.txt", 100)
        assert rows == 0  # IGNORE 命中重复，affected=0

    def test_get_not_found(self, manager):
        """查询不存在的 document_id 应返回 None。"""
        repo = DocumentRepository(manager)
        doc = repo.get("test_nonexistent_" + uuid.uuid4().hex)
        assert doc is None

    def test_get_by_file_name(self, manager):
        """通过文件名应能查到文档。"""
        repo = DocumentRepository(manager)
        doc_id = _gen_id()
        file_name = "test_file_name_{}.txt".format(uuid.uuid4().hex[:8])
        repo.insert(doc_id, file_name, 512)

        doc = repo.get_by_file_name(file_name)
        assert doc is not None
        assert doc["document_id"] == doc_id

    def test_list_all(self, manager):
        """list_all 应返回多条记录。"""
        repo = DocumentRepository(manager)
        # 插入 3 条
        for i in range(3):
            repo.insert(_gen_id(), "test_list_{}.txt".format(i), 100 * (i + 1))

        docs = repo.list_all(limit=100)
        assert len(docs) >= 3
        # 验证按 created_at DESC 排序
        for i in range(len(docs) - 1):
            assert docs[i]["created_at"] >= docs[i + 1]["created_at"]

    def test_count(self, manager):
        """count 应返回文档总数。"""
        repo = DocumentRepository(manager)
        before = repo.count()
        repo.insert(_gen_id(), "test_count.txt", 100)
        after = repo.count()
        assert after == before + 1

    def test_delete(self, manager):
        """删除文档后 get 应返回 None。"""
        repo = DocumentRepository(manager)
        doc_id = _gen_id()
        repo.insert(doc_id, "test_delete.txt", 100)

        rows = repo.delete(doc_id)
        assert rows == 1
        assert repo.get(doc_id) is None


# ============================================================
# 3. ChunkRepository 测试
# ============================================================

class TestChunkRepository:

    def _setup_doc(self, doc_repo, doc_id=None):
        """创建测试文档（chunks 外键依赖）。"""
        doc_id = doc_id or _gen_id()
        doc_repo.insert(doc_id, "test_chunks.txt", 500)
        return doc_id

    def test_insert_and_get(self, manager):
        """插入 chunk 后应能通过 get 查询到。"""
        doc_repo = DocumentRepository(manager)
        chunk_repo = ChunkRepository(manager)
        doc_id = self._setup_doc(doc_repo)

        chunk_id = _gen_id("test_chunk")
        rows = chunk_repo.insert(
            chunk_id, doc_id, chunk_index=0,
            content="测试内容片段", start_offset=0, end_offset=6,
            metadata={"page": 1, "section": "intro"},
        )
        assert rows == 1

        chunk = chunk_repo.get(chunk_id)
        assert chunk is not None
        assert chunk["chunk_id"] == chunk_id
        assert chunk["document_id"] == doc_id
        assert chunk["chunk_index"] == 0
        assert chunk["content"] == "测试内容片段"
        assert chunk["start_offset"] == 0
        assert chunk["end_offset"] == 6
        assert chunk["metadata"]["page"] == 1
        assert chunk["metadata"]["section"] == "intro"

    def test_insert_without_metadata(self, manager):
        """metadata 为 None 时应正常插入。"""
        doc_repo = DocumentRepository(manager)
        chunk_repo = ChunkRepository(manager)
        doc_id = self._setup_doc(doc_repo)

        chunk_id = _gen_id("test_chunk")
        chunk_repo.insert(
            chunk_id, doc_id, chunk_index=0,
            content="无元数据", start_offset=0, end_offset=4,
            metadata=None,
        )
        chunk = chunk_repo.get(chunk_id)
        assert chunk is not None
        assert chunk["metadata"] is None

    def test_batch_insert(self, manager):
        """批量插入 Chunk 对象列表。"""
        doc_repo = DocumentRepository(manager)
        chunk_repo = ChunkRepository(manager)
        doc_id = self._setup_doc(doc_repo)

        chunks = [
            _make_chunk(doc_id, index=i, content="批量chunk_{}".format(i))
            for i in range(5)
        ]
        rows = chunk_repo.batch_insert(chunks)
        assert rows == 5

        # 验证每条都能查到
        for c in chunks:
            chunk = chunk_repo.get(c.chunk_id)
            assert chunk is not None
            assert chunk["content"] == c.content

    def test_batch_insert_empty(self, manager):
        """空列表 batch_insert 应返回 0。"""
        chunk_repo = ChunkRepository(manager)
        assert chunk_repo.batch_insert([]) == 0

    def test_get_by_document(self, manager):
        """get_by_document 应返回该文档的所有 chunks（按 chunk_index 排序）。"""
        doc_repo = DocumentRepository(manager)
        chunk_repo = ChunkRepository(manager)
        doc_id = self._setup_doc(doc_repo)

        # 插入 3 个 chunk（乱序插入）
        for i in [2, 0, 1]:
            chunk_repo.insert(
                "test_chunk_{}_d{}_{}".format(i, doc_id, uuid.uuid4().hex[:4]),
                doc_id, chunk_index=i, content="chunk_{}".format(i),
                start_offset=i * 10, end_offset=i * 10 + 7,
            )

        chunks = chunk_repo.get_by_document(doc_id)
        assert len(chunks) == 3
        # 验证按 chunk_index 升序排列
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1
        assert chunks[2]["chunk_index"] == 2

    def test_delete_by_document(self, manager):
        """delete_by_document 应删除该文档的所有 chunks。"""
        doc_repo = DocumentRepository(manager)
        chunk_repo = ChunkRepository(manager)
        doc_id = self._setup_doc(doc_repo)

        # 插入 3 个 chunk
        for i in range(3):
            chunk_repo.insert(
                "test_chunk_del_{}_{}".format(i, uuid.uuid4().hex[:4]),
                doc_id, chunk_index=i, content="chunk_{}".format(i),
                start_offset=0, end_offset=6,
            )

        rows = chunk_repo.delete_by_document(doc_id)
        assert rows == 3
        assert len(chunk_repo.get_by_document(doc_id)) == 0

    def test_count(self, manager):
        """count 应返回 chunk 总数。"""
        doc_repo = DocumentRepository(manager)
        chunk_repo = ChunkRepository(manager)
        doc_id = self._setup_doc(doc_repo)

        before = chunk_repo.count()
        chunk_repo.insert(
            _gen_id("test_chunk"), doc_id, chunk_index=0,
            content="count测试", start_offset=0, end_offset=8,
        )
        after = chunk_repo.count()
        assert after == before + 1


# ============================================================
# 4. 外键 CASCADE 删除测试
# ============================================================

class TestCascadeDelete:

    def test_delete_document_cascades_chunks(self, manager):
        """删除 documents 记录时，外键 CASCADE 应自动删除关联 chunks。"""
        doc_repo = DocumentRepository(manager)
        chunk_repo = ChunkRepository(manager)
        doc_id = _gen_id()
        doc_repo.insert(doc_id, "test_cascade.txt", 1000)

        # 插入 3 个 chunks
        for i in range(3):
            chunk_repo.insert(
                "test_cascade_chunk_{}_{}".format(i, uuid.uuid4().hex[:4]),
                doc_id, chunk_index=i, content="cascade_{}".format(i),
                start_offset=i * 10, end_offset=i * 10 + 8,
            )

        # 确认 chunks 存在
        assert len(chunk_repo.get_by_document(doc_id)) == 3

        # 删除文档（CASCADE 应自动删除 chunks）
        doc_repo.delete(doc_id)

        # 验证文档已删除
        assert doc_repo.get(doc_id) is None
        # 验证 chunks 已被 CASCADE 删除
        assert len(chunk_repo.get_by_document(doc_id)) == 0
