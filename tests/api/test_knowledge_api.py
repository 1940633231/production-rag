r"""知识库管理路由测试（#15）：upload / delete / documents / status。

覆盖:
  - upload 正常返回（mock 索引构建）
  - delete 删除文件 + 触发增量重建（mock MySQL / IndexWriter）
  - documents 无 MySQL 时回退文件扫描
  - status 返回两个策略的索引状态

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_knowledge_api.py -v
"""
import uuid
from types import SimpleNamespace

import pytest

from app.core.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def no_mysql(monkeypatch):
    """把 Config.storage_mysql_enabled 置为 False，跳过 MySQL 路径。

    路径助手委托给真实 Config（保持租户目录语义）。
    """
    import app.core.config as config_mod
    from app.core.config import Config as RealConfig

    class FakeConfig:
        storage_mysql_enabled = False
        storage_es_enabled = False
        storage_milvus_enabled = False

        @staticmethod
        def raw_dir_for(tenant_id="default"):
            return RealConfig().raw_dir_for(tenant_id)

        @staticmethod
        def index_dir_for(strategy, tenant_id="default"):
            return RealConfig().index_dir_for(strategy, tenant_id)

    monkeypatch.setattr(config_mod, "Config", FakeConfig)
    return FakeConfig


@pytest.fixture
def fake_index_writer(monkeypatch):
    """mock IndexWriter，记录增量重建调用。"""
    import app.ingestion.writer as writer_mod

    calls = {"strategies": []}

    class FakeIndexWriter:
        def __init__(self, config=None):
            pass

        def incremental_rebuild_after_delete(self, strategy, deleted_doc_ids, **kw):
            calls["strategies"].append(strategy)
            return {"document_count": 0}

    monkeypatch.setattr(writer_mod, "IndexWriter", FakeIndexWriter)
    return calls


class TestUploadRoute:
    def test_upload_returns_result(self, client, no_mysql, monkeypatch):
        import app.api.knowledge as knowledge

        monkeypatch.setattr(
            knowledge, "_do_upload",
            lambda path, strategy, tenant_id="default", owner_user_id="": {
                "document_count": 1, "chunk_count": 2, "dimension": 768,
                "index_path": "data/index/recursive/faiss.index",
                "metadata_path": "data/index/recursive/metadata.json",
            },
        )
        fname = "_pytest_kapi_{}.txt".format(uuid.uuid4().hex[:8])
        resp = client.post(
            "/api/knowledge/upload",
            files={"file": (fname, "内容".encode("utf-8"), "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 200
        assert resp.json()["chunk_count"] == 2
        import pathlib
        pathlib.Path("data/raw", fname).unlink()  # 清理


class TestDeleteRoute:
    def test_delete_removes_file_and_rebuilds(self, client, no_mysql,
                                              fake_index_writer, raw_dir,
                                              sync_background_rebuild):
        fname = "_pytest_del_{}.txt".format(uuid.uuid4().hex[:8])
        target = raw_dir / fname
        target.write_text("待删除文档", encoding="utf-8")

        resp = client.delete("/api/knowledge/{}".format(fname))
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_file"] is True
        assert not target.exists()  # 文件已删
        # 两个策略都已提交后台重建（sync fixture 下同步执行）
        assert data["rebuilt_indexes"] == ["fixed", "recursive"]
        assert data["task_id"]
        assert set(fake_index_writer["strategies"]) == {"fixed", "recursive"}

    def test_delete_not_found_404(self, client, no_mysql, raw_dir):
        resp = client.delete("/api/knowledge/_pytest_missing_{}".format(
            uuid.uuid4().hex[:8]
        ))
        assert resp.status_code == 404


class TestListDocuments:
    def test_fallback_to_raw_dir_scan(self, client, no_mysql, raw_dir):
        fname = "_pytest_list_{}.txt".format(uuid.uuid4().hex[:8])
        target = raw_dir / fname
        target.write_text("文档", encoding="utf-8")
        try:
            resp = client.get("/api/knowledge/documents")
            assert resp.status_code == 200
            data = resp.json()
            assert "total" in data
            names = [d["file_name"] for d in data["documents"]]
            assert fname in names  # 扫描到了测试文件
        finally:
            target.unlink()


class TestIndexStatus:
    def test_status_returns_both_strategies(self, client):
        resp = client.get("/api/knowledge/status")
        assert resp.status_code == 200
        indexes = resp.json()["indexes"]
        assert "fixed" in indexes and "recursive" in indexes
        # 字段齐全（真实 data/index 可能不存在）
        for strategy, info in indexes.items():
            assert "faiss_exists" in info
            assert "metadata_exists" in info
            assert "chunk_count" in info
