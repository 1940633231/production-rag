r"""端到端闭环测试（#16）：upload → chat → delete 全链路。

依赖均 mock（索引构建 / LLM 生成 / MySQL），验证 API 层路由协作正确。
真实链路（索引构建、模型加载）由各模块单元测试覆盖。

流程:
  1. POST /api/knowledge/upload   上传文档（文件真实落盘 data/raw）
  2. POST /api/chat               问答（mock service）
  3. POST /api/chat/stream        流式问答（SSE 事件序列）
  4. DELETE /api/knowledge/{id}   删除文档（文件删除 + 增量重建）

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_e2e.py -v
"""
import uuid
from types import SimpleNamespace

import pytest

from app.core.logger import get_logger

logger = get_logger(__name__)


class FakeService:
    """mock RAGService：同步/流式均返回固定结果。"""

    def query(self, query, history=None, document_ids=None):
        from app.rag.result import RAGResponse
        return RAGResponse(
            query=query,
            context="[1] 测试上下文",
            chunks=[{"chunk_id": "c1", "content": "测试上下文"}],
            stats={"query_count": 1},
            answer="测试回答[1]",
            citations=[{"number": 1, "chunk_id": "c1"}],
        )

    def query_stream(self, query, history=None, document_ids=None):
        yield {"type": "meta", "chunks": [], "stats": {}}
        yield {"type": "delta", "content": "测试回答"}
        yield {"type": "citations", "citations": []}
        yield {"type": "done", "answer_length": 4}


@pytest.fixture
def mock_deps(monkeypatch):
    """mock 全部外部依赖：索引构建 / LLM 生成 / MySQL / IndexWriter。"""
    import app.api.chat as chat
    import app.api.knowledge as knowledge
    import app.core.config as config_mod
    import app.ingestion.writer as writer_mod

    # 1. chat 服务
    monkeypatch.setattr(chat, "get_service", lambda **kw: FakeService())

    # 2. 上传索引构建
    monkeypatch.setattr(
        knowledge, "_do_upload",
        lambda path, strategy, tenant_id="default", owner_user_id="": {
            "document_count": 1, "chunk_count": 2, "dimension": 768,
            "index_path": "data/index/recursive/faiss.index",
            "metadata_path": "data/index/recursive/metadata.json",
        },
    )

    # 3. 无 MySQL
    from app.core.config import Config as RealConfig

    class FakeConfig:
        storage_mysql_enabled = False
        storage_es_enabled = False
        storage_milvus_enabled = False
        cache_enabled = False  # e2e 流程关闭查询缓存
        scope_enabled = False  # Query Scope 关闭（默认）
        scope_mode = "auto"
        scope_require_entity = False
        scope_match_top_k = 10

        @staticmethod
        def raw_dir_for(tenant_id="default"):
            return RealConfig().raw_dir_for(tenant_id)

        @staticmethod
        def index_dir_for(strategy, tenant_id="default"):
            return RealConfig().index_dir_for(strategy, tenant_id)

    monkeypatch.setattr(config_mod, "Config", FakeConfig)

    # 4. IndexWriter mock
    class FakeIndexWriter:
        def __init__(self, config=None):
            pass

        def incremental_rebuild_after_delete(self, strategy, deleted_doc_ids, **kw):
            return {"document_count": 0}

    monkeypatch.setattr(writer_mod, "IndexWriter", FakeIndexWriter)


class TestEndToEnd:
    def _unique_name(self, tag):
        return "_pytest_e2e_{}_{}.txt".format(tag, uuid.uuid4().hex[:8])

    def test_upload_chat_delete_loop(self, client, mock_deps, raw_dir):
        # ---- 1. upload：文件真实落盘 ----
        fname = self._unique_name("doc")
        resp = client.post(
            "/api/knowledge/upload",
            files={"file": (fname, "铁矿石供需数据文档内容。".encode("utf-8"), "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 200
        assert resp.json()["document_count"] == 1
        assert (raw_dir / fname).exists()

        # ---- 2. chat：同步问答 ----
        resp = client.post("/api/chat", json={"query": "铁矿供需如何？"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "测试回答[1]"
        assert data["query"] == "铁矿供需如何？"
        assert data["citations"][0]["number"] == 1

        # ---- 3. chat stream：SSE 事件序列 ----
        with client.stream(
            "POST", "/api/chat/stream", json={"query": "铁矿供需"}
        ) as resp_stream:
            assert resp_stream.status_code == 200
            text = resp_stream.read().decode("utf-8")
            # SSE 事件包含 meta / delta / done
            assert "event: meta" in text
            assert "event: delta" in text
            assert "event: done" in text
            assert "测试回答" in text

        # ---- 4. delete：删除文档 + 文件清理 ----
        resp = client.delete("/api/knowledge/{}".format(fname))
        assert resp.status_code == 200
        assert resp.json()["deleted_file"] is True
        assert not (raw_dir / fname).exists()

    def test_chat_rejects_stream_flag(self, client, mock_deps):
        """stream=true 打到 /api/chat 应返回 400 提示使用流式端点。"""
        resp = client.post("/api/chat", json={"query": "q", "stream": True})
        assert resp.status_code == 400
        assert "stream" in resp.json()["detail"]

    def test_chat_missing_query_422(self, client, mock_deps):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422
