"""文档版本化（P1）单元测试：chunk 版本化 / 版本解析 / 活跃切换 + GC。

覆盖:
  - _apply_version_to_chunks: 未启用保持旧格式；启用时 chunk_id 含版本 + metadata.version
  - _resolve_doc_versions: upload bump / rebuild 保持 / 未启用全 1
  - _finalize_versions: 新文档跳过；已存在 → 原子切换 + GC 旧版本；并发冲突跳过 GC
  - _gc_version: 向量 / metadata / MySQL / ES 各端清理

运行:
  .venv\\Scripts\\python.exe -m pytest tests\\ingestion\\test_versioning.py -v
"""
import pytest

from app.ingestion.chunk import Chunk
from app.ingestion.writer import IndexWriter


class _Cfg:
    """最小配置替身（版本化可开关）。"""
    document_versioning_enabled = False
    document_versioning_retention = "latest"
    storage_mysql_enabled = True
    storage_pool_size = 2
    storage_milvus_enabled = False
    storage_es_enabled = False

    @staticmethod
    def index_dir_for(strategy, tenant_id="default"):
        from app.core.config import Config
        return Config().index_dir_for(strategy, tenant_id)


def _chunk(doc_id="report", idx=0):
    return Chunk(
        chunk_id="{}_chunk_{}".format(doc_id, idx),
        document_id=doc_id,
        chunk_index=idx,
        content="内容{}".format(idx),
    )


class TestApplyVersionToChunks:
    def test_disabled_keeps_old_format(self):
        cfg = _Cfg()
        cfg.document_versioning_enabled = False
        w = IndexWriter(config=cfg)
        c = _chunk()
        w._apply_version_to_chunks([c], {"report": 1})
        assert c.chunk_id == "report_chunk_0"   # 未启用：保持 chunker 格式
        assert c.version == 1
        assert "version" not in c.metadata

    def test_enabled_embeds_version(self):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)
        c = _chunk()
        w._apply_version_to_chunks([c], {"report": 2})
        assert c.chunk_id == "report_v2_chunk_0"
        assert c.version == 2
        assert c.metadata["version"] == 2

    def test_idempotent_reapply_same_version(self):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)
        c = _chunk()
        w._apply_version_to_chunks([c], {"report": 2})
        first = c.chunk_id
        w._apply_version_to_chunks([c], {"report": 2})
        assert c.chunk_id == first == "report_v2_chunk_0"


class TestResolveDocVersions:
    def test_disabled_returns_one(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = False
        w = IndexWriter(config=cfg)
        from app.ingestion.document import Document

        docs = [Document(document_id="report", content="x", metadata={})]
        assert w._resolve_doc_versions(docs, "default", bump=True) == {"report": 1}

    def test_bump_existing_document(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)

        class FakeDocRepo:
            def get_current_version(self, doc_id, tenant_id=None):
                return {"report": 1, "newdoc": None}.get(doc_id)

        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: FakeDocRepo()
        )
        from app.ingestion.document import Document

        docs = [Document(document_id="report", content="x", metadata={}),
                Document(document_id="newdoc", content="y", metadata={})]
        versions = w._resolve_doc_versions(docs, "default", bump=True)
        assert versions == {"report": 2, "newdoc": 1}   # 已存在 +1，新文档 1

    def test_rebuild_keeps_current(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)

        class FakeDocRepo:
            def get_current_version(self, doc_id, tenant_id=None):
                return 3   # 当前活跃 v3

        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: FakeDocRepo()
        )
        from app.ingestion.document import Document

        docs = [Document(document_id="report", content="x", metadata={})]
        # bump=False（rebuild）：保持 v3，不推进
        assert w._resolve_doc_versions(docs, "default", bump=False) == {"report": 3}


class TestFinalizeVersions:
    def test_new_document_skips_switch_and_gc(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)
        called = []
        monkeypatch.setattr(w, "_gc_version", lambda *a, **k: called.append(a))
        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: object()
        )
        monkeypatch.setattr(
            "app.storage.ChunkRepository", lambda manager=None, strategy=None,
            tenant_id=None: object()
        )
        w._finalize_versions({"report": 1}, "recursive", "default")
        assert called == []   # v1 新文档：无旧版本可 GC

    def test_existing_document_switches_then_gc(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)
        switched = []
        gced = []

        class FakeDocRepo:
            def set_current_version(self, doc_id, new_version,
                                    expected_version=None, tenant_id=None):
                switched.append((doc_id, new_version, expected_version))
                return True

        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: FakeDocRepo()
        )
        monkeypatch.setattr(
            "app.storage.ChunkRepository", lambda manager=None, strategy=None,
            tenant_id=None: object()
        )
        monkeypatch.setattr(
            w, "_gc_version",
            lambda doc_id, version, strategy, tenant_id, chunk_repo:
                gced.append((doc_id, version)),
        )
        w._finalize_versions({"report": 2}, "recursive", "default")
        assert switched == [("report", 2, 1)]   # 原子切换 1 → 2
        assert gced == [("report", 1)]          # GC 旧版本 v1

    def test_switch_conflict_skips_gc(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)
        gced = []

        class FakeDocRepo:
            def set_current_version(self, doc_id, new_version,
                                    expected_version=None, tenant_id=None):
                return False   # 并发冲突（current_version 已不是 1）

        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: FakeDocRepo()
        )
        monkeypatch.setattr(
            "app.storage.ChunkRepository", lambda manager=None, strategy=None,
            tenant_id=None: object()
        )
        monkeypatch.setattr(
            w, "_gc_version",
            lambda doc_id, version, strategy, tenant_id, chunk_repo:
                gced.append((doc_id, version)),
        )
        w._finalize_versions({"report": 2}, "recursive", "default")
        assert gced == []   # 切换失败 → 不 GC（避免误删他人新版本）


class TestGcVersion:
    def test_gc_cleans_all_backends(self, monkeypatch):
        cfg = _Cfg()
        cfg.storage_es_enabled = False
        w = IndexWriter(config=cfg)
        removed_vectors = []
        removed_meta = []
        deleted_db = []

        class FakeChunkRepo:
            def get_vector_ids_by_document(self, document_id, tenant_id=None,
                                           version=None):
                return [10, 20]

            def delete_by_document_version(self, document_id, version,
                                           tenant_id=None):
                deleted_db.append((document_id, version))

        monkeypatch.setattr(
            w, "_remove_vectors",
            lambda ids, strategy, tenant_id: removed_vectors.append(ids),
        )
        monkeypatch.setattr(
            w, "_remove_metadata",
            lambda doc_id, version, strategy, tenant_id:
                removed_meta.append((doc_id, version)),
        )
        w._gc_version("report", 1, "recursive", "default", FakeChunkRepo())
        assert removed_vectors == [[10, 20]]
        assert removed_meta == [("report", 1)]
        assert deleted_db == [("report", 1)]

    def test_gc_failure_is_caught(self, monkeypatch):
        cfg = _Cfg()
        w = IndexWriter(config=cfg)

        class BoomChunkRepo:
            def get_vector_ids_by_document(self, **kw):
                raise RuntimeError("db down")

        # 不应抛异常（仅告警）
        w._gc_version("report", 1, "recursive", "default", BoomChunkRepo())
