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
        assert w._resolve_doc_versions(docs, "default", "recursive", bump=True) == {"report": 1}

    def test_bump_existing_document(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)

        class FakeDocRepo:
            def next_version(self, doc_id, strategy, tenant_id=None):
                return {"report": 2, "newdoc": 1}.get(doc_id)

        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: FakeDocRepo()
        )
        from app.ingestion.document import Document

        docs = [Document(document_id="report", content="x", metadata={}),
                Document(document_id="newdoc", content="y", metadata={})]
        versions = w._resolve_doc_versions(docs, "default", "recursive", bump=True)
        assert versions == {"report": 2, "newdoc": 1}   # 已存在 +1，新文档 1

    def test_rebuild_keeps_current(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)

        class FakeDocRepo:
            def get_active_versions(self, document_ids, strategy, tenant_id=None):
                return {"report": 3}   # 该策略当前活跃 v3

        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: FakeDocRepo()
        )
        from app.ingestion.document import Document

        docs = [Document(document_id="report", content="x", metadata={})]
        # bump=False（rebuild）：保持 v3，不推进
        assert w._resolve_doc_versions(docs, "default", "recursive", bump=False) == {"report": 3}


class _LedgerDocRepo:
    """记录调用的 DocumentRepository 替身（per-strategy 契约）。"""
    def __init__(self):
        self.deleted = []
        self.inserted = []
        self.active = {}

    def set_active(self, active_map):
        self.active = active_map

    def get_active_versions(self, document_ids, strategy, tenant_id=None):
        return {d: v for d, v in self.active.items() if d in document_ids}

    def next_version(self, doc_id, strategy, tenant_id=None):
        return int(self.active.get(doc_id, 0)) + 1

    def has_version(self, doc_id, strategy, version, tenant_id=None):
        return True

    def delete_version(self, doc_id, strategy, version, tenant_id=None):
        self.deleted.append((doc_id, strategy, version))

    def insert_version(self, doc_id, strategy, version, tenant_id="default",
                       chunk_count=0):
        self.inserted.append((doc_id, strategy, version))


class _FakeChunkRepo:
    def get_vector_ids_by_document(self, document_id, tenant_id=None,
                                   version=None, strategy=None):
        return []


class TestFinalizeVersions:
    def test_new_document_no_gc(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)
        gced = []
        fdr = _LedgerDocRepo()
        monkeypatch.setattr("app.storage.DocumentRepository", lambda manager=None: fdr)
        monkeypatch.setattr("app.storage.ChunkRepository", lambda manager=None, **kw: _FakeChunkRepo())
        monkeypatch.setattr(w, "_gc_version", lambda *a, **k: gced.append(a))
        w._finalize_versions({"report": 1}, "recursive", "default")
        assert gced == []                  # v1 新文档：无旧版本
        assert fdr.inserted == [("report", "recursive", 1)]  # 已记录台账

    def test_latest_gc_previous_version(self, monkeypatch):
        """retention=latest，上传到 v2 → 记台账 v2，GC v1。"""
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        w = IndexWriter(config=cfg)
        gced = []
        fdr = _LedgerDocRepo()
        monkeypatch.setattr("app.storage.DocumentRepository", lambda manager=None: fdr)
        monkeypatch.setattr("app.storage.ChunkRepository", lambda manager=None, **kw: _FakeChunkRepo())
        monkeypatch.setattr(w, "_gc_version",
                            lambda doc_id, v, strategy, tenant, repo: gced.append((doc_id, v)))
        w._finalize_versions({"report": 2}, "recursive", "default")
        assert gced == [("report", 1)]                     # GC 旧版本 v1（当前策略）
        assert ("report", "recursive", 1) in fdr.deleted   # 台账同步删 v1


class TestRetentionPolicy:
    """retention=N（P2）：保留最近 N 版，清理更早版本 / 回滚丢弃更新版本（per-strategy）。"""

    @pytest.fixture
    def w(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = True
        cfg.document_versioning_retention = "3"
        w = IndexWriter(config=cfg)

        class _NoopLock:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(w, "_write_lock", lambda strat, tenant: _NoopLock())
        return w

    def test_retention_n_prunes_beyond_keep(self, monkeypatch, w):
        """retention=3, 上传到 v5 → 保留 v3/v4/v5，GC v1/v2（当前策略）。"""
        fdr = _LedgerDocRepo()
        monkeypatch.setattr("app.storage.DocumentRepository", lambda manager=None: fdr)
        monkeypatch.setattr("app.storage.ChunkRepository", lambda manager=None, **kw: _FakeChunkRepo())
        gced = []
        monkeypatch.setattr(w, "_gc_version",
                            lambda doc_id, v, strategy, tenant, repo: gced.append(v))
        w._finalize_versions({"report": 5}, "recursive", "default")
        assert fdr.inserted[0][2] == 5                     # 台账记 v5
        assert sorted(gced) == [1, 2]                      # 3 之前的被清
        assert ("report", "recursive", 1) in fdr.deleted
        assert ("report", "recursive", 2) in fdr.deleted

    def test_rollback_discards_newer_versions(self, monkeypatch, w):
        """回滚 v4 → v2（recursive）：GC 并删台账 v3/v4（仅当前策略）。"""
        fdr = _LedgerDocRepo()
        fdr.set_active({"report": 4})
        monkeypatch.setattr("app.storage.DocumentRepository", lambda manager=None: fdr)
        gced = []
        monkeypatch.setattr(w, "_gc_version",
                            lambda doc_id, v, strategy, tenant, repo: gced.append(v))
        bumps = []
        monkeypatch.setattr(w, "_bump_version", lambda strat, tenant: bumps.append(strat))

        new_ver = w.rollback_document("report", 2, "recursive", "default")
        assert new_ver == 2
        assert sorted(gced) == [3, 4]                      # 丢弃 3、4（仅 recursive）
        assert ("report", "recursive", 3) in fdr.deleted
        assert ("report", "recursive", 4) in fdr.deleted
        assert bumps == ["recursive"]


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
                                           version=None, strategy=None):
                return [10, 20]

            def delete_by_document_version(self, document_id, version,
                                           tenant_id=None, strategy=None):
                deleted_db.append((document_id, version, strategy))

        monkeypatch.setattr(
            w, "_remove_vectors",
            lambda ids, strategy, tenant_id: removed_vectors.append((strategy, ids)),
        )
        monkeypatch.setattr(
            w, "_remove_metadata",
            lambda doc_id, version, strategy, tenant_id:
                removed_meta.append((strategy, doc_id, version)),
        )
        w._gc_version("report", 1, "recursive", "default", FakeChunkRepo())
        # 策略隔离：只清当前策略（recursive），MySQL 按 strategy 过滤
        assert removed_vectors == [("recursive", [10, 20])]
        assert removed_meta == [("recursive", "report", 1)]
        assert deleted_db == [("report", 1, "recursive")]

    def test_gc_failure_is_caught(self, monkeypatch):
        cfg = _Cfg()
        w = IndexWriter(config=cfg)

        class BoomChunkRepo:
            def get_vector_ids_by_document(self, **kw):
                raise RuntimeError("db down")

        # 不应抛异常（仅告警）
        w._gc_version("report", 1, "recursive", "default", BoomChunkRepo())


class TestOverwritePurge:
    """覆盖更新（版本化关闭时的同名文档处理）。"""

    def test_purge_only_existing_documents(self, monkeypatch):
        cfg = _Cfg()
        cfg.document_versioning_enabled = False
        w = IndexWriter(config=cfg)
        purged = []

        class FakeDocRepo:
            def get(self, doc_id, tenant_id=None):
                return {"document_id": doc_id} if doc_id == "report" else None

        class FakeChunkRepo:
            def get_vector_ids_by_document(self, document_id, tenant_id=None):
                return []

        monkeypatch.setattr(
            "app.storage.DocumentRepository", lambda manager=None: FakeDocRepo()
        )
        monkeypatch.setattr(
            "app.storage.ChunkRepository",
            lambda manager=None, strategy=None, tenant_id=None: FakeChunkRepo(),
        )
        monkeypatch.setattr(
            w, "_purge_document", lambda doc_id, strategy, tenant_id, chunk_repo:
                purged.append(doc_id),
        )
        from app.ingestion.document import Document

        docs = [Document(document_id="report", content="x", metadata={}),
                Document(document_id="newdoc", content="y", metadata={})]
        w._overwrite_purge(docs, "recursive", "default")
        assert purged == ["report"]   # 只清同名已存在的，新文档跳过

    def test_purge_document_cleans_all_backends(self, monkeypatch):
        cfg = _Cfg()
        cfg.storage_es_enabled = False
        w = IndexWriter(config=cfg)
        removed_vectors = []
        removed_meta = []
        deleted_db = []

        class FakeChunkRepo:
            def get_vector_ids_by_document(self, document_id, tenant_id=None,
                                           version=None, strategy=None):
                return [10, 20]

            def delete_by_document(self, document_id, tenant_id=None,
                                   strategy=None):
                deleted_db.append((document_id, strategy))

        monkeypatch.setattr(
            w, "_remove_vectors",
            lambda ids, strategy, tenant_id: removed_vectors.append((strategy, ids)),
        )
        monkeypatch.setattr(
            w, "_remove_metadata_document",
            lambda doc_id, strategy, tenant_id: removed_meta.append((strategy, doc_id)),
        )
        w._purge_document("report", "recursive", "default", FakeChunkRepo())
        # 策略隔离：只清当前策略（recursive）；documents 行保留
        assert removed_vectors == [("recursive", [10, 20])]
        assert removed_meta == [("recursive", "report")]
        assert deleted_db == [("report", "recursive")]

    def test_purge_db_failure_raises(self, monkeypatch):
        cfg = _Cfg()
        cfg.storage_es_enabled = False
        w = IndexWriter(config=cfg)

        class BoomChunkRepo:
            def get_vector_ids_by_document(self, document_id, tenant_id=None,
                                           version=None, strategy=None):
                return []

            def delete_by_document(self, document_id, tenant_id=None,
                                   strategy=None):
                raise RuntimeError("db down")

        # MySQL 删除失败 → 抛异常中止写入（避免半覆盖不一致）
        with pytest.raises(RuntimeError):
            w._purge_document("report", "recursive", "default", BoomChunkRepo())


class TestActiveVersionFilter:
    """检索活跃过滤：只保留每文档活跃版本候选。"""
    @staticmethod
    def _pipeline(provider):
        from app.rag.pipeline import RAGPipeline
        p = RAGPipeline(retriever=object())
        p.active_version_provider = provider
        return p

    def test_filters_non_active_versions(self):
        p = self._pipeline(lambda ids: {"a": 2, "b": 1})
        cands = [
            {"document_id": "a", "version": 2},  # 活跃
            {"document_id": "a", "version": 1},  # 非活跃 → 弃
            {"document_id": "b", "version": 1},  # 活跃
            {"document_id": "b", "version": 3},  # 非活跃 → 弃
        ]
        kept = p._filter_active_versions(cands)
        assert [c["version"] for c in kept] == [2, 1]

    def test_no_provider_passthrough(self):
        p = self._pipeline(None)
        cands = [{"document_id": "a", "version": 9}]
        assert p._filter_active_versions(cands) == cands

    def test_unknown_doc_passthrough(self):
        p = self._pipeline(lambda ids: {"a": 1})
        cands = [{"document_id": "a", "version": 1},
                 {"document_id": "ghost", "version": 5}]
        kept = p._filter_active_versions(cands)
        assert len(kept) == 2


class TestDocumentVersionsMigration:
    """老库 document_versions 缺 strategy 列时的迁移（Unknown column 修复）。"""

    def test_adds_strategy_and_rebuilds_pk(self):
        from app.storage.mysql import MySQLManager

        mgr = object.__new__(MySQLManager)
        calls = []

        class FakeCur:
            def execute(self, sql, *a):
                if sql.startswith("SELECT strategy FROM document_versions"):
                    raise Exception("Unknown column 'strategy'")
                calls.append(sql)

        mgr._migrate_document_versions_strategy(FakeCur())
        assert any("ADD COLUMN strategy" in s for s in calls)
        pk_stmts = [s for s in calls if "PRIMARY KEY" in s]
        assert len(pk_stmts) == 2  # DROP + ADD 复合主键

    def test_skips_when_strategy_present(self):
        from app.storage.mysql import MySQLManager

        mgr = object.__new__(MySQLManager)
        ran = []

        class FakeCur:
            def execute(self, sql, *a):
                ran.append(sql)   # SELECT strategy 成功 → 不触发迁移

        mgr._migrate_document_versions_strategy(FakeCur())
        assert ran == ["SELECT strategy FROM document_versions LIMIT 1"]

    def test_drop_stale_current_version_column(self):
        from app.storage.mysql import MySQLManager

        mgr = object.__new__(MySQLManager)
        calls = []

        class FakeCur:
            def execute(self, sql, *a):
                calls.append(sql)   # SELECT current_version 成功 → 执行 DROP

        mgr._migrate_documents_remove_current_version(FakeCur())
        assert any("DROP COLUMN current_version" in s for s in calls)

    def test_remove_current_version_noop_when_missing(self):
        from app.storage.mysql import MySQLManager

        mgr = object.__new__(MySQLManager)
        calls = []

        class FakeCur:
            def execute(self, sql, *a):
                if sql.startswith("SELECT current_version"):
                    raise Exception("no column")
                calls.append(sql)

        mgr._migrate_documents_remove_current_version(FakeCur())
        assert calls == []
