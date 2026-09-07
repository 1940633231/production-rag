"""对账器单元测试：以 MySQL chunks 为事实源，修复派生索引（metadata/ES）。

覆盖:
  - metadata 缺失 → 补齐；孤儿 → 摘除
  - ES 缺失 → 补写；孤儿 → 删除
  - dry-run（fix=False）只报告不修复
  - MySQL 未启用 → skipped

运行:
  .venv\\Scripts\\python.exe -m pytest tests\\storage\\test_reconciler.py -v
"""
import pytest

from app.core.config import Config


def _chunk(vid, doc="doc-a", content=None):
    if content is None:
        content = "内容{}".format(vid)
    return {
        "chunk_id": "c{}".format(vid),
        "document_id": doc,
        "vector_id": vid,
        "version": 1,
        "chunk_index": vid,
        "content": content,
        "start_offset": 0,
        "end_offset": 0,
        "metadata": {},
    }


class FakeChunkRepo:
    def __init__(self, chunks):
        self._chunks = chunks

    def list_all(self):
        return self._chunks


class FakeMetadataStore:
    def __init__(self):
        self.loaded = {}
        self.saved = None

    def load(self, path):
        return dict(self.loaded)

    def save_entries(self, entries, path):
        self.saved = entries


class FakeESRepo:
    def __init__(self, es_chunks=None):
        self._es_chunks = list(es_chunks or [])
        self.inserted = []
        self.deleted_terms = []

    def list_all(self):
        return list(self._es_chunks)

    def batch_insert(self, docs, strategy=None):
        self.inserted.extend(docs)

    class _FakeClient:
        def delete_by_query(self, index, body, refresh=True):
            pass

    @property
    def _es(self):
        return self  # reconciler 访问 es_repo._es._client

    class _ClientHolder:
        pass

    _client = _FakeClient()


class FakeVectorStore:
    def __init__(self, ids=None):
        self._ids = list(ids or [])

    def ids(self):
        return list(self._ids)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """把对账器依赖全部替换为 fake。"""
    from app.storage import metadata_store as ms_mod
    import app.storage as storage_mod

    config = Config()
    # property 只读：类级 patch（monkeypatch 自动还原）
    monkeypatch.setattr(Config, "storage_mysql_enabled", True)
    monkeypatch.setattr(Config, "storage_es_enabled", True)
    # 指向临时目录，避免触碰真实 data/index
    monkeypatch.setattr(config, "index_dir_for", lambda strategy, tenant: tmp_path)

    meta = FakeMetadataStore()
    es = FakeESRepo()

    def fake_meta(*a, **kw):
        return meta

    monkeypatch.setattr(ms_mod, "MetadataStore", fake_meta)
    # 默认空事实源；_run 会再次替换
    monkeypatch.setattr(storage_mod, "ChunkRepository",
                        lambda mgr, strategy="recursive", tenant_id=None: FakeChunkRepo([]))
    return config, meta, es, monkeypatch


def _run(config, meta, es, monkeypatch, chunks, es_chunks=None, fix=True):
    from app.storage.reconciler import Reconciler
    import app.storage as storage_mod
    from app.storage import es_repository as es_mod

    es._es_chunks = list(es_chunks or [])
    monkeypatch.setattr(storage_mod, "ChunkRepository",
                        lambda mgr, strategy="recursive", tenant_id=None: FakeChunkRepo(chunks))
    monkeypatch.setattr(es_mod, "ChunkESRepository",
                        lambda strategy="recursive", tenant_id="default": es)
    # 对账器内部 create_vector_store 从 app.vector import → patch app.vector
    import app.vector as vector_mod
    monkeypatch.setattr(vector_mod, "create_vector_store", lambda **kw: FakeVectorStore())
    return Reconciler(config).reconcile("recursive", "default", fix=fix)


class TestReconcile:
    def test_metadata_missing_is_filled(self, env):
        config, meta, es, monkeypatch = env
        report = _run(config, meta, es, monkeypatch, [_chunk(1), _chunk(2)])
        assert report["metadata_missing"] == 2
        assert report["fixed_metadata_missing"] == 2
        assert meta.saved is not None
        assert "1" in meta.saved and "2" in meta.saved

    def test_metadata_orphan_removed(self, env, tmp_path):
        config, meta, es, monkeypatch = env
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")  # 文件需存在
        meta.loaded = {"99": _chunk(99)}  # MySQL 没有 99 → 孤儿
        report = _run(config, meta, es, monkeypatch, [_chunk(1)])
        assert report["metadata_orphan"] == 1
        assert report["fixed_metadata_orphan"] == 1
        assert meta.saved is not None and "99" not in meta.saved

    def test_es_missing_filled_and_orphan_deleted(self, env):
        config, meta, es, monkeypatch = env
        es_chunks = [_chunk(9)]  # 9 是孤儿（MySQL 无）
        report = _run(config, meta, es, monkeypatch,
                      [_chunk(1), _chunk(2)], es_chunks=es_chunks)
        assert report["es_missing"] == 2
        assert report["fixed_es_missing"] == 2
        assert len(es.inserted) == 2
        assert report["es_orphan"] == 1
        assert report["fixed_es_orphan"] == 1

    def test_dry_run_no_fix(self, env):
        config, meta, es, monkeypatch = env
        report = _run(config, meta, es, monkeypatch,
                      [_chunk(1)], fix=False)
        assert report["metadata_missing"] == 1
        assert report["fixed_metadata_missing"] == 0
        assert meta.saved is None
        assert es.inserted == []

    def test_mysql_disabled_skipped(self, env):
        config, meta, es, monkeypatch = env
        monkeypatch.setattr(Config, "storage_mysql_enabled", False)
        from app.storage.reconciler import Reconciler
        report = Reconciler(config).reconcile("recursive", "default")
        assert report["status"] == "skipped"
