"""发布前校验（_validate_build）单元测试：Build → Validation → Publish 分离。

覆盖:
  - 各端就绪 → 校验通过（不抛异常）
  - metadata 缺失 → strict 抛异常中止发布 / 非 strict 告警继续
  - ES 缺失 → strict 抛异常
  - 向量后端（FAISS）缺失 → strict 抛异常
  - write_validation.enabled=false → 跳过校验

运行:
  .venv\\Scripts\\python.exe -m pytest tests\\ingestion\\test_write_validation.py -v
"""
from types import SimpleNamespace

import pytest

from app.core.config import Config
from app.ingestion.writer import IndexWriter


def _chunk(vid):
    return SimpleNamespace(vector_id=vid)


class FakeMetaStore:
    def __init__(self, ids):
        self._ids = set(ids)

    def load(self, path):
        return {str(i): {"vector_id": i} for i in self._ids}


class FakeESRepo:
    def __init__(self, ids):
        self._ids = list(ids)

    def list_all(self):
        return [{"vector_id": i} for i in self._ids]


class FakeVecStore:
    def __init__(self, ids):
        self._ids = list(ids)

    def ids(self):
        return list(self._ids)

    def load(self, path):
        pass


@pytest.fixture
def setup(monkeypatch, tmp_path):
    """替换依赖：config 类级属性 + 临时索引目录 + fake 存储。"""
    from app.storage import metadata_store as ms_mod
    from app.storage import es_repository as es_mod
    import app.vector as vector_mod

    monkeypatch.setattr(Config, "write_validation_enabled", True)
    monkeypatch.setattr(Config, "write_validation_strict", True)
    monkeypatch.setattr(Config, "storage_es_enabled", False)
    monkeypatch.setattr(Config, "storage_milvus_enabled", False)
    monkeypatch.setattr(Config, "vector_index_type", "flat")

    config = Config()
    monkeypatch.setattr(config, "index_dir_for", lambda strategy, tenant: tmp_path)

    # 注册 fake 到各自模块（_validate_build 在函数内 import）
    monkeypatch.setattr(ms_mod, "MetadataStore",
                        lambda *a, **kw: FakeMetaStore([1, 2]))
    monkeypatch.setattr(vector_mod, "create_vector_store",
                        lambda **kw: FakeVecStore([1, 2]))

    class _Holder:
        es = None

    def fake_es_repo(strategy="recursive", tenant_id="default"):
        return _Holder.es or FakeESRepo([1, 2])

    monkeypatch.setattr(es_mod, "ChunkESRepository", fake_es_repo)
    holder = _Holder

    # 预置 FAISS 索引文件存在（否则向量校验跳过）
    (tmp_path / "faiss.index").write_bytes(b"fake")
    return IndexWriter(config), monkeypatch, tmp_path, holder


class TestValidateBuild:
    def test_all_ready_passes(self, setup):
        writer, monkeypatch, tmp_path, holder = setup
        writer._validate_build([_chunk(1), _chunk(2)], "recursive", "default")
        # 不抛异常即通过

    def test_metadata_missing_strict_raises(self, setup):
        writer, monkeypatch, tmp_path, holder = setup
        from app.storage import metadata_store as ms_mod
        monkeypatch.setattr(ms_mod, "MetadataStore",
                            lambda *a, **kw: FakeMetaStore([1]))  # 缺 2
        with pytest.raises(RuntimeError, match="写入校验失败"):
            writer._validate_build([_chunk(1), _chunk(2)], "recursive", "default")

    def test_metadata_missing_non_strict_continues(self, setup):
        writer, monkeypatch, tmp_path, holder = setup
        from app.storage import metadata_store as ms_mod
        monkeypatch.setattr(Config, "write_validation_strict", False)
        monkeypatch.setattr(ms_mod, "MetadataStore",
                            lambda *a, **kw: FakeMetaStore([1]))  # 缺 2
        writer._validate_build([_chunk(1), _chunk(2)], "recursive", "default")
        # 非 strict：仅告警，不抛

    def test_es_missing_strict_raises(self, setup):
        writer, monkeypatch, tmp_path, holder = setup
        from app.storage import es_repository as es_mod
        monkeypatch.setattr(Config, "storage_es_enabled", True)
        monkeypatch.setattr(es_mod, "ChunkESRepository",
                            lambda strategy="recursive", tenant_id="default": FakeESRepo([1]))  # 缺 2
        with pytest.raises(RuntimeError, match="写入校验失败"):
            writer._validate_build([_chunk(1), _chunk(2)], "recursive", "default")

    def test_vector_missing_strict_raises(self, setup):
        writer, monkeypatch, tmp_path, holder = setup
        import app.vector as vector_mod
        monkeypatch.setattr(vector_mod, "create_vector_store",
                            lambda **kw: FakeVecStore([1]))  # 缺 2
        with pytest.raises(RuntimeError, match="写入校验失败"):
            writer._validate_build([_chunk(1), _chunk(2)], "recursive", "default")

    def test_disabled_skips_validation(self, setup):
        writer, monkeypatch, tmp_path, holder = setup
        monkeypatch.setattr(Config, "write_validation_enabled", False)
        from app.storage import metadata_store as ms_mod
        monkeypatch.setattr(ms_mod, "MetadataStore",
                            lambda *a, **kw: FakeMetaStore([]))  # 全缺
        writer._validate_build([_chunk(1), _chunk(2)], "recursive", "default")
        # 开关关闭：不校验不抛
