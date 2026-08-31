r"""租户隔离测试（阶段 2）：路径/命名隔离 + 上传/删除/列表按租户 + 缓存按租户。

覆盖:
  - Config 路径助手：default 租户沿用旧布局，非 default 租户隔离目录/命名
  - 上传按租户落盘（data/raw/{tenant}/）
  - documents 列表按租户隔离（互不可见）
  - 删除按租户隔离（B 不能删 A 的文档 → 404）
  - service 缓存按租户隔离（permission-aware cache：同租户复用，异租户独立）

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_tenant_isolation.py -v
"""
import os
import uuid

import pytest

from app.core.config import Config


@pytest.fixture
def tenant_token(make_token):
    """生成指定租户的 superadmin token 工厂。"""
    def _factory(tenant_id, perms=None):
        from app.auth.rbac import ALL_PERMISSIONS
        return make_token(
            roles=["superadmin"],
            permissions=perms or list(ALL_PERMISSIONS.keys()),
            tenant_id=tenant_id,
        )
    return _factory


@pytest.fixture
def mock_upload(monkeypatch):
    """替换 _do_upload 为假实现，跳过真实索引构建。"""
    import app.api.knowledge as knowledge

    def fake_do_upload(save_path, strategy, tenant_id="default", owner_user_id=""):
        return {
            "document_count": 1, "chunk_count": 2, "dimension": 768,
            "index_path": "data/index/{}/recursive/faiss.index".format(tenant_id),
            "metadata_path": "data/index/{}/recursive/metadata.json".format(tenant_id),
        }

    monkeypatch.setattr(knowledge, "_do_upload", fake_do_upload)
    return knowledge


@pytest.fixture
def no_mysql(monkeypatch):
    """跳过 MySQL，路径助手委托真实 Config。"""
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
    """mock IndexWriter，记录 tenant_id 透传。"""
    import app.ingestion.writer as writer_mod

    calls = {"tenants": []}

    class FakeIndexWriter:
        def __init__(self, config=None):
            pass

        def incremental_rebuild_after_delete(self, strategy, deleted_doc_ids,
                                             tenant_id="default", **kw):
            calls["tenants"].append((strategy, tenant_id))
            return {"document_count": 0}

    monkeypatch.setattr(writer_mod, "IndexWriter", FakeIndexWriter)
    return calls


def _uniq(tag):
    return "_pytest_tnt_{}_{}.txt".format(tag, uuid.uuid4().hex[:8])


# ---------------- Config 路径/命名隔离 ----------------

class TestTenantPaths:
    def test_raw_dir_for(self):
        c = Config()
        assert str(c.raw_dir_for("default")) == os.path.join("data", "raw")
        assert str(c.raw_dir_for("acme")) == os.path.join("data", "raw", "acme")

    def test_index_dir_for(self):
        c = Config()
        assert str(c.index_dir_for("recursive", "default")) == os.path.join(
            "data", "index", "recursive"
        )
        assert str(c.index_dir_for("recursive", "acme")) == os.path.join(
            "data", "index", "acme", "recursive"
        )

    def test_milvus_collection_for(self):
        c = Config()
        prefix = c.milvus_collection_prefix
        assert c.milvus_collection_for("recursive", "default") == c.milvus_collection_name("recursive")
        assert c.milvus_collection_for("recursive", "acme") == "{}_acme_recursive".format(prefix)

    def test_es_index_name_for(self):
        c = Config()
        prefix = c.storage_backends.get("es", {}).get("index_prefix", "production_rag")
        assert c.es_index_name_for("recursive", "default") == "{}_recursive".format(prefix)
        assert c.es_index_name_for("recursive", "acme") == "{}_acme_recursive".format(prefix)


# ---------------- 上传/列表/删除按租户隔离 ----------------

class TestTenantDataIsolation:
    def test_upload_lands_in_tenant_dir(self, anon_client, tenant_token,
                                        mock_upload):
        """租户 A 上传的文件应落在 data/raw/acme/，不污染全局 data/raw/。"""
        token = tenant_token("acme")
        fname = _uniq("up")
        resp = anon_client.post(
            "/api/knowledge/upload",
            headers={"Authorization": "Bearer {}".format(token)},
            files={"file": (fname, "A 租户文档".encode("utf-8"), "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 200
        tenant_file = Config().raw_dir_for("acme") / fname
        assert tenant_file.exists()
        try:
            # 全局 data/raw/ 不应出现该文件
            assert not (Config().raw_dir_for("default") / fname).exists()
        finally:
            tenant_file.unlink()

    def test_documents_list_tenant_scoped(self, anon_client, tenant_token,
                                          mock_upload, no_mysql):
        """A 上传后：A 能看见，B 看不见（文件扫描回退路径）。"""
        token_a = tenant_token("acme")
        token_b = tenant_token("beta")
        fname = _uniq("list")
        resp = anon_client.post(
            "/api/knowledge/upload",
            headers={"Authorization": "Bearer {}".format(token_a)},
            files={"file": (fname, "A 文档".encode("utf-8"), "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 200
        try:
            # B 的 documents 不应包含 A 的文件
            resp_b = anon_client.get(
                "/api/knowledge/documents",
                headers={"Authorization": "Bearer {}".format(token_b)},
            )
            assert resp_b.status_code == 200
            names_b = [d["file_name"] for d in resp_b.json()["documents"]]
            assert fname not in names_b

            # A 的 documents 包含自己的文件
            resp_a = anon_client.get(
                "/api/knowledge/documents",
                headers={"Authorization": "Bearer {}".format(token_a)},
            )
            assert resp_a.status_code == 200
            names_a = [d["file_name"] for d in resp_a.json()["documents"]]
            assert fname in names_a
        finally:
            (Config().raw_dir_for("acme") / fname).unlink()

    def test_delete_tenant_scoped(self, anon_client, tenant_token, mock_upload,
                                  no_mysql, fake_index_writer, sync_background_rebuild):
        """A 上传后：B 删除 → 404；A 删除 → 200 且触发租户重建。"""
        token_a = tenant_token("acme")
        token_b = tenant_token("beta")
        fname = _uniq("del")
        resp = anon_client.post(
            "/api/knowledge/upload",
            headers={"Authorization": "Bearer {}".format(token_a)},
            files={"file": (fname, "A 待删文档".encode("utf-8"), "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 200
        try:
            # B 尝试删除 A 的文件：文件不在 B 的租户目录 → 404
            resp_b = anon_client.delete(
                "/api/knowledge/{}".format(fname),
                headers={"Authorization": "Bearer {}".format(token_b)},
            )
            assert resp_b.status_code == 404
            # A 的文件应仍在
            assert (Config().raw_dir_for("acme") / fname).exists()

            # A 删除自己的文件 → 200
            resp_a = anon_client.delete(
                "/api/knowledge/{}".format(fname),
                headers={"Authorization": "Bearer {}".format(token_a)},
            )
            assert resp_a.status_code == 200
            assert resp_a.json()["deleted_file"] is True
            assert not (Config().raw_dir_for("acme") / fname).exists()
            # 增量重建按 acme 租户触发
            assert all(tenant == "acme" for _, tenant in fake_index_writer["tenants"])
        finally:
            f = Config().raw_dir_for("acme") / fname
            if f.exists():
                f.unlink()


# ---------------- 缓存按租户隔离（permission-aware cache） ----------------

class TestTenantAwareCache:
    def test_service_cache_isolated_by_tenant(self, monkeypatch):
        """同租户复用同一 service，异租户使用独立 service（缓存 key 含 tenant）。"""
        import app.rag.service as svc

        created = []

        class FakeService:
            def __init__(self, config=None, strategy="recursive", mode="vector",
                         use_rerank=True, tenant_id="default"):
                self.tenant_id = tenant_id
                created.append(tenant_id)

        monkeypatch.setattr(svc, "RAGService", FakeService)
        monkeypatch.setattr(svc, "_index_version", lambda strategy, tenant_id="default": "v1")

        try:
            a1 = svc.get_service(tenant_id="acme")
            a2 = svc.get_service(tenant_id="acme")
            b1 = svc.get_service(tenant_id="beta")
            b2 = svc.get_service(tenant_id="beta")
            d1 = svc.get_service(tenant_id="default")

            # 同租户复用实例
            assert a1 is a2
            assert b1 is b2
            # 异租户独立实例
            assert a1 is not b1
            assert a1 is not d1
            # 实例携带正确 tenant
            assert a1.tenant_id == "acme"
            assert b1.tenant_id == "beta"
            assert d1.tenant_id == "default"
            # 缓存 key 含 tenant：不同租户各自只构建一次
            assert created == ["acme", "beta", "default"]
        finally:
            svc.reset_service_cache()
