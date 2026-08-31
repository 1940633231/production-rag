r"""审计日志测试（阶段 4）。

覆盖:
  - 中间件：401 / 403 越权拒绝自动记录（authz.denied）
  - 登录：成功/失败记录（login.success / login.failure）
  - 管理操作：用户创建、角色操作记录（user.create 等）
  - 文档操作：上传/删除记录（document.upload / document.delete）
  - 审计查询 API：admin:audit 权限门禁 + 返回日志

审计写入在测试中由 conftest 注入 InMemoryAuditLogger（不落真实 MySQL）。

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_audit_log.py -v
"""
import pytest

from app.auth.rbac import ALL_PERMISSIONS


@pytest.fixture
def fake_rbac_repo(monkeypatch):
    """mock RBACRepository，供登录 / 用户管理测试使用（避免真实 MySQL）。"""
    import app.auth.rbac_repository as rbac_mod

    class FakeRepo:
        users = {}
        roles = {}

        def __init__(self, manager=None):
            pass

        def get_user_by_username(self, username):
            return self.users.get(username)

        def get_user_by_id(self, user_id):
            for u in self.users.values():
                if u.get("user_id") == user_id:
                    return u
            return None

        def create_user(self, **kw):
            self.users[kw["username"]] = kw

        def set_user_roles(self, user_id, roles):
            pass

        def get_user_roles(self, user_id):
            return ["admin"]

        def get_user_permissions(self, user_id):
            return set(ALL_PERMISSIONS.keys())

        def update_user(self, *a, **kw):
            return 1

        def delete_user(self, user_id):
            return 1

        def get_role_by_code(self, role_code):
            return self.roles.get(role_code)

        def create_role(self, role_code, name, permission_codes=None, builtin=False):
            self.roles[role_code] = {"role_code": role_code, "name": name}

    monkeypatch.setattr(rbac_mod, "RBACRepository", FakeRepo)
    return FakeRepo


@pytest.fixture
def admin_headers(make_token):
    token = make_token(
        roles=["superadmin"], permissions=list(ALL_PERMISSIONS.keys()),
    )
    return {"Authorization": "Bearer {}".format(token)}


def _seed_admin(fake_rbac_repo):
    from app.auth.security import hash_password
    fake_rbac_repo.users["admin"] = {
        "user_id": "u-admin", "username": "admin",
        "password_hash": hash_password("admin123"),
        "display_name": "管理员", "tenant_id": "default", "is_active": 1,
    }
    return fake_rbac_repo.users["admin"]


# ---------------- 中间件：越权拒绝 ----------------

class TestMiddlewareAuthz:
    def test_401_recorded(self, anon_client, audit_log):
        resp = anon_client.get("/api/knowledge/status")
        assert resp.status_code == 401
        events = audit_log.query(action="authz.denied")
        assert len(events) >= 1
        assert any(e["resource"] == "/api/knowledge/status" for e in events)
        assert events[0]["result"] == "denied"

    def test_403_recorded_with_actor(self, anon_client, make_token, audit_log):
        # viewer 只有 chat:query，无 knowledge:upload → 403
        token = make_token(roles=["viewer"], permissions=["chat:query"])
        resp = anon_client.post(
            "/api/knowledge/upload",
            headers={"Authorization": "Bearer {}".format(token)},
            files={"file": ("_pytest_audit_x.txt", b"x", "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 403
        events = audit_log.query(action="authz.denied")
        assert len(events) >= 1
        # 中间件从 token 中解析出 actor
        assert events[-1]["actor_username"] == "test"


# ---------------- 登录 ----------------

class TestLoginAudit:
    def test_login_success(self, anon_client, fake_rbac_repo, audit_log):
        _seed_admin(fake_rbac_repo)
        resp = anon_client.post("/api/auth/login", json={
            "username": "admin", "password": "admin123",
        })
        assert resp.status_code == 200
        events = audit_log.query(action="login.success")
        assert len(events) == 1
        assert events[0]["actor_username"] == "admin"
        assert events[0]["result"] == "success"

    def test_login_failure(self, anon_client, fake_rbac_repo, audit_log):
        _seed_admin(fake_rbac_repo)
        resp = anon_client.post("/api/auth/login", json={
            "username": "admin", "password": "wrong",
        })
        assert resp.status_code == 401
        events = audit_log.query(action="login.failure")
        assert len(events) == 1
        assert events[0]["resource"] == "admin"
        assert events[0]["result"] == "failure"

    def test_login_unknown_user_failure(self, anon_client, fake_rbac_repo, audit_log):
        resp = anon_client.post("/api/auth/login", json={
            "username": "nobody", "password": "whatever",
        })
        assert resp.status_code == 401
        events = audit_log.query(action="login.failure")
        assert len(events) == 1
        assert events[0]["resource"] == "nobody"


# ---------------- 管理操作 ----------------

class TestAdminAudit:
    def test_user_create_recorded(self, anon_client, fake_rbac_repo, admin_headers, audit_log):
        resp = anon_client.post(
            "/api/admin/users", headers=admin_headers,
            json={"username": "newuser", "password": "secret123", "roles": ["viewer"]},
        )
        assert resp.status_code == 200
        events = audit_log.query(action="user.create")
        assert len(events) == 1
        assert events[0]["actor_username"] == "test"
        assert "newuser" in events[0]["detail"]

    def test_role_create_recorded(self, anon_client, fake_rbac_repo, admin_headers, audit_log):
        resp = anon_client.post(
            "/api/admin/roles", headers=admin_headers,
            json={"role_code": "ops", "name": "运维", "permissions": ["chat:query"]},
        )
        assert resp.status_code == 200
        events = audit_log.query(action="role.create")
        assert len(events) == 1
        assert events[0]["resource"] == "ops"


# ---------------- 文档操作 ----------------

@pytest.fixture
def mock_upload(monkeypatch):
    import app.api.knowledge as knowledge
    monkeypatch.setattr(
        knowledge, "_do_upload",
        lambda path, strategy, tenant_id="default", owner_user_id="": {
            "document_count": 1, "chunk_count": 2, "dimension": 768,
            "index_path": "data/index/recursive/faiss.index",
            "metadata_path": "data/index/recursive/metadata.json",
        },
    )
    return knowledge


class TestDocumentAudit:
    def test_upload_recorded(self, anon_client, admin_headers, audit_log, mock_upload, raw_dir):
        fname = "_pytest_audit_up_{}.txt".format(__import__("uuid").uuid4().hex[:8])
        resp = anon_client.post(
            "/api/knowledge/upload", headers=admin_headers,
            files={"file": (fname, "审计上传".encode("utf-8"), "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 200
        events = audit_log.query(action="document.upload")
        assert len(events) == 1
        assert events[0]["resource"] == fname
        (raw_dir / fname).unlink()

    def test_delete_recorded(self, anon_client, admin_headers, audit_log, raw_dir,
                             monkeypatch, sync_background_rebuild):
        import app.api.knowledge as knowledge
        import app.ingestion.writer as writer_mod
        import app.core.config as config_mod
        from app.core.config import Config as RealConfig

        class FakeConfig:
            storage_mysql_enabled = False
            storage_es_enabled = False
            storage_milvus_enabled = False
            cache_enabled = False

            @staticmethod
            def raw_dir_for(tenant_id="default"):
                return RealConfig().raw_dir_for(tenant_id)

            @staticmethod
            def index_dir_for(strategy, tenant_id="default"):
                return RealConfig().index_dir_for(strategy, tenant_id)

        monkeypatch.setattr(config_mod, "Config", FakeConfig)

        class FakeIndexWriter:
            def __init__(self, config=None):
                pass

            def incremental_rebuild_after_delete(self, strategy, deleted_doc_ids,
                                                 tenant_id="default", **kw):
                return {"document_count": 0}

        monkeypatch.setattr(writer_mod, "IndexWriter", FakeIndexWriter)

        fname = "_pytest_audit_del_{}.txt".format(__import__("uuid").uuid4().hex[:8])
        target = raw_dir / fname
        target.write_text("待删除", encoding="utf-8")
        resp = anon_client.delete(
            "/api/knowledge/{}".format(fname), headers=admin_headers,
        )
        assert resp.status_code == 200
        events = audit_log.query(action="document.delete")
        assert len(events) == 1
        assert events[0]["resource"] == fname
        assert not target.exists()


# ---------------- 审计查询 API ----------------

class TestAuditQueryApi:
    def test_returns_logs(self, anon_client, admin_headers, audit_log):
        audit_log.record(action="document.upload", tenant_id="default", actor_username="test")
        resp = anon_client.get("/api/admin/audit-logs", headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["logs"][0]["action"] == "document.upload"

    def test_filter_by_action(self, anon_client, admin_headers, audit_log):
        audit_log.record(action="document.upload", tenant_id="default")
        audit_log.record(action="document.delete", tenant_id="default")
        resp = anon_client.get(
            "/api/admin/audit-logs", headers=admin_headers,
            params={"action": "document.delete"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["logs"][0]["action"] == "document.delete"

    def test_requires_admin_audit_permission(self, anon_client, make_token):
        token = make_token(roles=["viewer"], permissions=["chat:query"])
        resp = anon_client.get(
            "/api/admin/audit-logs",
            headers={"Authorization": "Bearer {}".format(token)},
        )
        assert resp.status_code == 403
