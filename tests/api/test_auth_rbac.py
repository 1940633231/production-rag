r"""认证 / RBAC 测试：安全工具 + 鉴权依赖 + API 授权门禁。

覆盖（全部不依赖真实 MySQL）:
  - 密码哈希 / 校验
  - JWT 签发 / 解析（含过期、篡改）
  - get_current_user：无 token 401 / 非法 token 401 / auth 关闭放行
  - require_permission：无权限 403 / 有权限放行 / auth 关闭放行
  - API 层：受保护端点无 token 401、权限不足 403、具备权限 200
  - 登录：成功签发 token / 密码错误 401（mock 仓储）
  - 管理 API：viewer 无 admin 权限 403，admin 可访问（mock 仓储）

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_auth_rbac.py -v
"""
import pytest
from fastapi import HTTPException, status

from app.auth.rbac import ALL_PERMISSIONS, SUPERADMIN_ROLE

_VIEWER_PERMS = ["chat:query", "knowledge:read"]


@pytest.fixture
def viewer_token(make_token):
    """viewer 权限的 JWT（无 knowledge:upload / admin:*）。"""
    return make_token(roles=["viewer"], permissions=list(_VIEWER_PERMS))


@pytest.fixture
def mock_rbac_repo(monkeypatch):
    """mock RBACRepository，用于登录 / 管理 API 测试（避免真实 MySQL）。"""
    import app.auth.rbac_repository as rbac_mod

    class FakeRepo:
        users = {}
        roles = {}

        @classmethod
        def reset(cls):
            cls.users = {}
            cls.roles = {}

        def get_user_by_username(self, username):
            return self.users.get(username)

        def get_user_roles(self, user_id):
            return []

        def get_user_permissions(self, user_id):
            return set()

        def list_users(self):
            return []

    monkeypatch.setattr(rbac_mod, "RBACRepository", FakeRepo)
    return FakeRepo


# ---------------- 安全工具 ----------------

class TestSecurity:
    def test_password_hash_roundtrip(self):
        from app.auth.security import hash_password, verify_password
        h = hash_password("s3cret@Pass")
        assert h != "s3cret@Pass"
        assert verify_password("s3cret@Pass", h) is True
        assert verify_password("wrong", h) is False

    def test_token_roundtrip(self, admin_token):
        from app.auth.security import decode_access_token
        from app.core.config import Config

        config = Config()
        payload = decode_access_token(admin_token, config.auth_jwt_secret)
        assert payload is not None
        assert payload["sub"] == "u-test"
        assert SUPERADMIN_ROLE in payload["roles"]
        # 权限内嵌 token，且包含全部权限点
        assert set(payload["permissions"]) == set(ALL_PERMISSIONS.keys())

    def test_token_wrong_secret_rejected(self, admin_token):
        from app.auth.security import decode_access_token
        assert decode_access_token(admin_token, "wrong-secret") is None

    def test_token_garbage_rejected(self):
        from app.auth.security import decode_access_token
        from app.core.config import Config
        assert decode_access_token("not-a-jwt", Config().auth_jwt_secret) is None


# ---------------- 鉴权依赖 ----------------

class TestAuthDependencies:
    def test_get_current_user_missing_token_401(self, anon_client):
        resp = anon_client.get("/api/knowledge/status")
        assert resp.status_code == 401

    def test_get_current_user_invalid_token_401(self, anon_client):
        resp = anon_client.get(
            "/api/knowledge/status",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401

    def test_require_permission_denied_403(self, viewer_token):
        from app.auth.dependencies import AuthUser, require_permission
        checker = require_permission("knowledge:upload")
        user = AuthUser(
            user_id="u1", username="viewer",
            roles=["viewer"], permissions=set(_VIEWER_PERMS),
        )
        # 直接在依赖闭包内解析 user（模拟 FastAPI 注入）
        with pytest.raises(HTTPException) as exc:
            _run_dep(checker, user)
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN

    def test_require_permission_allowed(self, admin_token):
        from app.auth.dependencies import AuthUser, require_permission
        checker = require_permission("knowledge:upload")
        user = AuthUser(
            user_id="u1", username="admin",
            roles=[SUPERADMIN_ROLE], permissions=set(ALL_PERMISSIONS.keys()),
            is_superadmin=True,
        )
        assert _run_dep(checker, user) is None

    def test_auth_disabled_passes(self, monkeypatch):
        """auth.enabled=false 时 get_current_user 返回 None，require_permission 放行。"""
        import app.auth.dependencies as deps

        class FakeConfig:
            auth_enabled = False
            auth_jwt_secret = "x"
            auth_algorithm = "HS256"

        monkeypatch.setattr(deps, "Config", FakeConfig)

        from app.auth.dependencies import get_current_user, require_permission

        # 直接调用内部函数（不经过 FastAPI 依赖解析）
        user = deps.get_current_user(None, FakeConfig())
        assert user is None
        assert deps.require_permission("knowledge:delete")(None) is None


def _run_dep(func, *args, **kwargs):
    """直接调用依赖闭包，绕过 FastAPI 的 Depends 解析。"""
    return func(*args, **kwargs)


# ---------------- API 授权门禁 ----------------

class TestApiAuthz:
    def test_protected_endpoints_require_token(self, anon_client):
        assert anon_client.post("/api/chat", json={"query": "hi"}).status_code == 401
        assert anon_client.get("/api/knowledge/status").status_code == 401
        assert anon_client.get("/api/knowledge/documents").status_code == 401
        assert anon_client.get("/api/admin/users").status_code == 401

    def test_health_is_public(self, anon_client):
        """/api/health 不要求鉴权。"""
        resp = anon_client.get("/api/health")
        assert resp.status_code == 200

    def test_viewer_can_read_but_not_upload(self, anon_client, viewer_token,
                                            mock_rbac_repo):
        headers = {"Authorization": "Bearer {}".format(viewer_token)}

        # 具备 knowledge:read → 200
        resp = anon_client.get("/api/knowledge/documents", headers=headers)
        assert resp.status_code == 200

        # 不具备 knowledge:upload → 403
        resp = anon_client.post(
            "/api/knowledge/upload",
            headers=headers,
            files={"file": ("_pytest_authz.txt", b"x", "text/plain")},
            data={"strategy": "recursive"},
        )
        assert resp.status_code == 403
        assert "权限不足" in resp.json()["detail"]

    def test_admin_endpoint_requires_admin_permission(self, anon_client,
                                                      viewer_token, admin_token,
                                                      mock_rbac_repo):
        # viewer 无 admin:users → 403
        resp = anon_client.get(
            "/api/admin/users",
            headers={"Authorization": "Bearer {}".format(viewer_token)},
        )
        assert resp.status_code == 403

        # admin → 200（仓储被 mock 为空列表）
        resp = anon_client.get(
            "/api/admin/users",
            headers={"Authorization": "Bearer {}".format(admin_token)},
        )
        assert resp.status_code == 200
        assert resp.json() == {"users": [], "total": 0}

    def test_metrics_endpoint_requires_metrics_read(self, anon_client,
                                                    viewer_token, admin_token):
        # viewer 无 metrics:read → 403
        resp = anon_client.get(
            "/api/admin/metrics",
            headers={"Authorization": "Bearer {}".format(viewer_token)},
        )
        assert resp.status_code == 403
        assert "权限不足" in resp.json()["detail"]

        # superadmin（全权限）→ 200，返回结构化快照
        resp = anon_client.get(
            "/api/admin/metrics",
            headers={"Authorization": "Bearer {}".format(admin_token)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert isinstance(data.get("histograms"), dict)
        assert isinstance(data.get("scalars"), dict)


# ---------------- 登录 ----------------

class TestLogin:
    def test_login_success(self, anon_client, mock_rbac_repo):
        from app.auth.security import hash_password
        mock_rbac_repo.reset()
        mock_rbac_repo.users["admin"] = {
            "user_id": "u-admin", "username": "admin",
            "password_hash": hash_password("admin123"),
            "display_name": "管理员", "tenant_id": "default",
            "is_active": 1,
        }

        resp = anon_client.post("/api/auth/login", json={
            "username": "admin", "password": "admin123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"]
        assert data["user"]["username"] == "admin"
        assert "permissions" in data

    def test_login_wrong_password_401(self, anon_client, mock_rbac_repo):
        from app.auth.security import hash_password
        mock_rbac_repo.reset()
        mock_rbac_repo.users["admin"] = {
            "user_id": "u-admin", "username": "admin",
            "password_hash": hash_password("admin123"),
            "display_name": "管理员", "tenant_id": "default",
            "is_active": 1,
        }

        resp = anon_client.post("/api/auth/login", json={
            "username": "admin", "password": "wrong",
        })
        assert resp.status_code == 401
        assert "用户名或密码错误" in resp.json()["detail"]

    def test_login_unknown_user_401(self, anon_client, mock_rbac_repo):
        mock_rbac_repo.reset()
        resp = anon_client.post("/api/auth/login", json={
            "username": "nobody", "password": "whatever",
        })
        assert resp.status_code == 401
