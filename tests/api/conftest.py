r"""API 层测试共享 fixture。

提供:
  - client: 带 admin token 的 FastAPI TestClient（session 级共享，供现有业务测试复用）
  - anon_client: 不带认证头的 TestClient（测试 401 / 403）
  - admin_token / make_token: 生成指定角色的 JWT（不依赖数据库）

注意:
  - 所有测试避免真实依赖（embedding 模型下载 / MySQL / ES / Milvus）
  - 测试产生的 data/raw 文件使用 `_pytest_` 前缀，由各测试自行清理
  - 鉴权按 config.yaml 的 auth 段生效（auth.enabled=true），故默认 client 注入 admin token
"""
import pytest
from fastapi.testclient import TestClient

from app.auth.rbac import ALL_PERMISSIONS, SUPERADMIN_ROLE

# 在 conftest 导入时读取真实 auth 配置并缓存（避免运行时 monkeypatch Config 影响 token 签发）
from app.core.config import Config as _RealConfig
_AUTH_CONFIG = _RealConfig()


def _make_token(roles, permissions, user_id="u-test", username="test",
                tenant_id="default"):
    """用真实 auth 密钥签发 JWT（无状态，不依赖数据库）。"""
    from app.auth.security import create_access_token

    return create_access_token(
        user_id=user_id,
        username=username,
        display_name="测试用户",
        tenant_id=tenant_id,
        roles=roles,
        permissions=permissions,
        secret=_AUTH_CONFIG.auth_jwt_secret,
        expires_hours=_AUTH_CONFIG.auth_token_expire_hours,
        algorithm=_AUTH_CONFIG.auth_algorithm,
    )


@pytest.fixture(scope="session")
def admin_token():
    """superadmin 角色的 JWT（拥有全部权限点）。"""
    return _make_token(
        roles=[SUPERADMIN_ROLE],
        permissions=list(ALL_PERMISSIONS.keys()),
    )


@pytest.fixture
def make_token():
    """返回签发指定角色/权限/租户 JWT 的工厂函数（不依赖数据库）。"""
    def _factory(roles, permissions, user_id="u-test", username="test",
                 tenant_id="default"):
        return _make_token(
            roles, permissions, user_id=user_id, username=username,
            tenant_id=tenant_id,
        )
    return _factory


@pytest.fixture(scope="session")
def client(admin_token):
    """FastAPI TestClient（session 级，默认携带 admin token）。"""
    from app.api import app

    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer {}".format(admin_token)})
        yield c


@pytest.fixture
def anon_client():
    """不带认证头的 TestClient（测试 401 / 403 场景）。"""
    from app.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def raw_dir():
    """data/raw 目录（相对项目根）。"""
    from pathlib import Path
    d = Path("data/raw")
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def _isolate_query_cache():
    """每个测试前重置权限感知查询缓存单例，避免跨测试串缓存。"""
    from app.cache.query_cache import reset_query_cache
    reset_query_cache()
    yield
    reset_query_cache()


@pytest.fixture(autouse=True)
def _audit_in_memory():
    """每个测试用内存审计实现替换单例，避免审计写入真实 MySQL 且便于断言。"""
    from app.audit import logger as audit_mod
    from app.audit.logger import InMemoryAuditLogger

    mem = InMemoryAuditLogger()
    audit_mod._logger = mem
    yield mem
    audit_mod._logger = None


@pytest.fixture
def audit_log(_audit_in_memory):
    """返回当前测试的内存审计日志（InMemoryAuditLogger）。"""
    return _audit_in_memory


@pytest.fixture
def sync_background_rebuild(monkeypatch):
    """让 task_manager.submit 同步执行任务（删除测试用：索引重建立即跑完，断言可确定）。

    避免真实后台线程的时序不确定性；配合 IndexWriter mock 时为 no-op。
    """
    from app.core import task_queue as tq

    def _sync_submit(task_type, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return "task_{}_sync".format(task_type)

    monkeypatch.setattr(tq.task_manager, "submit", _sync_submit)
    return _sync_submit


@pytest.fixture
def noop_background_rebuild(monkeypatch):
    """让 task_manager.submit 只返回 id 不执行任务（真实 DB 删除测试用，避免触发耗时重建）。"""
    from app.core import task_queue as tq

    def _noop_submit(task_type, fn, *args, **kwargs):
        return "task_{}_noop".format(task_type)

    monkeypatch.setattr(tq.task_manager, "submit", _noop_submit)
    return _noop_submit
