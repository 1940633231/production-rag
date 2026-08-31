r"""文档级 ACL 仓储测试（真实 DB）。

验证 ACLRepository 的 grant/revoke/list_grants / has_permission /
get_readable_document_ids 核心逻辑。

前置条件: MySQL 可用（同 tests/storage/test_mysql_crud.py），不可用自动 skip。

运行:
  .venv\Scripts\python.exe -m pytest tests\storage\test_acl.py -v
"""
import os
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.env import load_env
load_env()

from app.acl.repository import ACLRepository
from app.auth.dependencies import AuthUser
from app.storage.mysql import MySQLManager, _MYSQL_AVAILABLE
from app.storage.document_repository import DocumentRepository


@pytest.fixture(scope="module")
def manager():
    """创建 MySQLManager 并初始化表结构；MySQL 不可用则跳过。"""
    if not _MYSQL_AVAILABLE:
        pytest.skip("pymysql/dbutils 未安装")
    mgr = MySQLManager()
    try:
        mgr.init_schema()
    except Exception as e:
        pytest.skip("MySQL 连接失败: {}。请确认 MySQL 服务运行中且环境变量已配置。".format(e))
    return mgr


@pytest.fixture(autouse=True)
def cleanup(manager):
    """测试后清理 ACL / 文档数据。"""
    yield
    try:
        with manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_acl WHERE document_id LIKE '_pytest_acl_%'")
                cur.execute("DELETE FROM documents WHERE document_id LIKE '_pytest_acl_%'")
    except Exception:
        pass


def _user(user_id="u-1", roles=("editor",), is_superadmin=False):
    return AuthUser(
        user_id=user_id, username=user_id,
        roles=list(roles), permissions=set(), is_superadmin=is_superadmin,
    )


class TestACLRepository:
    def test_grant_revoke_list(self, manager):
        acl = ACLRepository(manager)
        doc = "_pytest_acl_gr_{}".format(uuid.uuid4().hex[:8])
        acl.grant(doc, "user", "u-alice", "read")
        acl.grant(doc, "user", "u-alice", "write")
        acl.grant(doc, "role", "viewer", "read")

        grants = acl.list_grants(doc)
        assert len(grants) == 3
        perms = {(g["principal_type"], g["principal_id"], g["permission"]) for g in grants}
        assert ("user", "u-alice", "read") in perms
        assert ("user", "u-alice", "write") in perms
        assert ("role", "viewer", "read") in perms

        # 部分撤销
        removed = acl.revoke(doc, principal_type="role")
        assert removed == 1
        assert len(acl.list_grants(doc)) == 2
        # 全部撤销
        removed_all = acl.revoke(doc)
        assert removed_all == 2
        assert acl.list_grants(doc) == []

    def test_has_permission_owner(self, manager):
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc = "_pytest_acl_own_{}".format(uuid.uuid4().hex[:8])
        doc_repo.insert(doc, "acl.txt", 10, tenant_id="acme", owner_user_id="u-owner")

        owner = _user("u-owner")
        assert acl.has_permission(owner, doc, "read", "acme") is True
        assert acl.has_permission(owner, doc, "write", "acme") is True
        assert acl.has_permission(owner, doc, "delete", "acme") is True

        stranger = _user("u-stranger")
        assert acl.has_permission(stranger, doc, "read", "acme") is False
        assert acl.has_permission(stranger, doc, "delete", "acme") is False

    def test_has_permission_grants(self, manager):
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc = "_pytest_acl_grt_{}".format(uuid.uuid4().hex[:8])
        doc_repo.insert(doc, "acl.txt", 10, tenant_id="acme", owner_user_id="u-owner")

        acl.grant(doc, "user", "u-bob", "read")
        acl.grant(doc, "role", "editor", "delete")

        bob = _user("u-bob", roles=("viewer",))
        assert acl.has_permission(bob, doc, "read", "acme") is True
        assert acl.has_permission(bob, doc, "delete", "acme") is False

        editor = _user("u-editor", roles=("editor",))
        assert acl.has_permission(editor, doc, "delete", "acme") is True
        assert acl.has_permission(editor, doc, "read", "acme") is False  # 只授权了 delete

    def test_legacy_doc_read_only(self, manager):
        """存量文档（owner 为空）租户内共享，但仅可读不可删。"""
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc = "_pytest_acl_leg_{}".format(uuid.uuid4().hex[:8])
        doc_repo.insert(doc, "legacy.txt", 10, tenant_id="acme", owner_user_id="")

        user = _user("u-any")
        assert acl.has_permission(user, doc, "read", "acme") is True
        assert acl.has_permission(user, doc, "delete", "acme") is False

    def test_superadmin_always_allowed(self, manager):
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc = "_pytest_acl_sa_{}".format(uuid.uuid4().hex[:8])
        doc_repo.insert(doc, "acl.txt", 10, tenant_id="acme", owner_user_id="u-owner")

        sa = _user("u-sa", is_superadmin=True)
        assert acl.has_permission(sa, doc, "delete", "acme") is True

    def test_get_readable_document_ids(self, manager):
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        own = "_pytest_acl_ro_{}".format(uuid.uuid4().hex[:8])
        granted = "_pytest_acl_rg_{}".format(uuid.uuid4().hex[:8])
        legacy = "_pytest_acl_rl_{}".format(uuid.uuid4().hex[:8])
        other_tenant = "_pytest_acl_rt_{}".format(uuid.uuid4().hex[:8])
        for d, owner, tenant in (
            (own, "u-reader", "acme"),
            (granted, "u-owner2", "acme"),
            (legacy, "", "acme"),
            (other_tenant, "u-owner2", "beta"),
        ):
            doc_repo.insert(d, "x.txt", 10, tenant_id=tenant, owner_user_id=owner)
        acl.grant(granted, "user", "u-reader", "read")

        reader = _user("u-reader")
        readable = acl.get_readable_document_ids(reader, "acme")
        assert own in readable
        assert granted in readable
        assert legacy in readable
        assert other_tenant not in readable  # 跨租户不可读

        # superadmin → None（不设文档级过滤）
        assert acl.get_readable_document_ids(_user("u-sa", is_superadmin=True), "acme") is None
