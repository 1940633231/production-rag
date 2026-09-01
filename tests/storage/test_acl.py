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
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc = "_pytest_acl_gr_{}".format(uuid.uuid4().hex[:8])
        # 外键约束下授权必须先存在文档记录
        doc_repo.insert(doc, "gr.txt", 10, tenant_id="default", owner_user_id="u-alice")
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

    def test_delete_by_document_removes_all_grants(self, manager):
        """文档删除时 delete_by_document 必须清空该文档全部授权（防孤儿 ACL）。"""
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc = "_pytest_acl_del_{}".format(uuid.uuid4().hex[:8])
        other = "_pytest_acl_del_{}".format(uuid.uuid4().hex[:8])
        # 外键约束下授权必须先存在文档记录
        doc_repo.insert(doc, "del.txt", 10, tenant_id="default", owner_user_id="u-alice")
        doc_repo.insert(other, "other.txt", 10, tenant_id="default", owner_user_id="u-bob")
        acl.grant(doc, "user", "u-alice", "read")
        acl.grant(doc, "user", "u-alice", "write")
        acl.grant(doc, "role", "viewer", "read")
        assert len(acl.list_grants(doc)) == 3

        removed = acl.delete_by_document(doc)
        assert removed == 3
        assert acl.list_grants(doc) == []
        # 不影响其他文档的授权
        acl.grant(other, "user", "u-bob", "read")
        assert len(acl.list_grants(other)) == 1

    def test_delete_by_user_removes_user_grants(self, manager):
        """用户删除时 delete_by_user 必须清理该用户的全部授权，且不影响角色授权。"""
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc1 = "_pytest_acl_u1_{}".format(uuid.uuid4().hex[:8])
        doc2 = "_pytest_acl_u2_{}".format(uuid.uuid4().hex[:8])
        doc_repo.insert(doc1, "u1.txt", 10, tenant_id="default", owner_user_id="u-owner")
        doc_repo.insert(doc2, "u2.txt", 10, tenant_id="default", owner_user_id="u-owner")
        # 用户 u-bob 在多个文档上的授权 + 一个角色授权
        acl.grant(doc1, "user", "u-bob", "read")
        acl.grant(doc2, "user", "u-bob", "read")
        acl.grant(doc1, "role", "editor", "read")
        assert len(acl.list_grants(doc1)) == 2

        removed = acl.delete_by_user("u-bob")
        assert removed == 2  # 只清 u-bob 的 user 授权
        assert acl.list_grants(doc1) == [g for g in acl.list_grants(doc1)
                                         if g["principal_type"] != "user"]
        # 角色授权仍在（principal_id 多态，不能被用户 FK 误删）
        remaining = {g["principal_type"] for g in acl.list_grants(doc1)}
        assert remaining == {"role"}

    def test_delete_by_role_removes_role_grants(self, manager):
        """角色删除时 delete_by_role 必须清理该角色的全部授权，且不影响用户授权。"""
        doc_repo = DocumentRepository(manager)
        acl = ACLRepository(manager)
        doc1 = "_pytest_acl_r1_{}".format(uuid.uuid4().hex[:8])
        doc2 = "_pytest_acl_r2_{}".format(uuid.uuid4().hex[:8])
        doc_repo.insert(doc1, "r1.txt", 10, tenant_id="default", owner_user_id="u-owner")
        doc_repo.insert(doc2, "r2.txt", 10, tenant_id="default", owner_user_id="u-owner")
        # 角色 editor 在多个文档上的授权 + 一个用户授权
        acl.grant(doc1, "role", "editor", "read")
        acl.grant(doc2, "role", "editor", "read")
        acl.grant(doc1, "user", "u-bob", "read")
        assert len(acl.list_grants(doc1)) == 2

        removed = acl.delete_by_role("editor")
        assert removed == 2  # 只清 editor 的 role 授权
        # 用户授权仍在
        remaining = {g["principal_type"] for g in acl.list_grants(doc1)}
        assert remaining == {"user"}

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
