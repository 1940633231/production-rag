r"""文档级 ACL API 测试（阶段 5）：授权/撤销/可见性/删除权限。

使用真实 MySQL（本环境可用）：直接向 documents 插入带 owner 的文档，
再通过 API 验证 grant/revoke/list、documents 列表可见性、删除权限门禁。

前置条件: MySQL 可用；测试数据 _pytest_aclapi_* 用后清理。

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_document_acl.py -v
"""
import uuid

import pytest

from app.auth.rbac import ALL_PERMISSIONS
from app.storage.document_repository import DocumentRepository
from app.storage.mysql import MySQLManager

_ACL_PERMS = [
    "chat:query", "knowledge:read", "knowledge:upload",
    "knowledge:delete", "knowledge:rebuild", "knowledge:grant",
]


@pytest.fixture
def mysql_manager():
    mgr = MySQLManager()
    try:
        mgr.init_schema()
    except Exception:
        pytest.skip("MySQL 不可用，跳过 ACL API 测试")
    return mgr


@pytest.fixture
def seeded_docs(mysql_manager):
    """插入两个带 owner 的测试文档（default 租户），返回 (doc_id, owner_user_id, other_doc_id)。"""
    doc_id = "_pytest_aclapi_{}".format(uuid.uuid4().hex[:8])
    other_id = "_pytest_aclapi_{}".format(uuid.uuid4().hex[:8])
    repo = DocumentRepository(mysql_manager)
    repo.insert(doc_id, "owner_doc.txt", 100, tenant_id="default", owner_user_id="u-owner")
    repo.insert(other_id, "other_doc.txt", 100, tenant_id="default", owner_user_id="u-other")
    yield doc_id, "u-owner", other_id
    try:
        with mysql_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_acl WHERE document_id LIKE '_pytest_aclapi_%'")
                cur.execute("DELETE FROM documents WHERE document_id LIKE '_pytest_aclapi_%'")
    except Exception:
        pass


def _headers(make_token, user_id, perms=_ACL_PERMS, roles=("admin",)):
    token = make_token(roles=list(roles), permissions=perms, user_id=user_id)
    return {"Authorization": "Bearer {}".format(token)}


class TestGrantRevoke:
    def test_grant_requires_knowledge_grant_permission(self, anon_client, make_token, seeded_docs):
        doc_id, _, _ = seeded_docs
        # viewer 无 knowledge:grant → 403（权限点门禁）
        headers = _headers(make_token, "u-viewer", perms=["chat:query", "knowledge:read"], roles=("viewer",))
        resp = anon_client.post(
            "/api/knowledge/{}/acl".format(doc_id), headers=headers,
            json={"principal_type": "user", "principal_id": "u-bob", "permission": "read"},
        )
        assert resp.status_code == 403

    def test_grant_requires_owner(self, anon_client, make_token, seeded_docs):
        doc_id, _, _ = seeded_docs
        # u-other 有 knowledge:grant 但不是 doc 归属人 → 403
        headers = _headers(make_token, "u-other")
        resp = anon_client.post(
            "/api/knowledge/{}/acl".format(doc_id), headers=headers,
            json={"principal_type": "user", "principal_id": "u-bob", "permission": "read"},
        )
        assert resp.status_code == 403

    def test_owner_can_grant_and_list(self, anon_client, make_token, seeded_docs):
        doc_id, owner, _ = seeded_docs
        headers = _headers(make_token, owner)
        resp = anon_client.post(
            "/api/knowledge/{}/acl".format(doc_id), headers=headers,
            json={"principal_type": "user", "principal_id": "u-bob", "permission": "read"},
        )
        assert resp.status_code == 200
        assert resp.json()["granted"]["principal_id"] == "u-bob"

        # 列表能看到该授权
        resp = anon_client.get("/api/knowledge/{}/acl".format(doc_id), headers=headers)
        assert resp.status_code == 200
        grants = resp.json()["grants"]
        assert any(g["principal_id"] == "u-bob" and g["permission"] == "read" for g in grants)

    def test_revoke(self, anon_client, make_token, seeded_docs):
        doc_id, owner, _ = seeded_docs
        headers = _headers(make_token, owner)
        anon_client.post(
            "/api/knowledge/{}/acl".format(doc_id), headers=headers,
            json={"principal_type": "user", "principal_id": "u-bob", "permission": "read"},
        )
        resp = anon_client.delete(
            "/api/knowledge/{}/acl".format(doc_id), headers=headers,
            params={"principal_type": "user", "principal_id": "u-bob"},
        )
        assert resp.status_code == 200
        assert resp.json()["removed"] == 1
        grants = anon_client.get("/api/knowledge/{}/acl".format(doc_id), headers=headers).json()["grants"]
        assert all(g["principal_id"] != "u-bob" for g in grants)


class TestVisibility:
    def test_documents_list_filtered_by_acl(self, anon_client, make_token, seeded_docs):
        doc_id, owner, other_id = seeded_docs
        # 未授权用户看不到他人文档
        stranger_headers = _headers(make_token, "u-stranger")
        resp = anon_client.get("/api/knowledge/documents", headers=stranger_headers)
        assert resp.status_code == 200
        names = [d["document_id"] for d in resp.json()["documents"]]
        assert doc_id not in names
        assert other_id not in names

        # 归属人能看到自己的文档
        owner_headers = _headers(make_token, owner)
        resp = anon_client.get("/api/knowledge/documents", headers=owner_headers)
        names = [d["document_id"] for d in resp.json()["documents"]]
        assert doc_id in names
        assert other_id not in names  # 不是自己的文档

    def test_granted_user_can_see_doc(self, anon_client, make_token, seeded_docs):
        doc_id, owner, _ = seeded_docs
        # owner 授予 u-bob 读权限
        anon_client.post(
            "/api/knowledge/{}/acl".format(doc_id),
            headers=_headers(make_token, owner),
            json={"principal_type": "user", "principal_id": "u-bob", "permission": "read"},
        )
        # u-bob 现在能看到该文档
        bob_headers = _headers(make_token, "u-bob")
        resp = anon_client.get("/api/knowledge/documents", headers=bob_headers)
        names = [d["document_id"] for d in resp.json()["documents"]]
        assert doc_id in names


class TestDeletePermission:
    def test_non_owner_without_grant_cannot_delete(self, anon_client, make_token, seeded_docs):
        doc_id, _, _ = seeded_docs
        stranger_headers = _headers(make_token, "u-stranger")
        resp = anon_client.delete("/api/knowledge/{}".format(doc_id), headers=stranger_headers)
        assert resp.status_code == 403
        assert "无权删除" in resp.json()["detail"]

    def test_owner_can_delete(self, anon_client, make_token, seeded_docs):
        doc_id, owner, _ = seeded_docs
        owner_headers = _headers(make_token, owner)
        resp = anon_client.delete("/api/knowledge/{}".format(doc_id), headers=owner_headers)
        assert resp.status_code == 200
