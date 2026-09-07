"""安全加固回归测试：P0 fail-closed 与 P1-1 孤儿可观测。

覆盖:
  - P0-1 chat：ACL 查询异常 → fail-closed 返回空集（不放行任何文档）
  - P0-2 knowledge：_can_delete 异常 → 拒绝(False)；_readable_document_ids 异常 → 空集
  - P1-1 writer：remove_document 向量移除失败 → 可观测（默认抛异常 / raise_on_orphan=False 返回失败列表）
  - P1-1 writer：remove_document 成功后无孤儿

运行:
  .venv\\Scripts\\python.exe -m pytest tests/api/test_security_hardening.py -v
"""
from types import SimpleNamespace

import pytest


def _user(is_superadmin=False, user_id="u1", tenant_id="t1", roles=()):
    return SimpleNamespace(
        is_superadmin=is_superadmin, user_id=user_id,
        tenant_id=tenant_id, roles=list(roles),
    )


# ---------------- P0-1：chat ACL fail-closed ----------------

def test_chat_acl_readable_fail_closed_returns_empty_set(monkeypatch):
    """非 superadmin 用户，ACL 查询抛异常 → 返回空集（绝不放行）。"""
    import app.api.chat as chat

    class _Boom:
        def get_readable_document_ids(self, *a, **k):
            raise RuntimeError("ACL DB down")

    monkeypatch.setattr("app.acl.repository.ACLRepository", _Boom)
    ids = chat._acl_readable_document_ids(_user(), "t1")
    assert ids == set()


def test_chat_acl_readable_returned_set_passthrough(monkeypatch):
    """ACL 查询正常 → 返回对应的可读文档集。"""
    import app.api.chat as chat

    class _Ok:
        def get_readable_document_ids(self, *a, **k):
            return {"d1", "d2"}

    monkeypatch.setattr("app.acl.repository.ACLRepository", _Ok)
    ids = chat._acl_readable_document_ids(_user(), "t1")
    assert ids == {"d1", "d2"}


def test_chat_acl_readable_superadmin_no_filter(monkeypatch):
    """superadmin 不设文档级过滤（返回 None）。"""
    import app.api.chat as chat

    class _NeverCalled:
        def get_readable_document_ids(self, *a, **k):
            raise AssertionError("superadmin 不应触发 ACL 查询")

    monkeypatch.setattr("app.acl.repository.ACLRepository", _NeverCalled)
    assert chat._acl_readable_document_ids(_user(is_superadmin=True), "t1") is None


# ---------------- P0-2：knowledge fail-closed ----------------

def test_knowledge_can_delete_fail_closed_deny(monkeypatch):
    """删除权限判定异常 → 拒绝（放行删除=越权风险，必须 deny）。"""
    import app.api.knowledge as knowledge

    class _Boom:
        def has_permission(self, *a, **k):
            raise RuntimeError("ACL down")

    monkeypatch.setattr("app.acl.repository.ACLRepository", _Boom)
    assert knowledge._can_delete(_user(), "t1", "d1") is False


def test_knowledge_can_delete_superadmin_allowed(monkeypatch):
    """superadmin 删除判定直接放行（不查 ACL）。"""
    import app.api.knowledge as knowledge

    class _NeverCalled:
        def has_permission(self, *a, **k):
            raise AssertionError("superadmin 不应触发 ACL 查询")

    monkeypatch.setattr("app.acl.repository.ACLRepository", _NeverCalled)
    assert knowledge._can_delete(_user(is_superadmin=True), "t1", "d1") is True


def test_knowledge_readable_fail_closed_returns_empty_set(monkeypatch):
    """可读文档查询异常 → 返回空集（不放行任何文档）。"""
    import app.api.knowledge as knowledge

    class _Boom:
        def get_readable_document_ids(self, *a, **k):
            raise RuntimeError("ACL down")

    monkeypatch.setattr("app.acl.repository.ACLRepository", _Boom)
    assert knowledge._readable_document_ids(_user(), "t1") == set()


def test_knowledge_readable_superadmin_no_filter(monkeypatch):
    """superadmin 可读文档不设限（返回 None）。"""
    import app.api.knowledge as knowledge

    class _NeverCalled:
        def get_readable_document_ids(self, *a, **k):
            raise AssertionError("superadmin 不应触发 ACL 查询")

    monkeypatch.setattr("app.acl.repository.ACLRepository", _NeverCalled)
    assert knowledge._readable_document_ids(_user(is_superadmin=True), "t1") is None


# ---------------- P1-1：remove_document 孤儿可观测 ----------------

class _FakeCfg:
    storage_mysql_enabled = False
    storage_es_enabled = False
    storage_milvus_enabled = False

    def __init__(self, index_dir):
        self._dir = index_dir

    def index_dir_for(self, strategy, tenant_id):
        return self._dir


@pytest.fixture
def writer(tmp_path):
    from app.ingestion.writer import IndexWriter
    return IndexWriter(config=_FakeCfg(tmp_path))


def test_remove_document_orphan_raises_by_default(writer, monkeypatch):
    """向量移除失败（孤儿）默认抛异常，让调用方可观测。"""
    monkeypatch.setattr(
        writer, "_remove_vectors_verified",
        lambda vids, strategy, tenant_id: ["FAISS 移除后残留向量"],
    )
    with pytest.raises(RuntimeError):
        writer.remove_document(
            "recursive", "t1", "d1", [1, 2, 3], raise_on_orphan=True,
        )


def test_remove_document_orphan_returns_failures_when_disabled(writer, monkeypatch):
    """raise_on_orphan=False 时返回失败列表，不抛异常。"""
    monkeypatch.setattr(
        writer, "_remove_vectors_verified",
        lambda vids, strategy, tenant_id: ["sim vector failure"],
    )
    failures = writer.remove_document(
        "recursive", "t1", "d1", [1, 2, 3], raise_on_orphan=False,
    )
    assert failures == ["sim vector failure"]


def test_remove_document_no_orphan_returns_empty(writer, monkeypatch):
    """向量移除校验通过 → 无孤儿，返回空失败列表。"""
    monkeypatch.setattr(
        writer, "_remove_vectors_verified", lambda vids, strategy, tenant_id: [],
    )
    failures = writer.remove_document(
        "recursive", "t1", "d1", [1, 2, 3], raise_on_orphan=False,
    )
    assert failures == []