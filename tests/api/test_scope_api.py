"""chat API 的 Query Scope 集成测试（mock resolver/检索，验证过滤集贯通）。

覆盖:
  - scope 启用 + 显式传 scope → service 收到"ACL ∩ scope"过滤集
  - 不同 scope → 缓存 key 不同（不串缓存）
  - stats 标注 scope_enabled / scope_used / scope_entity（提醒作用）
  - scope 未启用（默认）→ 行为不变，stats.scope_enabled=false

运行:
  .venv\\Scripts\\python.exe -m pytest tests\\api\\test_scope_api.py -v
"""
import pytest


class _FakeConfig:
    """scope 启用 + memory 缓存的最小配置。"""
    scope_enabled = True
    scope_mode = "explicit"
    scope_require_entity = False
    scope_match_top_k = 10
    cache_enabled = True
    cache_ttl_seconds = 300
    cache_max_entries = 50
    cache_backend = "memory"


class _FakeConfigDisabled(_FakeConfig):
    scope_enabled = False


@pytest.fixture
def scope_chat(monkeypatch):
    """替换 chat 的 config / service / cache / _index_version，记录 service 收到的过滤集。"""
    import app.api.chat as chat
    from app.cache.query_cache import PermissionAwareQueryCache

    state = {
        "calls": 0,
        "last_document_ids": None,
        "cache": PermissionAwareQueryCache(ttl_seconds=300, max_entries=50),
    }

    class FakeService:
        def query(self, query, history=None, document_ids=None):
            state["calls"] += 1
            state["last_document_ids"] = document_ids
            return _make_rag_response(query, "答案{}".format(state["calls"]))

        def query_stream(self, query, history=None, document_ids=None):
            state["calls"] += 1
            state["last_document_ids"] = document_ids
            yield {"type": "meta", "chunks": [], "stats": {}, "context": ""}
            yield {"type": "delta", "content": "答案{}".format(state["calls"])}
            yield {"type": "done", "answer_length": 4}

    monkeypatch.setattr(chat, "_get_config", lambda: _FakeConfig())
    monkeypatch.setattr(chat, "get_service", lambda **kw: FakeService())
    monkeypatch.setattr(chat, "get_query_cache", lambda config=None: state["cache"])
    monkeypatch.setattr(
        chat, "_index_version",
        lambda strategy, tenant_id="default": "v1",
    )
    return state


def _make_rag_response(query, answer):
    from app.rag.result import RAGResponse

    return RAGResponse(
        query=query,
        answer=answer,
        context="",
        chunks=[],
        citations=[],
        stats={"input_count": 0},
    )


def _mock_acl(monkeypatch, doc_ids):
    """替换 ACLRepository.get_readable_document_ids（chat 内为函数内 import）。"""
    import app.acl.repository as acl_mod

    monkeypatch.setattr(
        acl_mod.ACLRepository, "get_readable_document_ids",
        lambda self, user, tenant_id="default": doc_ids,
    )


def _mock_scope(monkeypatch, scope_ids, entity):
    import app.api.chat as chat

    monkeypatch.setattr(
        chat, "_resolve_scope",
        lambda config, req, tenant: (scope_ids, entity),
    )


class TestScopeApi:
    def test_scope_passes_intersection_to_service(self, anon_client, scope_chat,
                                                  monkeypatch, make_token):
        """显式 scope 过滤集与 ACL 可读集取交集后传入 service。"""
        _mock_acl(monkeypatch, {"doc-a", "doc-b", "doc-c"})
        _mock_scope(monkeypatch, {"doc-b", "doc-c", "doc-d"}, "宝钢股份")

        token = make_token(roles=["viewer"], permissions=["chat:query"])
        r = anon_client.post(
            "/api/chat",
            headers={"Authorization": "Bearer {}".format(token)},
            json={"query": "宝钢股份财报如何", "scope": "宝钢股份"},
        )
        assert r.status_code == 200
        # 交集：ACL {a,b,c} ∩ scope {b,c,d} = {b,c}
        assert scope_chat["last_document_ids"] == {"doc-b", "doc-c"}

    def test_scope_isolation_in_cache_key(self, anon_client, scope_chat,
                                          monkeypatch, make_token):
        """不同 scope → 有效过滤集不同 → 缓存 key 不同（不串缓存）。"""
        _mock_acl(monkeypatch, None)
        scopes = iter([
            ({"doc-a"}, "公司A"),
            ({"doc-b"}, "公司B"),
        ])
        import app.api.chat as chat

        monkeypatch.setattr(
            chat, "_resolve_scope",
            lambda config, req, tenant: next(scopes),
        )

        token = make_token(roles=["viewer"], permissions=["chat:query"])
        headers = {"Authorization": "Bearer {}".format(token)}
        r1 = anon_client.post("/api/chat", headers=headers,
                              json={"query": "财报如何", "scope": "公司A"})
        assert r1.status_code == 200
        assert scope_chat["calls"] == 1
        # scope=公司B → 不同过滤集 → 不命中公司A的缓存
        r2 = anon_client.post("/api/chat", headers=headers,
                              json={"query": "财报如何", "scope": "公司B"})
        assert r2.status_code == 200
        assert scope_chat["calls"] == 2

    def test_stats_annotate_scope(self, anon_client, scope_chat,
                                  monkeypatch, make_token):
        """stats 标注 scope_enabled/used/entity（提醒业务驱动 RAG）。"""
        _mock_acl(monkeypatch, None)
        _mock_scope(monkeypatch, {"doc-a"}, "宝钢股份")

        token = make_token(roles=["viewer"], permissions=["chat:query"])
        r = anon_client.post(
            "/api/chat",
            headers={"Authorization": "Bearer {}".format(token)},
            json={"query": "宝钢股份财报如何", "scope": "宝钢股份"},
        )
        assert r.status_code == 200
        stats = r.json()["stats"]
        assert stats["scope_enabled"] is True
        assert stats["scope_used"] is True
        assert stats["scope_entity"] == "宝钢股份"

    def test_scope_disabled_keeps_behavior(self, anon_client, scope_chat,
                                           monkeypatch, make_token):
        """scope 未启用（默认）：不过滤，stats.scope_enabled=false。"""
        import app.api.chat as chat

        _mock_acl(monkeypatch, None)
        monkeypatch.setattr(chat, "_get_config", lambda: _FakeConfigDisabled())

        token = make_token(roles=["viewer"], permissions=["chat:query"])
        r = anon_client.post(
            "/api/chat",
            headers={"Authorization": "Bearer {}".format(token)},
            json={"query": "财报如何"},
        )
        assert r.status_code == 200
        assert scope_chat["last_document_ids"] is None  # 无 scope → 不设过滤
        assert r.json()["stats"]["scope_enabled"] is False
