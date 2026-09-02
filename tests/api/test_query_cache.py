r"""权限感知查询缓存测试（阶段 3）。

覆盖:
  - 缓存单元：set/get / TTL 过期 / LRU 淘汰 / 统计
  - key 构造：tenant / permissions / query / 索引版本 参与哈希；权限顺序无关
  - API 集成：命中直接返回、未命中回填、TTL 过期失效、索引版本变更失效
  - 权限隔离：同租户不同权限集合不共享缓存

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_query_cache.py -v
"""
import pytest

from app.cache.query_cache import (
    PermissionAwareQueryCache,
    build_permission_fingerprint,
    build_query_cache_key,
)


def _make_rag_response(query, answer):
    from app.rag.result import RAGResponse
    return RAGResponse(
        query=query,
        context="[1] 测试上下文",
        chunks=[{"chunk_id": "c1", "content": "测试上下文"}],
        stats={"query_count": 1},
        answer=answer,
        citations=[{"number": 1, "chunk_id": "c1"}],
    )


# ---------------- 缓存单元 ----------------

class TestQueryCacheUnit:
    def test_set_get(self):
        c = PermissionAwareQueryCache(ttl_seconds=300, max_entries=10)
        c.set("k1", {"answer": "a"})
        assert c.get("k1") == {"answer": "a"}
        assert c.get("missing") is None

    def test_ttl_expiry(self):
        c = PermissionAwareQueryCache(ttl_seconds=0, max_entries=10)
        c.set("k1", "v")
        # ttl=0：立即过期
        assert c.get("k1") is None
        assert len(c) == 0

    def test_lru_eviction(self):
        c = PermissionAwareQueryCache(ttl_seconds=300, max_entries=2)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)  # 超出容量，淘汰最久未用 a
        assert c.get("a") is None
        assert c.get("b") == 2
        assert c.get("c") == 3
        assert c.stats()["evictions"] == 1

    def test_stats_counts(self):
        c = PermissionAwareQueryCache(ttl_seconds=300, max_entries=10)
        c.get("miss1")
        c.get("miss2")
        c.set("hit", "v")
        c.get("hit")
        stats = c.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["entries"] == 1

    def test_permission_fingerprint_order_insensitive(self):
        assert build_permission_fingerprint(["b", "a"]) == build_permission_fingerprint(["a", "b"])
        assert build_permission_fingerprint([]) == "none"
        assert build_permission_fingerprint(None) == "none"


# ---------------- key 构造 ----------------

class TestCacheKey:
    def _key(self, **overrides):
        params = dict(
            tenant_id="default", user_id="u-test", permissions=["chat:query"],
            query="问题", strategy="recursive", mode="vector", use_rerank=True,
            index_version="v1",
        )
        params.update(overrides)
        return build_query_cache_key(**params)

    def test_same_input_same_key(self):
        assert self._key() == self._key()

    def test_tenant_changes_key(self):
        assert self._key(tenant_id="acme") != self._key(tenant_id="beta")
        assert self._key(tenant_id="acme") != self._key()

    def test_user_id_changes_key(self):
        """文档级 ACL：不同用户不应共享缓存。"""
        assert self._key(user_id="u-1") != self._key(user_id="u-2")
        assert self._key(user_id="u-1") != self._key()

    def test_permissions_change_key(self):
        k_full = self._key(permissions=["chat:query", "knowledge:read"])
        k_min = self._key(permissions=["chat:query"])
        assert k_full != k_min

    def test_index_version_changes_key(self):
        assert self._key(index_version="v1") != self._key(index_version="v2")

    def test_query_changes_key(self):
        assert self._key(query="问题A") != self._key(query="问题B")

    def test_permission_order_insensitive(self):
        a = self._key(permissions=["chat:query", "knowledge:read"])
        b = self._key(permissions=["knowledge:read", "chat:query"])
        assert a == b


# ---------------- API 集成 ----------------

@pytest.fixture
def chat_env(monkeypatch):
    """mock chat.get_service + 注入独立缓存实例，返回调用计数与缓存对象。"""
    import app.api.chat as chat
    from app.cache.query_cache import PermissionAwareQueryCache

    state = {"calls": 0, "cache": PermissionAwareQueryCache(ttl_seconds=300, max_entries=50)}

    class FakeService:
        def query(self, query, history=None, document_ids=None):
            state["calls"] += 1
            return _make_rag_response(query, "答案{}".format(state["calls"]))

        def query_stream(self, query, history=None, document_ids=None):
            state["calls"] += 1
            yield {"type": "meta", "chunks": [{"chunk_id": "c1"}],
                   "stats": {}, "context": "[1] 测试上下文"}
            yield {"type": "delta", "content": "答案{}".format(state["calls"])}
            yield {"type": "citations", "citations": [{"number": 1, "chunk_id": "c1"}]}
            yield {"type": "done", "answer_length": 3}

    monkeypatch.setattr(chat, "get_service", lambda **kw: FakeService())
    monkeypatch.setattr(chat, "get_query_cache", lambda config=None: state["cache"])
    return state


class TestChatCacheIntegration:
    def test_hit_skips_service(self, anon_client, make_token, chat_env):
        token = make_token(roles=["viewer"], permissions=["chat:query"])
        headers = {"Authorization": "Bearer {}".format(token)}
        body = {"query": "同一问题"}

        r1 = anon_client.post("/api/chat", headers=headers, json=body)
        assert r1.status_code == 200
        assert chat_env["calls"] == 1

        r2 = anon_client.post("/api/chat", headers=headers, json=body)
        assert r2.status_code == 200
        # 命中缓存：service 未被再次调用
        assert chat_env["calls"] == 1
        assert r2.json()["answer"] == r1.json()["answer"]

    def test_isolated_by_permissions(self, anon_client, make_token, chat_env):
        token_a = make_token(
            roles=["admin"],
            permissions=["chat:query", "knowledge:read", "knowledge:upload"],
        )
        token_b = make_token(roles=["viewer"], permissions=["chat:query"])
        headers_a = {"Authorization": "Bearer {}".format(token_a)}
        headers_b = {"Authorization": "Bearer {}".format(token_b)}
        body = {"query": "权限隔离问题"}

        r1 = anon_client.post("/api/chat", headers=headers_a, json=body)
        assert chat_env["calls"] == 1

        # B 权限集合不同 → 缓存 key 不同 → 未命中
        r2 = anon_client.post("/api/chat", headers=headers_b, json=body)
        assert r2.status_code == 200
        assert chat_env["calls"] == 2

        # A 再问 → 命中 A 自己的缓存
        r3 = anon_client.post("/api/chat", headers=headers_a, json=body)
        assert r3.status_code == 200
        assert chat_env["calls"] == 2
        assert r3.json()["answer"] == r1.json()["answer"]

    def test_ttl_expiry_reinvokes(self, anon_client, make_token, monkeypatch):
        import app.api.chat as chat
        from app.cache.query_cache import PermissionAwareQueryCache

        state = {"calls": 0}

        class FakeService:
            def query(self, query, history=None, document_ids=None):
                state["calls"] += 1
                return _make_rag_response(query, "答案")

        monkeypatch.setattr(chat, "get_service", lambda **kw: FakeService())
        # ttl=0：写即过期 → 每次请求都重新调用 service
        monkeypatch.setattr(
            chat, "get_query_cache",
            lambda config=None: PermissionAwareQueryCache(ttl_seconds=0, max_entries=10),
        )

        token = make_token(roles=["viewer"], permissions=["chat:query"])
        headers = {"Authorization": "Bearer {}".format(token)}
        body = {"query": "过期测试"}

        assert anon_client.post("/api/chat", headers=headers, json=body).status_code == 200
        assert anon_client.post("/api/chat", headers=headers, json=body).status_code == 200
        assert state["calls"] == 2

    def test_index_version_invalidation(self, anon_client, make_token, chat_env, monkeypatch):
        import app.api.chat as chat

        # 模拟索引版本：v1, v1（命中）, v2（失效）
        versions = iter(["v1", "v1", "v2"])
        monkeypatch.setattr(
            chat, "_index_version",
            lambda strategy, tenant_id="default": next(versions),
        )

        token = make_token(roles=["viewer"], permissions=["chat:query"])
        headers = {"Authorization": "Bearer {}".format(token)}
        body = {"query": "版本失效测试"}

        assert anon_client.post("/api/chat", headers=headers, json=body).status_code == 200
        assert chat_env["calls"] == 1
        assert anon_client.post("/api/chat", headers=headers, json=body).status_code == 200
        assert chat_env["calls"] == 1  # v1 命中

        assert anon_client.post("/api/chat", headers=headers, json=body).status_code == 200
        assert chat_env["calls"] == 2  # v2 失效 → 重新调用


class TestStreamCacheIntegration:
    def test_stream_miss_then_hit(self, anon_client, make_token, chat_env):
        """流式未命中 → 正常流式并写缓存；第二次命中 → 从缓存重放，不调 service。"""
        token = make_token(roles=["viewer"], permissions=["chat:query"])
        headers = {"Authorization": "Bearer {}".format(token)}
        body = {"query": "流式缓存问题"}

        r1 = anon_client.post("/api/chat/stream", headers=headers, json=body)
        assert r1.status_code == 200
        assert chat_env["calls"] == 1
        assert "答案1" in r1.text

        r2 = anon_client.post("/api/chat/stream", headers=headers, json=body)
        assert r2.status_code == 200
        # 命中缓存：service 未再次调用，答案从缓存重放
        assert chat_env["calls"] == 1
        assert "答案1" in r2.text

    def test_stream_shares_cache_with_sync(self, anon_client, make_token, chat_env):
        """同步写入的缓存，流式可直接命中（同一 key，传输方式无关）。"""
        token = make_token(roles=["viewer"], permissions=["chat:query"])
        headers = {"Authorization": "Bearer {}".format(token)}
        body = {"query": "同步流式共用"}

        r1 = anon_client.post("/api/chat", headers=headers, json=body)
        assert r1.status_code == 200
        assert chat_env["calls"] == 1

        # 流式命中同步写入的缓存 → 不再调 service
        r2 = anon_client.post("/api/chat/stream", headers=headers, json=body)
        assert r2.status_code == 200
        assert chat_env["calls"] == 1
        assert "答案1" in r2.text

    def test_stream_writes_cache_used_by_sync(self, anon_client, make_token, chat_env):
        """流式写入的缓存，同步可直接命中。"""
        token = make_token(roles=["viewer"], permissions=["chat:query"])
        headers = {"Authorization": "Bearer {}".format(token)}
        body = {"query": "流式写同步读"}

        r1 = anon_client.post("/api/chat/stream", headers=headers, json=body)
        assert r1.status_code == 200
        assert chat_env["calls"] == 1

        r2 = anon_client.post("/api/chat", headers=headers, json=body)
        assert r2.status_code == 200
        assert chat_env["calls"] == 1
        assert r2.json()["answer"] == "答案1"
