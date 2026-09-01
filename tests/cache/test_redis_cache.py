r"""Redis 查询缓存测试。

覆盖:
  - RedisQueryCache 单元：set/get 往返 / miss / 脏数据清理 / 故障降级 / 统计
  - 后端选择：get_query_cache 按 cache.backend 选 Redis，不可用时降级内存
  - 集成（可选）：连真实 Redis（config 的 cache.redis 地址），不可达自动 skip

运行:
  .venv\Scripts\python.exe -m pytest tests\cache\test_redis_cache.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.cache.redis_cache import RedisQueryCache  # noqa: E402


# ---- 内存版 fake Redis（注入 client，避免依赖真实服务做单元测试）----

class _FakeRedis:
    """实现 RedisQueryCache 用到的 redis-py 方法子集。"""

    def __init__(self):
        self._store = {}  # key(str) -> value(str)
        self._ttls = {}
        self.calls = []

    def ping(self):
        return True

    def get(self, key):
        self.calls.append(("get", key))
        return self._store.get(key)

    def setex(self, key, ttl, value):
        self.calls.append(("setex", key, ttl))
        self._store[key] = value
        self._ttls[key] = ttl

    def delete(self, *keys):
        self.calls.append(("delete", keys))
        n = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                n += 1
        return n

    def scan_iter(self, match=None, count=100):
        prefix = match.rstrip("*") if match else ""
        return iter([k for k in self._store if k.startswith(prefix)])


def _make_cache(fake, **kwargs):
    defaults = dict(
        host="fake", port=6379, ttl_seconds=300, prefix="rag:qcache:", client=fake,
    )
    defaults.update(kwargs)
    return RedisQueryCache(**defaults)


class TestRedisCacheUnit:
    def test_set_get_roundtrip(self):
        fake = _FakeRedis()
        c = _make_cache(fake)
        c.set("k1", {"answer": "你好", "n": 1})
        assert c.get("k1") == {"answer": "你好", "n": 1}
        assert fake.calls[0][0] == "setex"
        assert fake._ttls["rag:qcache:k1"] == 300  # TTL 落到 Redis 端

    def test_miss_returns_none(self):
        fake = _FakeRedis()
        c = _make_cache(fake)
        assert c.get("missing") is None
        assert c.stats()["misses"] == 1

    def test_dirty_value_deleted(self):
        fake = _FakeRedis()
        fake._store["rag:qcache:bad"] = "{not-json"
        c = _make_cache(fake)
        assert c.get("bad") is None
        assert "rag:qcache:bad" not in fake._store  # 反序列化失败删除脏数据

    def test_graceful_degradation_on_error(self):
        """Redis 运行期故障：get 视为 miss、set/clear 静默跳过，不抛错。"""
        class _Err:
            def get(self, key):
                raise ConnectionError("down")
            def setex(self, *a, **k):
                raise ConnectionError("down")
            def delete(self, *a, **k):
                raise ConnectionError("down")
            def scan_iter(self, **k):
                raise ConnectionError("down")
        c = _make_cache(_Err())
        assert c.get("k") is None
        c.set("k", {"a": 1})  # 不抛错
        c.clear()              # 不抛错
        assert len(c) == 0

    def test_stats(self):
        fake = _FakeRedis()
        c = _make_cache(fake)
        c.set("k", "v")
        c.get("k")
        c.get("k")       # 命中
        c.get("miss")    # miss
        s = c.stats()
        assert s["backend"] == "redis"
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["entries"] == 1

    def test_clear_and_len(self):
        fake = _FakeRedis()
        c = _make_cache(fake)
        c.set("a", 1)
        c.set("b", 2)
        assert len(c) == 2
        c.clear()
        assert len(c) == 0


# ---- 后端选择（get_query_cache）----

class TestBackendSelection:
    def test_redis_backend_used(self, monkeypatch):
        import app.cache.query_cache as qc

        # 用 list 作 sentinel：带 clear()（reset_query_cache 会调用）
        sentinel = []
        monkeypatch.setattr(qc, "_try_build_redis_cache", lambda config: sentinel)

        class Cfg:
            cache_backend = "redis"
            cache_ttl_seconds = 300
            cache_max_entries = 100

        qc.reset_query_cache()
        try:
            assert qc.get_query_cache(Cfg()) is sentinel
        finally:
            qc.reset_query_cache()

    def test_falls_back_to_memory(self, monkeypatch):
        import app.cache.query_cache as qc

        monkeypatch.setattr(qc, "_try_build_redis_cache", lambda config: None)

        class Cfg:
            cache_backend = "redis"
            cache_ttl_seconds = 300
            cache_max_entries = 100

        qc.reset_query_cache()
        try:
            c = qc.get_query_cache(Cfg())
            assert isinstance(c, qc.PermissionAwareQueryCache)
        finally:
            qc.reset_query_cache()

    def test_memory_backend_skips_redis(self, monkeypatch):
        import app.cache.query_cache as qc

        called = []
        monkeypatch.setattr(
            qc, "_try_build_redis_cache",
            lambda config: called.append(1) or object(),
        )

        class Cfg:
            cache_backend = "memory"
            cache_ttl_seconds = 300
            cache_max_entries = 100

        qc.reset_query_cache()
        try:
            c = qc.get_query_cache(Cfg())
            assert isinstance(c, qc.PermissionAwareQueryCache)
            assert called == []  # memory 后端不尝试 Redis
        finally:
            qc.reset_query_cache()


# ---- 集成（真实 Redis，不可达自动 skip）----

@pytest.fixture(scope="module")
def real_redis_cache():
    from app.core.config import Config

    cfg = Config()
    if cfg.cache_backend != "redis":
        pytest.skip("config cache.backend 不是 redis")
    c = RedisQueryCache(
        host=cfg.cache_redis_host,
        port=cfg.cache_redis_port,
        db=cfg.cache_redis_db,
        password=cfg.cache_redis_password,
        ttl_seconds=60,
        prefix="pytest:qcache:",
    )
    if not c.ping():
        pytest.skip("Redis 不可达: {}:{}".format(cfg.cache_redis_host, cfg.cache_redis_port))
    c.clear()
    yield c
    c.clear()


class TestRedisIntegration:
    def test_roundtrip_against_real_redis(self, real_redis_cache):
        c = real_redis_cache
        c.set("it-key", {"a": 1, "b": "中文内容"})
        assert c.get("it-key") == {"a": 1, "b": "中文内容"}
        assert len(c) >= 1
        assert c.stats()["backend"] == "redis"

    def test_ttl_expires(self, real_redis_cache):
        c = real_redis_cache
        c._redis.setex(c.prefix + "ttl-key", 1, '{"v": 1}')
        import time
        time.sleep(1.2)
        assert c.get("ttl-key") is None
