"""权限感知查询缓存的 Redis 后端。

与 PermissionAwareQueryCache 接口完全一致（get/set/clear/__len__/stats），
可直接替换内存实现；key 复用 build_query_cache_key 生成的 sha256 摘要，
值以 JSON 序列化存入 Redis，TTL 由 Redis 端 SETEX 控制。

设计:
  - 值必须是 JSON 可序列化（chat.py 传入 result.model_dump() 的 dict）
  - 优雅降级：Redis 连接失败/运行期异常时不抛错——
    get 视为 miss（实时检索兜底）、set 跳过，保证 Redis 故障不影响服务
  - key 统一加 prefix 命名空间，便于按前缀清理/统计
"""
import json
import threading
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    redis_lib = None
    _REDIS_AVAILABLE = False


class RedisQueryCache:
    """基于 Redis 的查询缓存（TTL 由 SETEX 控制，值 JSON 序列化）。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: int = 300,
        prefix: str = "rag:qcache:",
        client=None,
    ):
        if not _REDIS_AVAILABLE:
            raise RuntimeError("redis-py 未安装，请运行: pip install 'redis>=5,<6'")
        self.host = host
        self.port = int(port)
        self.db = int(db)
        self.password = password
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.prefix = prefix
        # 测试可注入 client；否则按参数新建
        self._redis = client or redis_lib.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        # 本地统计（Redis 不可用时也维护命中率）
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._connected = True

    # ---- 内部 ----

    def _full_key(self, key: str) -> str:
        return self.prefix + key

    def _serialize(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _deserialize(raw: str) -> Any:
        return json.loads(raw)

    # ---- 对外（与 PermissionAwareQueryCache 对齐）----

    def ping(self) -> bool:
        """探测 Redis 连通性。"""
        try:
            return bool(self._redis.ping())
        except Exception as e:
            logger.warning("Redis ping 失败: %s", e)
            return False

    def get(self, key: str) -> Optional[Any]:
        """读取缓存。命中返回反序列化后的值，未命中/异常返回 None。"""
        try:
            raw = self._redis.get(self._full_key(key))
        except Exception as e:
            # Redis 故障：视为 miss，走实时检索兜底
            logger.debug("Redis get 异常（视为 miss）: %s", e)
            with self._lock:
                self._misses += 1
            return None
        if raw is None:
            with self._lock:
                self._misses += 1
            return None
        with self._lock:
            self._hits += 1
        try:
            return self._deserialize(raw)
        except (TypeError, ValueError) as e:
            logger.warning("Redis 缓存值反序列化失败，删除脏数据: %s", e)
            self.delete(key)
            return None

    def set(self, key: str, value: Any) -> None:
        """写入缓存（SETEX，TTL 秒后自动过期）。异常时跳过不抛错。"""
        try:
            self._redis.setex(
                self._full_key(key),
                self.ttl_seconds,
                self._serialize(value),
            )
        except Exception as e:
            logger.debug("Redis set 异常（跳过缓存写入）: %s", e)

    def delete(self, key: str) -> None:
        """删除单个缓存条目。"""
        try:
            self._redis.delete(self._full_key(key))
        except Exception as e:
            logger.debug("Redis delete 异常: %s", e)

    def clear(self) -> None:
        """清空本缓存命名空间下的全部条目。"""
        try:
            keys = list(self._redis.scan_iter(match=self.prefix + "*", count=500))
            if keys:
                self._redis.delete(*keys)
        except Exception as e:
            logger.warning("Redis clear 异常: %s", e)

    def __len__(self) -> int:
        try:
            return sum(1 for _ in self._redis.scan_iter(
                match=self.prefix + "*", count=500
            ))
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        """缓存统计（含 Redis 端条目数）。"""
        with self._lock:
            hits, misses = self._hits, self._misses
        return {
            "backend": "redis",
            "hits": hits,
            "misses": misses,
            "entries": len(self),
            "ttl_seconds": self.ttl_seconds,
            "host": self.host,
            "port": self.port,
            "db": self.db,
        }
