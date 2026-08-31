"""权限感知查询缓存：RAG 问答结果的内存缓存（TTL + LRU）。

设计（Permission-aware Cache）:
  - 缓存 key 由「租户 + 权限指纹 + 归一化 query + 策略/模式 + 索引版本 + 历史」构成：
      * 不同租户不共享缓存（租户隔离）
      * 同一租户内不同权限集合的用户不共享缓存（权限隔离）——
        权限指纹 = 用户权限集合的有序哈希，角色/权限变更后 key 自动变化、旧缓存失效
      * 索引重建/上传/删除后索引版本变化 → 自动失效，不会返回旧数据
  - 内存实现：OrderedDict 实现 LRU + 惰性 TTL 过期，线程安全

用法:
    from app.cache.query_cache import build_query_cache_key, get_query_cache

    key = build_query_cache_key(tenant_id="acme", permissions=["chat:query"], query="...", ...)
    cache = get_query_cache()
    if (value := cache.get(key)) is not None:
        return value
    ...
    cache.set(key, value)
"""
import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


def build_permission_fingerprint(permissions: Optional[List[str]]) -> str:
    """权限指纹：权限集合排序后拼接。

    相同权限集合得到相同指纹；权限变化（角色变更）得到不同指纹，
    从而让旧缓存 key 失效，防止把权限变更前的答案返回给新权限用户。
    """
    if not permissions:
        return "none"
    return ",".join(sorted(set(permissions)))


def build_query_cache_key(
    *,
    tenant_id: str,
    user_id: Optional[str],
    permissions: Optional[List[str]],
    query: str,
    strategy: str,
    mode: str,
    use_rerank: bool,
    index_version: str,
    history: Optional[List[Dict]] = None,
) -> str:
    """构造权限感知的查询缓存 key（sha256）。

    输入全部参与哈希；query 去掉首尾空白以吸收轻微输入差异。
    history 参与哈希（json 归一化），不同对话上下文视为不同请求。
    user_id 参与哈希：文档级 ACL 下不同用户的可读文档集合可能不同，
    避免跨用户共享缓存（permission-aware cache 的用户级隔离）。
    """
    parts = [
        "v1",
        tenant_id or "default",
        user_id or "",
        build_permission_fingerprint(permissions),
        strategy,
        mode,
        str(bool(use_rerank)),
        index_version or "missing",
        (query or "").strip(),
    ]
    if history:
        parts.append(json.dumps(history, ensure_ascii=False, sort_keys=True))
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class PermissionAwareQueryCache:
    """线程安全的权限感知查询缓存（TTL + LRU 淘汰）。"""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 2000):
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        # key -> (expire_at, value)
        self._store: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()
        # 统计
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ---- 内部 ----

    def _prune_expired_locked(self, now: float):
        """删除已过期条目（需持有锁）。exp <= now 视为过期（ttl=0 立即失效）。"""
        expired = [
            k for k, (exp, _) in self._store.items() if exp <= now
        ]
        for k in expired:
            del self._store[k]

    # ---- 对外 ----

    def get(self, key: str) -> Optional[Any]:
        """读取缓存。命中（未过期）返回 value，否则返回 None。"""
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None
            exp, value = item
            if exp <= now:
                del self._store[key]
                self._misses += 1
                return None
            # LRU：命中条目移到末尾
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """写入缓存。超出 max_entries 时淘汰最久未用条目。"""
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            self._store[key] = (now + self.ttl_seconds, value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """清空全部缓存。"""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            self._prune_expired_locked(time.time())
            return len(self._store)

    def stats(self) -> Dict[str, int]:
        """缓存统计。"""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "entries": len(self._store),
                "evictions": self._evictions,
                "max_entries": self.max_entries,
            }


# ---- 模块级单例 ----
_query_cache: Optional[PermissionAwareQueryCache] = None
_singleton_lock = threading.Lock()


def get_query_cache(config=None) -> PermissionAwareQueryCache:
    """获取（并首次按 config 初始化）全局查询缓存单例。"""
    global _query_cache
    with _singleton_lock:
        if _query_cache is None:
            if config is None:
                from app.core.config import Config
                config = Config()
            _query_cache = PermissionAwareQueryCache(
                ttl_seconds=config.cache_ttl_seconds,
                max_entries=config.cache_max_entries,
            )
            logger.info(
                "查询缓存初始化: ttl=%ss, max_entries=%d",
                _query_cache.ttl_seconds, _query_cache.max_entries,
            )
        return _query_cache


def reset_query_cache() -> None:
    """清空并重置单例（测试或配置切换时用）。"""
    global _query_cache
    with _singleton_lock:
        if _query_cache is not None:
            _query_cache.clear()
        _query_cache = None
