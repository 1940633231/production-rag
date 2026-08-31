"""审计日志：安全相关事件的持久化（MySQL audit_logs 表）。

设计:
  - 事件 action 统一为 "{类型}.{操作}"，如 login.success / login.failure /
    authz.denied / user.create / role.delete / document.upload / document.delete
  - 写入：后台 worker 队列异步落库（fire-and-forget），不阻塞业务请求；
    MySQL 不可用时丢弃并记 warning（审计为 best-effort，不影响主流程）
  - 读取：query() / count() 供审计查询 API 使用
  - 测试：提供 InMemoryAuditLogger（内存实现，不落库），由测试基建注入

用法（业务侧）:
    from app.audit.logger import record
    record(action="document.upload", actor_user_id=user.user_id,
           actor_username=user.username, tenant_id=user.tenant_id,
           resource=doc_id, ip=client_ip)
"""
import queue
import threading
import time
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.storage.mysql import MySQLManager

logger = get_logger(__name__)

_DDL_AUDIT_LOGS = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id             BIGINT        NOT NULL AUTO_INCREMENT,
    tenant_id      VARCHAR(64)   NOT NULL DEFAULT 'default',
    actor_user_id  VARCHAR(64)   NOT NULL DEFAULT '',
    actor_username VARCHAR(128)  NOT NULL DEFAULT '',
    action         VARCHAR(64)   NOT NULL,
    resource       VARCHAR(256)  NOT NULL DEFAULT '',
    result         VARCHAR(16)   NOT NULL DEFAULT 'success',
    ip             VARCHAR(64)   NOT NULL DEFAULT '',
    detail         VARCHAR(1024) NOT NULL DEFAULT '',
    created_at     TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_audit_tenant_created (tenant_id, created_at),
    KEY idx_audit_actor (actor_user_id),
    KEY idx_audit_action (action)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# 供 MySQLManager.init_schema 统一建表
DDL_STATEMENTS = [_DDL_AUDIT_LOGS]


class AuditLogger:
    """审计日志：后台 worker 队列异步写入 MySQL + 同步查询。

    start_worker=False 时后台线程不启动，仅通过 flush() 同步落库
    （测试 / 需要确定性写入的场景使用）。
    """

    def __init__(self, manager: Optional[MySQLManager] = None, max_queue: int = 1000,
                 start_worker: bool = True):
        self.manager = manager or MySQLManager()
        self._queue: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self._worker = threading.Thread(
                target=self._run, daemon=True, name="audit-logger"
            )
            self._worker.start()

    # ---- 写入（fire-and-forget）----

    def record(self, *, tenant_id: str = "default", actor_user_id: str = "",
               actor_username: str = "", action: str = "", resource: str = "",
               result: str = "success", ip: str = "", detail: str = "") -> bool:
        """记录一条审计事件（入队即返回，异步落库）。

        返回 True 表示已入队（不保证已写入）。
        """
        event = {
            "tenant_id": tenant_id or "default",
            "actor_user_id": actor_user_id or "",
            "actor_username": actor_username or "",
            "action": action,
            "resource": (resource or "")[:256],
            "result": result or "success",
            "ip": (ip or "")[:64],
            "detail": (detail or "")[:1024],
        }
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            logger.warning("审计队列已满，丢弃事件: %s", action)
            return False

    def flush(self, timeout: float = 5.0) -> int:
        """同步排空队列（测试/优雅停机用），返回实际写入条数。"""
        deadline = time.time() + timeout
        count = 0
        while time.time() < deadline:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self._insert(event)
            count += 1
        return count

    def _run(self):
        """后台 worker：持续消费队列写入 MySQL。"""
        while True:
            event = self._queue.get()
            self._insert(event)

    def _insert(self, event: Dict) -> None:
        """单条写入 MySQL（软失败）。"""
        try:
            sql = (
                "INSERT INTO audit_logs "
                "(tenant_id, actor_user_id, actor_username, action, resource, "
                "result, ip, detail) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            )
            with self.manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (
                            event["tenant_id"], event["actor_user_id"],
                            event["actor_username"], event["action"],
                            event["resource"], event["result"],
                            event["ip"], event["detail"],
                        ),
                    )
        except Exception as e:
            logger.warning(
                "审计日志写入失败（丢弃）: action=%s, error=%s",
                event.get("action"), e,
            )

    # ---- 查询 ----

    def query(self, tenant_id: Optional[str] = None, actor_user_id: Optional[str] = None,
              action: Optional[str] = None, result: Optional[str] = None,
              resource: Optional[str] = None,
              limit: int = 50, offset: int = 0) -> List[Dict]:
        """查询审计日志（可按租户/用户/action/result/resource 过滤）。"""
        clauses, params = [], []
        if tenant_id is not None:
            clauses.append("tenant_id = %s")
            params.append(tenant_id)
        if actor_user_id:
            clauses.append("actor_user_id = %s")
            params.append(actor_user_id)
        if action:
            clauses.append("action = %s")
            params.append(action)
        if result:
            clauses.append("result = %s")
            params.append(result)
        if resource:
            clauses.append("resource = %s")
            params.append(resource)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT id, tenant_id, actor_user_id, actor_username, action, "
            "resource, result, ip, detail, created_at "
            "FROM audit_logs{} ORDER BY id DESC LIMIT %s OFFSET %s"
        ).format(where)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (*params, int(limit), int(offset)))
                rows = cur.fetchall()
        for r in rows:
            r["created_at"] = str(r.get("created_at", ""))
        return rows

    def count(self, tenant_id: Optional[str] = None, actor_user_id: Optional[str] = None,
              action: Optional[str] = None, result: Optional[str] = None,
              resource: Optional[str] = None) -> int:
        """统计审计日志条数。"""
        clauses, params = [], []
        if tenant_id is not None:
            clauses.append("tenant_id = %s")
            params.append(tenant_id)
        if actor_user_id:
            clauses.append("actor_user_id = %s")
            params.append(actor_user_id)
        if action:
            clauses.append("action = %s")
            params.append(action)
        if result:
            clauses.append("result = %s")
            params.append(result)
        if resource:
            clauses.append("resource = %s")
            params.append(resource)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = "SELECT COUNT(*) AS cnt FROM audit_logs{}".format(where)
        with self.manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()["cnt"]


class InMemoryAuditLogger:
    """内存审计实现（测试用）：record 收集到 events 列表，query 从列表过滤。

    与 AuditLogger 接口一致，便于测试注入与断言，不落库。
    """

    def __init__(self):
        self.events: List[Dict] = []

    def record(self, **event) -> bool:
        self.events.append(dict(event))
        return True

    def flush(self, timeout: float = 5.0) -> int:
        return 0

    def query(self, tenant_id: Optional[str] = None, actor_user_id: Optional[str] = None,
              action: Optional[str] = None, result: Optional[str] = None,
              resource: Optional[str] = None,
              limit: int = 50, offset: int = 0) -> List[Dict]:
        items = list(self.events)
        if tenant_id is not None:
            items = [e for e in items if e.get("tenant_id") == tenant_id]
        if actor_user_id:
            items = [e for e in items if e.get("actor_user_id") == actor_user_id]
        if action:
            items = [e for e in items if e.get("action") == action]
        if result:
            items = [e for e in items if e.get("result") == result]
        if resource:
            items = [e for e in items if e.get("resource") == resource]
        # 与 DB 排序一致：后写入在前
        items = list(reversed(items))
        return items[offset:offset + limit]

    def count(self, **kwargs) -> int:
        return len(self.query(**kwargs))


# ---- 模块级单例 + 统一入口 ----
_logger = None
_lock = threading.Lock()


def get_audit_logger(config=None) -> AuditLogger:
    """获取全局审计日志单例（首次按 config 初始化）。"""
    global _logger
    with _lock:
        if _logger is None:
            if config is None:
                from app.core.config import Config
                config = Config()
            _logger = AuditLogger()
            logger.info("审计日志单例初始化完成")
        return _logger


def reset_audit_logger() -> None:
    """清空单例（测试用）。"""
    global _logger
    with _lock:
        _logger = None


def record(*, action: str = "", **event) -> bool:
    """业务侧统一审计入口：按配置开关/白名单过滤后写入。

    幂等且不影响主流程：配置关闭/类型不在白名单/写入失败均静默返回 False。
    """
    global _logger
    try:
        from app.core.config import Config
        cfg = Config()
        if not getattr(cfg, "audit_enabled", True):
            return False
        types = getattr(cfg, "audit_record_types", "*")
        if types != "*" and action:
            evt_type = action.split(".")[0]
            if evt_type not in types:
                return False
    except Exception:
        # 配置读取异常时默认记录（审计不应因配置问题被静默关闭）
        pass

    if _logger is None:
        _logger = AuditLogger()
    try:
        return _logger.record(action=action, **event)
    except Exception as e:
        logger.warning("审计事件记录异常: %s", e)
        return False
