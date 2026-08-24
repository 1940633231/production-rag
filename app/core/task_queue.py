"""后台任务队列：进程内线程池 + 任务状态管理。

用于长耗时操作（如大文档 upload / rebuild 索引），避免阻塞 HTTP 长连接。

设计:
    - 基于 concurrent.futures.ThreadPoolExecutor（标准库，零新增依赖）
    - 任务状态用 dict 维护，进程内共享
    - 任务 id 由 uuid4 生成
    - 任务完成后保留结果，超 1 小时未查询自动清理（简化实现，生产可换 Redis）

使用示例:
    from app.core.task_queue import task_manager

    # 提交后台任务
    task_id = task_manager.submit("ingest", my_long_func, arg1, kwarg=v)

    # 查询状态
    status = task_manager.get(task_id)
    # status = {"task_id": "...", "type": "ingest", "status": "running"|"done"|"failed",
    #           "progress": None, "result": {...}, "error": None, "started_at": ..., "finished_at": ...}

后续若需持久化或分布式，可替换为 Celery/RQ + Redis，接口保持一致。
"""
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# 任务记录保留时长（秒），超时自动清理
_RETENTION_SECONDS = 3600


class TaskInfo:
    """单条任务的状态记录。"""

    def __init__(self, task_id: str, task_type: str):
        self.task_id = task_id
        self.type = task_type
        self.status = "pending"  # pending / running / done / failed
        self.progress: Optional[float] = None  # 0.0~1.0，可选
        self.result: Any = None
        self.error: Optional[str] = None
        self.started_at: float = time.time()
        self.finished_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "type": self.type,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": (
                (self.finished_at or time.time()) - self.started_at
            ),
        }


class TaskManager:
    """后台任务管理器：线程池 + 状态字典。"""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rag-task"
        )
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()
        logger.info("TaskManager 初始化: max_workers=%d", max_workers)

    def submit(
        self,
        task_type: str,
        fn: Callable,
        *args,
        **kwargs,
    ) -> str:
        """提交后台任务，返回 task_id。

        参数:
            task_type: 任务类型标识（如 "upload" / "rebuild"），便于分类查询
            fn: 同步可调用对象（async 函数不支持，请用 asyncio.run 包裹）
            *args, **kwargs: 传给 fn 的参数
        """
        task_id = "task_{}_{}".format(task_type, uuid.uuid4().hex[:8])
        info = TaskInfo(task_id, task_type)
        info.status = "running"

        def _runner():
            try:
                result = fn(*args, **kwargs)
                info.result = result
                info.status = "done"
            except Exception as e:
                logger.error(
                    "后台任务失败: task_id=%s, type=%s, error=%s",
                    task_id, task_type, e, exc_info=True,
                )
                info.error = "后台任务失败: {}".format(e)
                info.status = "failed"
            finally:
                info.finished_at = time.time()
                logger.info(
                    "后台任务结束: task_id=%s, type=%s, status=%s, elapsed=%.3fs",
                    task_id, task_type, info.status,
                    info.finished_at - info.started_at,
                )

        with self._lock:
            self._tasks[task_id] = info
        self._executor.submit(_runner)
        logger.info("后台任务已提交: task_id=%s, type=%s", task_id, task_type)
        return task_id

    def get(self, task_id: str) -> Optional[Dict]:
        """查询任务状态，返回 dict 或 None（不存在）。"""
        with self._lock:
            info = self._tasks.get(task_id)
            if info is None:
                return None
            return info.to_dict()

    def list_by_type(self, task_type: str, limit: int = 50) -> list:
        """按类型查询任务列表（最近的在前）。"""
        with self._lock:
            items = [info.to_dict() for info in self._tasks.values()
                     if info.type == task_type]
        items.sort(key=lambda x: x["started_at"], reverse=True)
        return items[:limit]

    def cleanup_expired(self) -> int:
        """清理已完成且超 _RETENTION_SECONDS 的任务记录，返回清理数。"""
        now = time.time()
        removed = 0
        with self._lock:
            to_remove = []
            for tid, info in self._tasks.items():
                if info.status in ("done", "failed") and info.finished_at:
                    if now - info.finished_at > _RETENTION_SECONDS:
                        to_remove.append(tid)
            for tid in to_remove:
                del self._tasks[tid]
                removed += 1
        if removed:
            logger.info("清理过期任务记录: %d 条", removed)
        return removed


# 模块级单例
task_manager = TaskManager(max_workers=4)
