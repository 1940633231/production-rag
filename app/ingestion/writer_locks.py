"""跨进程可重入写锁（从 writer.py 拆出）。

解决 P1-4（FAISS/metadata.json 多 worker 并发写竞态）与
P1-5（rebuild 与 upload 之间缺 tenant+strategy 级互斥）。

实现：
  - 锁文件置于该 (strategy, tenant) 的 index 目录下（write.lock）
  - Unix 用 fcntl.flock(LOCK_EX)；Windows 用 msvcrt.locking 锁首字节
  - 进程内可重入：嵌套调用（如 rebuild → write）复用同一把 OS 锁，
    通过线程本地计数避免死锁；跨进程由 OS 互斥保证串行
"""
import sys
import threading
import time
from pathlib import Path

from app.core.logger import get_logger

logger = get_logger(__name__)


# 线程本地：记录当前线程已持有的写锁栈，实现进程内可重入
_lock_local = threading.local()


class IndexWriteLock:
    """（strategy, tenant）粒度跨进程互斥写锁（Windows/Unix 兼容、可重入）。"""

    def __init__(self, lock_file: Path):
        self.lock_file = Path(lock_file)
        self._fh = None
        self._own = False

    def __enter__(self):
        stack = getattr(_lock_local, "held", dict())
        key = str(self.lock_file)
        if key in stack:
            # 进程内已持有该锁 → 可重入，仅计数
            stack[key] += 1
            _lock_local.held = stack
            return self

        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.lock_file, "a+", encoding="utf-8")
        if sys.platform == "win32":
            import msvcrt
            self._fh.seek(0)
            if self._fh.read(1) == "":
                self._fh.write("L")
                self._fh.flush()
            self._fh.seek(0)
            while True:
                try:
                    # 锁文件区段：从当前位置起锁定 1 字节
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)

        stack[key] = 1
        _lock_local.held = stack
        self._own = True
        return self

    def __exit__(self, *exc):
        stack = getattr(_lock_local, "held", dict())
        key = str(self.lock_file)
        cnt = stack.get(key, 0) - 1
        if cnt <= 0:
            stack.pop(key, None)
        else:
            stack[key] = cnt
        _lock_local.held = stack

        if self._own and self._fh is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self._fh.seek(0)
                    try:
                        msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None
        self._own = False
        return False
