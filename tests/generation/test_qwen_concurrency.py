r"""QwenGenerator 并发限流单元测试：信号量限制同时进行的 DashScope 调用。

覆盖:
  - max_concurrency 限制并发峰值
  - max_concurrency=0 不限制
  - 流式消费后信号量正确释放

运行:
  .venv\Scripts\python.exe -m pytest tests\generation\test_qwen_concurrency.py -v
"""
import sys
import threading
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _install_dashscope():
    """安装 mock dashscope 模块（避免真实 SDK 依赖）。"""
    m = types.ModuleType("dashscope")
    m.__version__ = "0.1"
    m.Generation = types.SimpleNamespace()
    sys.modules["dashscope"] = m
    return m


def _fake_response(content="ok"):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg)
    output = types.SimpleNamespace(choices=[choice])
    return types.SimpleNamespace(status_code=200, output=output)


class _ConcurrencyCounter:
    """统计同时进入 call 的最大并发数。"""

    def __init__(self):
        self.cur = 0
        self.max = 0
        self.lock = threading.Lock()

    def enter(self):
        with self.lock:
            self.cur += 1
            self.max = max(self.max, self.cur)

    def exit(self):
        with self.lock:
            self.cur -= 1


class TestQwenConcurrency:
    def _make_generator(self, max_concurrency, delay=0.05):
        from app.generation.qwen import QwenGenerator

        dashscope = _install_dashscope()
        counter = _ConcurrencyCounter()

        def slow_call(**kw):
            counter.enter()
            try:
                time.sleep(delay)
                return _fake_response()
            finally:
                counter.exit()

        dashscope.Generation.call = slow_call
        g = QwenGenerator(
            api_key="sk-test", timeout=0, retry_times=0,
            max_concurrency=max_concurrency,
        )
        return g, counter

    def test_semaphore_limits_concurrency(self):
        """max_concurrency=2 时并发峰值不应超过 2。"""
        g, counter = self._make_generator(max_concurrency=2)
        threads = [
            threading.Thread(
                target=lambda: g.generate([{"role": "user", "content": "q"}])
            )
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert counter.max == 2

    def test_zero_concurrency_disables_limit(self):
        """max_concurrency=0 时不创建信号量。"""
        g, _ = self._make_generator(max_concurrency=0)
        assert g._sem is None
        # 直接调用应成功
        assert g.generate([{"role": "user", "content": "q"}]) == "ok"

    def test_stream_releases_semaphore(self):
        """流式消费完成后信号量应释放（可再次获取）。"""
        from app.generation.qwen import QwenGenerator

        dashscope = _install_dashscope()
        dashscope.Generation.call = lambda **kw: [_fake_response("片段A")]
        g = QwenGenerator(
            api_key="sk-test", timeout=0, retry_times=0, max_concurrency=1,
        )
        events = list(g.stream_generate([{"role": "user", "content": "q"}]))
        assert "".join(events) == "片段A"
        # max_concurrency=1 的 BoundedSemaphore 初始值=1，消费完应恢复
        assert g._sem._value == 1

    def test_stream_early_break_releases_semaphore(self):
        """流式中途 break（调用方放弃）也应释放信号量。"""
        from app.generation.qwen import QwenGenerator

        dashscope = _install_dashscope()
        dashscope.Generation.call = lambda **kw: [
            _fake_response("A"), _fake_response("B"),
        ]
        g = QwenGenerator(
            api_key="sk-test", timeout=0, retry_times=0, max_concurrency=1,
        )
        gen = g.stream_generate([{"role": "user", "content": "q"}])
        first = next(gen)  # 取一个就放弃
        assert first == "A"
        gen.close()  # 触发 GeneratorExit → finally 释放
        assert g._sem._value == 1
