"""Prometheus 指标采集模块。

依赖: prometheus-client (轻量纯 Python)

定义的指标:
  - rag_http_requests_total: Counter (method, path, status)
  - rag_http_request_duration_seconds: Histogram (method, path)
  - rag_query_total: Counter (strategy, mode, use_rerank)
  - rag_retrieval_duration_seconds: Histogram (strategy)
  - rag_rerank_duration_seconds: Histogram
  - rag_generation_duration_seconds: Histogram (backend)

使用方式:
  # 在 API 端点中记录
  from app.core.metrics import metrics
  metrics.record_http_request("GET", "/api/health", 200, 0.012)
  metrics.record_query("recursive", "vector", True)
  metrics.record_retrieval("recursive", 0.15)
  metrics.record_generation("qwen", 1.2)

  # /metrics 端点直接用 prometheus_client.generate_latest
"""
import time
from typing import Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# 尝试导入 prometheus_client，不可用时降级为 no-op
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client 未安装，指标采集降级为 no-op")


class _NoopMetric:
    """prometheus_client 不可用时的空实现。"""
    def labels(self, *args, **kwargs):
        return self
    def inc(self, *args, **kwargs):
        pass
    def observe(self, *args, **kwargs):
        pass
    def set(self, *args, **kwargs):
        pass


class MetricsRegistry:
    """指标注册中心：统一管理所有 Prometheus 指标。"""

    def __init__(self):
        if _PROMETHEUS_AVAILABLE:
            self._http_requests = Counter(
                "rag_http_requests_total",
                "HTTP 请求总数",
                ["method", "path", "status"],
            )
            self._http_duration = Histogram(
                "rag_http_request_duration_seconds",
                "HTTP 请求耗时",
                ["method", "path"],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            )
            self._query_total = Counter(
                "rag_query_total",
                "RAG 查询总数",
                ["strategy", "mode", "use_rerank"],
            )
            self._retrieval_duration = Histogram(
                "rag_retrieval_duration_seconds",
                "检索阶段耗时",
                ["strategy"],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
            self._rerank_duration = Histogram(
                "rag_rerank_duration_seconds",
                "重排阶段耗时",
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
            self._generation_duration = Histogram(
                "rag_generation_duration_seconds",
                "LLM 生成耗时",
                ["backend"],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
            )
            self._index_chunks = Gauge(
                "rag_index_chunks",
                "索引中 chunk 总数",
                ["strategy"],
            )
            self._active_tasks = Gauge(
                "rag_active_tasks",
                "活跃后台任务数",
            )
        else:
            noop = _NoopMetric()
            self._http_requests = noop
            self._http_duration = noop
            self._query_total = noop
            self._retrieval_duration = noop
            self._rerank_duration = noop
            self._generation_duration = noop
            self._index_chunks = noop
            self._active_tasks = noop

    # ---- HTTP 指标 ----
    def record_http_request(self, method: str, path: str, status: int, duration: float):
        """记录一次 HTTP 请求。"""
        self._http_requests.labels(method=method, path=path, status=status).inc()
        self._http_duration.labels(method=method, path=path).observe(duration)

    # ---- RAG 链路指标 ----
    def record_query(self, strategy: str, mode: str, use_rerank: bool):
        """记录一次 RAG 查询。"""
        self._query_total.labels(
            strategy=strategy, mode=mode, use_rerank=str(use_rerank),
        ).inc()

    def record_retrieval(self, strategy: str, duration: float):
        """记录检索阶段耗时。"""
        self._retrieval_duration.labels(strategy=strategy).observe(duration)

    def record_rerank(self, duration: float):
        """记录重排阶段耗时。"""
        self._rerank_duration.observe(duration)

    def record_generation(self, backend: str, duration: float):
        """记录 LLM 生成耗时。"""
        self._generation_duration.labels(backend=backend).observe(duration)

    # ---- 资源指标 ----
    def set_index_chunks(self, strategy: str, count: int):
        """设置索引 chunk 总数。"""
        self._index_chunks.labels(strategy=strategy).set(count)

    def set_active_tasks(self, count: int):
        """设置活跃后台任务数。"""
        self._active_tasks.set(count)

    # ---- 导出 ----
    def export(self) -> tuple:
        """导出 Prometheus 格式文本。

        返回 (content_bytes, content_type)。
        """
        if not _PROMETHEUS_AVAILABLE:
            return (b"# prometheus_client not installed\n", "text/plain")
        return (generate_latest(), CONTENT_TYPE_LATEST)


# 模块级单例
metrics = MetricsRegistry()
