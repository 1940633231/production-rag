"""Prometheus 指标采集模块。

依赖: prometheus-client (轻量纯 Python)

定义的指标:
  - rag_http_requests_total: Counter (method, path, status)
  - rag_http_request_duration_seconds: Histogram (method, path)
  - rag_query_total: Counter (strategy, mode, use_rerank)
  - rag_query_latency_seconds: Histogram (strategy, mode)
  - rag_query_latency_p50/p95/p99_seconds: Gauge (strategy, mode) 有界窗口分位数
  - rag_retrieval_duration_seconds: Histogram (strategy)
  - rag_embedding_duration_seconds: Histogram (strategy)  # query embedding
  - rag_vector_duration_seconds: Histogram (strategy)     # 向量库检索
  - rag_bm25_duration_seconds: Histogram (strategy)       # BM25 打分排序
  - rag_sparse_duration_seconds: Histogram (strategy)     # 稀疏检索（BM25 / ES 共用）
  - rag_rrf_duration_seconds: Histogram (strategy)        # RRF 融合
  - rag_rerank_duration_seconds: Histogram
  - rag_generation_duration_seconds: Histogram (backend)  # LLM 总耗时
  - rag_llm_ttft_seconds: Histogram (backend)             # LLM 首 token 延迟

使用方式:
  # 在 API 端点中记录
  from app.core.metrics import metrics
  metrics.record_http_request("GET", "/api/health", 200, 0.012)
  metrics.record_query("recursive", "vector", True)
  metrics.record_query_latency("recursive", "vector", 1.5)
  metrics.record_retrieval("recursive", 0.15)
  metrics.record_embedding("recursive", 0.02)
  metrics.record_vector("recursive", 0.03)
  metrics.record_bm25("recursive", 0.004)
  metrics.record_sparse("recursive", 0.004)
  metrics.record_rrf("recursive", 0.0001)
  metrics.record_llm_ttft("qwen", 0.4)
  metrics.record_generation("qwen", 1.2)

  # /metrics 端点直接用 prometheus_client.generate_latest
"""
import time
from collections import deque
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


class _NoopPercentile:
    """prometheus_client 不可用时的空分位数实现。"""
    def observe(self, *args, **kwargs):
        pass


class _PercentileWindow:
    """有界滑动窗口的 P50/P95/P99 分位数，以三个 Gauge 暴露。

    当前 prometheus-client 构建的 Summary 不支持 quantiles 参数，
    故自行实现：维护最近 N 个样本（deque，超窗自动淘汰），
    每次 observe 重算 P50/P95/P99 并写入对应 Gauge（nearest-rank）。
    另配同名 Histogram 保留 _count/_sum/bucket，供 PromQL histogram_quantile 使用。
    """

    def __init__(self, name: str, documentation: str, labelnames, window: int = 1000):
        self._window = window
        self._labelnames = list(labelnames)
        self._buf = {}
        self._gauges = {}
        for q in (0.5, 0.95, 0.99):
            self._gauges[q] = Gauge(
                "{}_p{:02d}_seconds".format(name, int(q * 100)),
                "{}（P{:.0f}）".format(documentation, q * 100),
                self._labelnames,
            )

    def _samples(self, labelvalues):
        key = tuple(labelvalues)
        if key not in self._buf:
            self._buf[key] = deque(maxlen=self._window)
        return self._buf[key]

    def observe(self, value: float, **labels):
        labelvalues = [labels.get(l) for l in self._labelnames]
        buf = self._samples(labelvalues)
        buf.append(value)
        n = len(buf)
        if n == 0:
            return
        sorted_samples = sorted(buf)
        for q, g in self._gauges.items():
            idx = min(n - 1, int(q * (n - 1)))
            g.labels(*labelvalues).set(sorted_samples[idx])


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
            self._query_latency = Histogram(
                "rag_query_latency_seconds",
                "RAG 查询总耗时",
                ["strategy", "mode"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            )
            self._query_latency_pct = _PercentileWindow(
                "rag_query_latency",
                "RAG 查询总耗时",
                ["strategy", "mode"],
            )
            self._retrieval_duration = Histogram(
                "rag_retrieval_duration_seconds",
                "检索阶段耗时",
                ["strategy"],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
            self._embedding_duration = Histogram(
                "rag_embedding_duration_seconds",
                "Query Embedding 阶段耗时",
                ["strategy"],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
            self._vector_duration = Histogram(
                "rag_vector_duration_seconds",
                "向量库检索阶段耗时",
                ["strategy"],
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            )
            self._bm25_duration = Histogram(
                "rag_bm25_duration_seconds",
                "BM25 打分排序阶段耗时",
                ["strategy"],
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            )
            self._sparse_duration = Histogram(
                "rag_sparse_duration_seconds",
                "稀疏检索耗时（BM25 / ES 共用）",
                ["strategy"],
                buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            )
            self._rrf_duration = Histogram(
                "rag_rrf_duration_seconds",
                "RRF 融合阶段耗时",
                ["strategy"],
                buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
            )
            self._rerank_duration = Histogram(
                "rag_rerank_duration_seconds",
                "重排阶段耗时",
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
            self._generation_duration = Histogram(
                "rag_generation_duration_seconds",
                "LLM 生成总耗时",
                ["backend"],
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
            )
            self._llm_ttft = Histogram(
                "rag_llm_ttft_seconds",
                "LLM 首 token 延迟（TTFT）",
                ["backend"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
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
            self._cache_hits = Counter(
                "rag_query_cache_hits_total",
                "权限感知查询缓存命中次数",
            )
            self._cache_misses = Counter(
                "rag_query_cache_misses_total",
                "权限感知查询缓存未命中次数",
            )
            self._cache_entries = Gauge(
                "rag_query_cache_entries",
                "权限感知查询缓存当前条目数",
            )
        else:
            noop = _NoopMetric()
            self._http_requests = noop
            self._http_duration = noop
            self._query_total = noop
            self._query_latency = noop
            self._query_latency_pct = _NoopPercentile()
            self._retrieval_duration = noop
            self._embedding_duration = noop
            self._vector_duration = noop
            self._bm25_duration = noop
            self._sparse_duration = noop
            self._rrf_duration = noop
            self._rerank_duration = noop
            self._generation_duration = noop
            self._llm_ttft = noop
            self._index_chunks = noop
            self._active_tasks = noop
            self._cache_hits = noop
            self._cache_misses = noop
            self._cache_entries = noop

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

    def record_query_latency(self, strategy: str, mode: str, duration: float):
        """记录查询总耗时（Histogram + P50/P95/P99 Gauge 窗口）。"""
        self._query_latency.labels(strategy=strategy, mode=mode).observe(duration)
        self._query_latency_pct.observe(duration, strategy=strategy, mode=mode)

    def record_retrieval(self, strategy: str, duration: float):
        """记录检索阶段耗时。"""
        self._retrieval_duration.labels(strategy=strategy).observe(duration)

    def record_embedding(self, strategy: str, duration: float):
        """记录 query embedding 阶段耗时。"""
        self._embedding_duration.labels(strategy=strategy).observe(duration)

    def record_vector(self, strategy: str, duration: float):
        """记录向量库检索阶段耗时。"""
        self._vector_duration.labels(strategy=strategy).observe(duration)

    def record_bm25(self, strategy: str, duration: float):
        """记录 BM25 打分排序阶段耗时。"""
        self._bm25_duration.labels(strategy=strategy).observe(duration)

    def record_sparse(self, strategy: str, duration: float):
        """记录稀疏检索耗时（BM25 / ES 后端共用，与具体后端无关）。"""
        self._sparse_duration.labels(strategy=strategy).observe(duration)

    def record_rrf(self, strategy: str, duration: float):
        """记录 RRF 融合阶段耗时。"""
        self._rrf_duration.labels(strategy=strategy).observe(duration)

    def record_rerank(self, duration: float):
        """记录重排阶段耗时。"""
        self._rerank_duration.observe(duration)

    def record_llm_ttft(self, backend: str, duration: float):
        """记录 LLM 首 token 延迟（TTFT）。"""
        self._llm_ttft.labels(backend=backend).observe(duration)

    def record_generation(self, backend: str, duration: float):
        """记录 LLM 生成总耗时。"""
        self._generation_duration.labels(backend=backend).observe(duration)

    # ---- 资源指标 ----
    def set_index_chunks(self, strategy: str, count: int):
        """设置索引 chunk 总数。"""
        self._index_chunks.labels(strategy=strategy).set(count)

    def set_active_tasks(self, count: int):
        """设置活跃后台任务数。"""
        self._active_tasks.set(count)

    # ---- 权限感知查询缓存指标 ----
    def record_cache_hit(self):
        """记录一次缓存命中。"""
        self._cache_hits.inc()

    def record_cache_miss(self):
        """记录一次缓存未命中。"""
        self._cache_misses.inc()

    def set_cache_entries(self, count: int):
        """设置缓存当前条目数。"""
        self._cache_entries.set(count)

    # ---- 导出 ----
    def export(self) -> tuple:
        """导出 Prometheus 格式文本。

        返回 (content_bytes, content_type)。
        """
        if not _PROMETHEUS_AVAILABLE:
            return (b"# prometheus_client not installed\n", "text/plain")
        return (generate_latest(), CONTENT_TYPE_LATEST)

    # ---- 结构化快照（后台监控页） ----
    def snapshot(self) -> dict:
        """返回当前指标的结构化快照（供后台监控页 JSON 渲染）。

        结构:
          {"available": true,
           "histograms": {metric_name: [{"labels": {...}, "count": n, "sum": s, "avg": a}]},
           "scalars": {metric_name: [{"labels": {...}, "value": v}]}}
        仅收集 rag_ 前缀指标；直方图省略 bucket 明细（保留 count/sum/avg）。
        """
        if not _PROMETHEUS_AVAILABLE:
            return {"available": False}
        from prometheus_client import REGISTRY
        histograms: dict = {}
        scalars: dict = {}
        for metric in REGISTRY.collect():
            name = metric.name
            if not name.startswith("rag_"):
                continue
            if metric.type == "histogram":
                groups = {}
                for s in metric.samples:
                    # 跳过 bucket 明细（count/sum 样本不带 le 标签，自成一组）
                    if s.name.endswith("_bucket"):
                        continue
                    labels = dict(s.labels or {})
                    key = tuple(sorted(labels.items()))
                    g = groups.setdefault(key, {"labels": labels, "count": 0, "sum": 0.0})
                    if s.name.endswith("_count"):
                        g["count"] = s.value
                    elif s.name.endswith("_sum"):
                        g["sum"] = s.value
                items = []
                for g in groups.values():
                    avg = g["sum"] / g["count"] if g["count"] else 0.0
                    items.append({
                        "labels": g["labels"],
                        "count": int(g["count"]),
                        "sum": round(g["sum"], 4),
                        "avg": round(avg, 4),
                    })
                if items:
                    histograms[name] = items
            else:
                vals = []
                for s in metric.samples:
                    vals.append({"labels": dict(s.labels or {}), "value": s.value})
                if vals:
                    scalars[name] = vals
        return {"available": True, "histograms": histograms, "scalars": scalars}


# 模块级单例
metrics = MetricsRegistry()
