"""核心指标模块测试：分阶段耗时指标 + 查询总耗时 P50/P95/P99 分位数。"""
import re

import pytest

from app.core.metrics import metrics


def _export_text() -> str:
    content, _ = metrics.export()
    return content.decode("utf-8")


def test_embedding_metric_recorded():
    metrics.record_embedding("recursive", 0.02)
    text = _export_text()
    assert "rag_embedding_duration_seconds" in text
    assert 'strategy="recursive"' in text


def test_vector_metric_recorded():
    metrics.record_vector("recursive", 0.03)
    text = _export_text()
    assert "rag_vector_duration_seconds" in text
    assert 'strategy="recursive"' in text


def test_bm25_metric_recorded():
    metrics.record_bm25("recursive", 0.004)
    text = _export_text()
    assert "rag_bm25_duration_seconds" in text
    assert 'strategy="recursive"' in text


def test_sparse_metric_recorded():
    metrics.record_sparse("recursive", 0.004)
    text = _export_text()
    assert "rag_sparse_duration_seconds" in text
    assert 'strategy="recursive"' in text


def test_rrf_metric_recorded():
    metrics.record_rrf("recursive", 0.0001)
    text = _export_text()
    assert "rag_rrf_duration_seconds" in text
    assert 'strategy="recursive"' in text


def test_llm_ttft_metric_recorded():
    metrics.record_llm_ttft("QwenGenerator", 0.4)
    text = _export_text()
    assert "rag_llm_ttft_seconds" in text
    assert 'backend="QwenGenerator"' in text


def test_llm_total_metric_recorded():
    metrics.record_generation("QwenGenerator", 1.2)
    text = _export_text()
    assert "rag_generation_duration_seconds" in text
    assert 'backend="QwenGenerator"' in text


def test_query_latency_has_p50_p95_p99_gauges():
    # 多次观察产生分位数样本（P50/P95/P99 Gauge 随样本更新）
    for _ in range(20):
        metrics.record_query_latency("recursive", "hybrid", 1.5)
    text = _export_text()
    # 直方图 + 三个分位数 Gauge 都应暴露
    assert "rag_query_latency_seconds" in text
    assert "rag_query_latency_p50_seconds" in text
    assert "rag_query_latency_p95_seconds" in text
    assert "rag_query_latency_p99_seconds" in text
    assert 'strategy="recursive"' in text
    assert 'mode="hybrid"' in text


def test_snapshot_returns_histograms_and_scalars():
    metrics.record_embedding("recursive", 0.02)
    metrics.record_vector("recursive", 0.03)
    metrics.record_query_latency("recursive", "hybrid", 1.5)
    snap = metrics.snapshot()
    assert snap["available"] is True
    # 直方图快照：count/sum/avg 齐全（单例跨测试累计，故 count 断言 >=1，avg 恒 0.02）
    emb = snap["histograms"]["rag_embedding_duration_seconds"][0]
    assert emb["count"] >= 1
    assert emb["avg"] == pytest.approx(0.02, abs=1e-4)
    assert emb["labels"]["strategy"] == "recursive"
    # 分位数 Gauge 在 scalars 里
    p50 = snap["scalars"]["rag_query_latency_p50_seconds"][0]
    assert p50["value"] == pytest.approx(1.5, abs=1e-4)


def test_cache_entries_metric_refresh():
    """chat 端点刷缓存条目数指标：从 cache.stats() 写入 Gauge。"""
    from app.api import chat as chat_mod

    class FakeCache:
        def stats(self):
            return {"entries": 3}

    chat_mod._cache_entries_last[0] = 0.0
    chat_mod._refresh_cache_entries_metric(FakeCache())
    text = _export_text()
    assert "rag_query_cache_entries" in text
    assert re.search(r"rag_query_cache_entries\s+3\.0", text)


def test_cache_entries_refresh_swallows_stats_errors():
    """cache.stats() 抛异常时刷新静默降级（不中断请求）。"""
    from app.api import chat as chat_mod

    class BadCache:
        def stats(self):
            raise RuntimeError("redis down")

    chat_mod._cache_entries_last[0] = 0.0
    chat_mod._refresh_cache_entries_metric(BadCache())  # 不应抛异常
