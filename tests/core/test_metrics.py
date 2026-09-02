"""核心指标模块测试：分阶段耗时指标 + 查询总耗时 P50/P95/P99 分位数。"""
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
