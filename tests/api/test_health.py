r"""健康检查测试（#15）：响应结构 + 线程池并行不阻塞事件循环。

覆盖:
  - GET /api/health → 200 + 完整响应结构
  - 慢组件检查并行执行（串行需 0.9s，线程池并行 ≈0.3s）

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_health.py -v
"""
import time


def test_health_structure(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert data["llm_backend"] in ("stub", "qwen", "openai")
    # indexes 覆盖两个策略
    assert "fixed" in data["indexes"] and "recursive" in data["indexes"]
    # 各组件都有 status 字段
    for name, comp in data["components"].items():
        assert comp["status"] in ("ok", "degraded", "error", "disabled"), name


def test_health_checks_run_in_parallel(client, monkeypatch):
    """慢组件检查应在线程池并行执行（不串行阻塞事件循环）。"""
    import app.api.health as health

    from app.api.health import ComponentStatus

    def slow_check():
        time.sleep(0.3)
        return ComponentStatus(status="disabled", detail="mock-slow")

    # 三个网络组件检查都变慢
    for name in ("_check_mysql", "_check_elasticsearch", "_check_milvus"):
        monkeypatch.setattr(health, name, slow_check)

    start = time.time()
    resp = client.get("/api/health")
    elapsed = time.time() - start
    assert resp.status_code == 200
    # 串行执行需 3×0.3=0.9s，线程池并行 ≈0.3s；0.7s 足以区分
    assert elapsed < 0.7, "健康检查未并行执行（耗时 {:.2f}s）".format(elapsed)
