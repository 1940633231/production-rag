"""评估回归模块单元测试：基线存档/加载、对比与门禁、趋势追加。

运行:
  .venv\\Scripts\\python.exe -m pytest tests/evaluation/test_regression.py -v
"""
import csv

from app.evaluation.regression import (
    append_trend,
    compare_metrics,
    format_table,
    load_baseline,
    save_baseline,
)


def test_baseline_roundtrip(tmp_path):
    """save_baseline 后 load_baseline 应还原 metrics 与 config。"""
    metrics = {"recall@1": 0.8, "faithfulness": 0.9}
    config = {"strategy": "recursive", "mode": "hybrid"}
    p = save_baseline(str(tmp_path / "baseline.json"), metrics, config)
    loaded = load_baseline(str(p))
    assert loaded["metrics"] == metrics
    assert loaded["config"] == config
    assert "created_at" in loaded
    assert "commit" in loaded


def test_compare_all_pass():
    """当前指标未跌破基线-阈值 → 全部 PASS，delta 计算正确。"""
    current = {"recall@1": 0.85, "recall@3": 0.90}
    baseline = {"recall@1": 0.80, "recall@3": 0.95}
    thresholds = {"recall@1": 0.10, "recall@3": 0.10}
    rows = compare_metrics(current, baseline, thresholds)
    assert all(r["passed"] for r in rows)
    assert {r["metric"]: r["delta"] for r in rows} == {
        "recall@1": 0.05, "recall@3": -0.05,
    }


def test_compare_detects_regression():
    """跌破阈值 → 该行 FAIL，delta 为负。"""
    current = {"recall@1": 0.70}
    baseline = {"recall@1": 0.80}
    rows = compare_metrics(current, baseline, {"recall@1": 0.05})
    assert len(rows) == 1
    assert rows[0]["passed"] is False
    assert rows[0]["delta"] == -0.10


def test_compare_skips_missing_metrics():
    """当前结果缺少某指标时跳过该指标，不影响其他对比。"""
    rows = compare_metrics({"recall@1": 0.8}, {"recall@1": 0.8, "recall@3": 0.9})
    assert [r["metric"] for r in rows] == ["recall@1"]


def test_compare_default_thresholds():
    """不传 thresholds 时使用 DEFAULT_THRESHOLDS（与导出常量一致）。"""
    from app.evaluation.regression import DEFAULT_THRESHOLDS
    rows = compare_metrics(
        {"recall@1": 0.80 - DEFAULT_THRESHOLDS["recall@1"]},
        {"recall@1": 0.80},
    )
    # 恰好等于阈值边界：delta == -limit → 通过
    assert rows[0]["passed"] is True
    assert rows[0]["limit"] == DEFAULT_THRESHOLDS["recall@1"]


def test_append_trend_writes_header_and_rows(tmp_path):
    """trend 文件首次写入表头，再次追加不重复表头。"""
    p = tmp_path / "trend.csv"
    append_trend(str(p), {"recall@1": 0.8})
    append_trend(str(p), {"recall@1": 0.85})
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("timestamp,commit,recall@1")
    assert len(lines) == 3  # 表头 + 2 行数据
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(r["commit"] for r in rows)  # commit 列非空
    assert [r["recall@1"] for r in rows] == ["0.8", "0.85"]


def test_format_table_contains_header_and_verdict():
    rows = [
        {"metric": "recall@1", "baseline": 0.8, "current": 0.85,
         "delta": 0.05, "limit": 0.1, "passed": True},
        {"metric": "recall@3", "baseline": 0.9, "current": 0.8,
         "delta": -0.1, "limit": 0.05, "passed": False},
    ]
    table = format_table(rows)
    assert "指标" in table and "判定" in table
    assert "PASS" in table and "FAIL" in table
