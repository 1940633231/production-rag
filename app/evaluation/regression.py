"""评估回归：基线存档 / 对比门禁 / 趋势沉淀。

评估闭环的核心逻辑（配合 scripts/evaluate.py 使用）：
  1. save_baseline  将一次完整评估（检索 Recall + 生成 Faithfulness/Relevance）存档为基线
  2. compare_metrics 当前结果 vs 基线，逐指标计算 delta 并判定是否跌破阈值
  3. format_table / all_passed 输出人类可读对比表；任一指标跌破阈值 → 门禁失败
  4. append_trend   追加趋势 CSV（每次对比一行，沉淀指标历史）

用法：
  python scripts/evaluate.py --save-baseline reports/baseline.json
  python scripts/evaluate.py --compare reports/baseline.json
"""
import csv
import datetime
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)

# 每个指标允许的最大降幅（阈值越小越严格）。
# 注意：当前数据集仅 2 条 query，单条 query 翻转即造成 0.5 级波动，
# 数据集扩充后应相应收紧阈值。
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "recall@1": 0.10,
    "recall@3": 0.10,
    "recall@5": 0.10,
    "recall@10": 0.10,
    "mrr@10": 0.10,
    "ndcg@10": 0.05,
    "faithfulness": 0.05,
    "relevance": 0.05,
}


def current_commit() -> str:
    """返回当前 git 短 commit（失败时回退 'unknown'）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_baseline(path: str, metrics: Dict[str, float], config_info: dict) -> Path:
    """将一次评估的指标存档为基线 JSON。

    baseline.json 结构:
        {created_at, commit, config: {...}, metrics: {...}}
    """
    baseline = {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": current_commit(),
        "config": config_info,
        "metrics": metrics,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("基线已保存: %s (metrics=%s)", p, metrics)
    return p


def load_baseline(path: str) -> dict:
    """读取基线 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_metrics(
    current: Dict[str, float],
    baseline_metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """逐指标对比当前结果与基线。

    返回行列表，每行:
        {metric, baseline, current, delta, limit, passed}
    - delta = current - baseline（负值表示回退）
    - passed = delta >= -limit（未跌破阈值即通过）
    - 仅对比当前与基线都存在的指标；当前缺失基线指标时跳过
    """
    limits = thresholds or DEFAULT_THRESHOLDS
    rows = []
    for metric, cur in sorted(current.items()):
        base = baseline_metrics.get(metric)
        if base is None:
            continue
        limit = limits.get(metric)
        delta = round(cur - base, 4)
        passed = limit is None or delta >= -limit
        rows.append({
            "metric": metric,
            "baseline": base,
            "current": cur,
            "delta": delta,
            "limit": limit,
            "passed": passed,
        })
    return rows


def format_table(rows: List[dict]) -> str:
    """生成人类可读对比表。"""
    if not rows:
        return "（无可用指标进行对比）"
    headers = ["指标", "基线", "本次", "Δ", "阈值", "判定"]
    data = [
        [r["metric"], str(r["baseline"]), str(r["current"]),
         "{:+.4f}".format(r["delta"]), str(r["limit"]),
         "PASS" if r["passed"] else "FAIL"]
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in data:
        lines.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def all_passed(rows: List[dict]) -> bool:
    """门禁判定：所有指标均未跌破阈值。"""
    return all(r["passed"] for r in rows)


def append_trend(path: str, metrics: Dict[str, float]) -> Path:
    """追加一行趋势 CSV（列：timestamp, commit, 各指标）。

    文件不存在时自动写表头；每次对比调用一次。
    """
    row = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": current_commit(),
        **metrics,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("趋势已追加: %s (%d 列)", p, len(row))
    return p
