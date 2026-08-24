"""检索评估标准指标：Precision@k / MRR / NDCG（含 chunk-id 版与 span 版）。

与 retrieval_eval.py 中 recall_at_k_spans 的标注约定一致：
    relevant_spans: [{"start": int, "end": int}, ...]，偏移相对文档 clean 后的 content
    retrieved_chunks: 每个 chunk 需含 start_offset / end_offset

设计要点:
    - 全部纯 Python 实现，零第三方依赖
    - 命中规则：chunk 区间 [start_offset, end_offset) 与任一 relevant span 区间有交集即命中
    - span 版指标对每个 relevant span 只计一次命中（去重，避免 recall > 1）
    - 支持多文档场景：span 版按 relevant_spans 全局计算
"""
import math
from typing import Dict, List


# ============================================================
# Chunk-ID 版（基于 chunk_id 集合判定相关性）
# ============================================================

def precision_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """Precision@k：top-k 中相关项的比例。

    参数:
        retrieved_ids: 检索返回的 chunk_id 列表（按相关度降序）
        relevant_ids: 标注为相关的 chunk_id 列表
        k: 截断位置

    返回:
        0.0~1.0；若 k<=0 返回 0.0
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    retrieved = retrieved_ids[:k]
    if not retrieved:
        return 0.0
    hit = sum(1 for cid in retrieved if cid in relevant_set)
    return hit / len(retrieved)


def mrr(
    retrieved_ids: List[str], relevant_ids: List[str]
) -> float:
    """Mean Reciprocal Rank：第一个相关项的倒数排名。

    参数:
        retrieved_ids: 检索返回的 chunk_id 列表（按相关度降序）
        relevant_ids: 标注为相关的 chunk_id 列表

    返回:
        0.0~1.0；若无命中返回 0.0
    """
    relevant_set = set(relevant_ids)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> float:
    """NDCG@k：归一化折损累积增益。

    采用二值相关性：相关=1，不相关=0。
    DCG@k  = Σ_{i=1}^{k} rel_i / log2(i+1)
    IDCG@k = 理想排序下的 DCG（min(k, |relevant|) 个相关项排在最前）
    NDCG@k = DCG@k / IDCG@k

    参数:
        retrieved_ids: 检索返回的 chunk_id 列表
        relevant_ids: 标注为相关的 chunk_id 列表
        k: 截断位置

    返回:
        0.0~1.0；若 relevant_ids 为空返回 0.0
    """
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0

    retrieved = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG：理想排序下 min(k, |relevant|) 个相关项排在最前
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================
# Span 版（基于字符偏移区间判定相关性，与项目约定一致）
# ============================================================

def _span_intersects(a_start, a_end, b_start, b_end) -> bool:
    """区间 [a_start, a_end) 与 [b_start, b_end) 是否有交集。"""
    return max(a_start, b_start) < min(a_end, b_end)


def _hit_flags_by_spans(
    retrieved_chunks: List[Dict], relevant_spans: List[Dict], k: int
) -> List[bool]:
    """返回每个 relevant span 是否被 top-k 任一 chunk 命中。"""
    relevant = [(s["start"], s["end"]) for s in relevant_spans]
    hit_flags = [False] * len(relevant)
    for chunk in retrieved_chunks[:k]:
        c_start = chunk.get("start_offset", 0)
        c_end = chunk.get("end_offset", 0)
        for i, (r_start, r_end) in enumerate(relevant):
            if not hit_flags[i] and _span_intersects(c_start, c_end, r_start, r_end):
                hit_flags[i] = True
    return hit_flags


def precision_at_k_spans(
    retrieved_chunks: List[Dict], relevant_spans: List[Dict], k: int
) -> float:
    """span 级 Precision@k：top-k 中命中相关 span 的 chunk 数 / k。

    注意: 分母为 k（检索返回数），而非 relevant_spans 数量，符合 Precision 定义。
    多个 chunk 命中同一 span 不重复计分（每个 chunk 独立判断是否命中任一 span）。
    """
    if k <= 0:
        return 0.0
    relevant = [(s["start"], s["end"]) for s in relevant_spans]
    if not relevant:
        return 0.0

    retrieved = retrieved_chunks[:k]
    hit_count = 0
    for chunk in retrieved:
        c_start = chunk.get("start_offset", 0)
        c_end = chunk.get("end_offset", 0)
        if any(_span_intersects(c_start, c_end, r_start, r_end) for r_start, r_end in relevant):
            hit_count += 1
    return hit_count / len(retrieved)


def mrr_spans(
    retrieved_chunks: List[Dict], relevant_spans: List[Dict]
) -> float:
    """span 级 MRR：第一个命中任一 relevant span 的 chunk 的倒数排名。"""
    relevant = [(s["start"], s["end"]) for s in relevant_spans]
    if not relevant:
        return 0.0

    for i, chunk in enumerate(retrieved_chunks, start=1):
        c_start = chunk.get("start_offset", 0)
        c_end = chunk.get("end_offset", 0)
        if any(_span_intersects(c_start, c_end, r_start, r_end) for r_start, r_end in relevant):
            return 1.0 / i
    return 0.0


def ndcg_at_k_spans(
    retrieved_chunks: List[Dict], relevant_spans: List[Dict], k: int
) -> float:
    """span 级 NDCG@k。

    理想排序：所有 relevant span 都被命中（每个 span 由独立 chunk 命中），
    ideal_hits = min(k, len(relevant_spans))。
    """
    if k <= 0:
        return 0.0
    if not relevant_spans:
        return 0.0

    retrieved = retrieved_chunks[:k]
    relevant = [(s["start"], s["end"]) for s in relevant_spans]

    # DCG: 每个 chunk 若命中任一 relevant span，rel=1
    dcg = 0.0
    for i, chunk in enumerate(retrieved, start=1):
        c_start = chunk.get("start_offset", 0)
        c_end = chunk.get("end_offset", 0)
        if any(_span_intersects(c_start, c_end, r_start, r_end) for r_start, r_end in relevant):
            dcg += 1.0 / math.log2(i + 1)

    # IDCG: 理想情况下前 min(k, |relevant|) 个位置全命中
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================
# 汇总：一次计算多指标
# ============================================================

def evaluate_retrieval(
    retrieved_ids: List[str], relevant_ids: List[str], k: int = 5
) -> Dict[str, float]:
    """一次性输出 chunk-id 版 Recall/Precision/MRR/NDCG@k。"""
    from app.evaluation.retrieval_eval import recall_at_k
    return {
        "recall@k": recall_at_k(retrieved_ids, relevant_ids, k),
        "precision@k": precision_at_k(retrieved_ids, relevant_ids, k),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "ndcg@k": ndcg_at_k(retrieved_ids, relevant_ids, k),
    }


def evaluate_retrieval_spans(
    retrieved_chunks: List[Dict], relevant_spans: List[Dict], k: int = 5
) -> Dict[str, float]:
    """一次性输出 span 版 Recall/Precision/MRR/NDCG@k。"""
    from app.evaluation.retrieval_eval import recall_at_k_spans
    return {
        "recall@k": recall_at_k_spans(retrieved_chunks, relevant_spans, k),
        "precision@k": precision_at_k_spans(retrieved_chunks, relevant_spans, k),
        "mrr": mrr_spans(retrieved_chunks, relevant_spans),
        "ndcg@k": ndcg_at_k_spans(retrieved_chunks, relevant_spans, k),
    }
