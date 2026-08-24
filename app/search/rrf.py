from typing import Dict, List


def rrf_fuse(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion：多路检索结果融合。

    公式：score(d) = Σ_i 1 / (k + rank_i(d))
    - k：平滑常数（默认 60，工业经验值），削弱 top-1 过大权重
    - rank_i(d)：文档 d 在第 i 路结果中的 1-based 排名
    - 未出现在某路时该项贡献为 0

    优势：用 rank 而非原始分数，天然规避 dense(余弦∈[0,1]) 与 sparse(BM25∈[0,∞)) 分数尺度不一致问题。

    ranked_lists: 多路检索结果，每路是 List[dict]，dict 需含 chunk_id（用于去重融合）
    return: 融合后按 RRF 分数降序的结果列表，保留首份出现的 chunk 元数据（含 offset）
    """

    scores: Dict[str, float] = {}
    payload: Dict[str, Dict] = {}

    for ranked in ranked_lists:

        for idx, item in enumerate(ranked):

            cid = item["chunk_id"]

            rank = idx + 1  # 1-based rank

            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

            # 首次出现时记录其字段（content/start_offset/end_offset/metadata）
            if cid not in payload:
                payload[cid] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []

    for rank, (cid, score) in enumerate(fused):

        item = dict(payload[cid])
        item["score"] = float(score)
        item["rank"] = rank
        results.append(item)

    return results
