from typing import List, Dict


def recall_at_k(
    retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int
) -> float:

    retrieved = set(retrieved_chunk_ids[:k])
    relevant = set(relevant_chunk_ids)

    if not relevant:
        return 0.0

    hit = retrieved.intersection(relevant)

    return len(hit) / len(relevant)


def recall_at_k_spans(
    retrieved_chunks: List[Dict], relevant_spans: List[Dict], k: int
) -> float:
    """span 级 Recall@k。

    retrieved_chunks: 每个 chunk 需含 start_offset / end_offset
    relevant_spans:    [{"start": int, "end": int}, ...]，偏移相对文档 clean 后的 content

    命中规则：chunk 区间 [start_offset, end_offset) 与任一 relevant span 区间有交集，
    即视为该 relevant span 被命中。
    """

    if not relevant_spans:
        return 0.0

    relevant = [(s["start"], s["end"]) for s in relevant_spans]

    # 标记每个 relevant span 是否被 top-k 任一 chunk 命中（去重，避免多 chunk 命中同一 span 导致 recall > 1）
    hit_flags = [False] * len(relevant)

    for chunk in retrieved_chunks[:k]:

        c_start = chunk.get("start_offset", 0)
        c_end = chunk.get("end_offset", 0)

        for i, (r_start, r_end) in enumerate(relevant):

            if not hit_flags[i] and max(c_start, r_start) < min(c_end, r_end):

                hit_flags[i] = True

    return sum(hit_flags) / len(relevant)
