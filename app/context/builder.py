"""Context Builder：去重、邻接合并、排序。

输入：检索/重排后的 List[dict]，每个 dict 至少含：
  - content: str
  - start_offset / end_offset: int
  - metadata: dict（含 document_id, chunk_index 等，chunk_index 可缺）
  - score 或 rerank_score: float（排序信号）

输出：处理后的 List[dict]，保持字段兼容；合并项额外带 merged=True 与 chunk_ids 列表。

设计要点（对应方案 2A/3/4）：
- 去重：同文档 span 重叠 > 阈值为主信号，跨文档用字符 n-gram Jaccard 兜底
- 邻接合并：同 document_id 且 chunk_index 连续或 span 端点贴合 → 合并 content 与 span
- 排序：默认 score（保留 rerank 序），可选 interleaved（缓解 lost-in-the-middle）
"""
from typing import Dict, List


def _get_score(r: Dict) -> float:
    """统一获取排序分数（rerank_score 优先，回退 score）。"""
    return r.get("rerank_score", r.get("score", 0.0))


def _span_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """计算 span a 与 b 的重叠比例（重叠长度 / 较小 span 长度）。"""
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    if overlap == 0:
        return 0.0
    min_len = min(a_end - a_start, b_end - b_start)
    return overlap / min_len if min_len > 0 else 0.0


def _jaccard_ngram(a: str, b: str, ngram: int = 3) -> float:
    """基于字符 n-gram 的 Jaccard 相似度，适合中文短文本。"""
    def ngrams(s: str) -> set:
        if len(s) < ngram:
            return {s}
        return {s[i:i + ngram] for i in range(len(s) - ngram + 1)}
    a_set, b_set = ngrams(a), ngrams(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


class ContextBuilder:
    """Context 构建器：去重 + 邻接合并 + 排序。

    所有方法均不修改输入，返回新列表，便于在 pipeline 中组合。
    """

    def __init__(
        self,
        dedup_span_overlap: float = 0.5,
        dedup_jaccard: float = 0.85,
        merge_span_gap: int = 5,
        ngram_size: int = 3,
    ):
        self.dedup_span_overlap = dedup_span_overlap
        self.dedup_jaccard = dedup_jaccard
        self.merge_span_gap = merge_span_gap
        self.ngram_size = ngram_size

    # ---- 去重 ----

    def deduplicate(self, results: List[Dict]) -> List[Dict]:
        """去重：同文档 span 重叠为主信号，跨文档 Jaccard 兜底。

        保留 score 较高的项，丢弃判定为重复的项。
        """
        # 按 score 降序，保证先入列的为高分项
        ordered = sorted(results, key=_get_score, reverse=True)
        kept: List[Dict] = []
        for r in ordered:
            is_dup = False
            for k in kept:
                if self._is_duplicate(r, k):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(r)
        return kept

    def _is_duplicate(self, a: Dict, b: Dict) -> bool:
        """判定 a 与 b 是否重复。"""
        a_doc = a.get("metadata", {}).get("document_id")
        b_doc = b.get("metadata", {}).get("document_id")
        # 同文档：span 重叠为主信号
        if a_doc and b_doc and a_doc == b_doc:
            ov = _span_overlap_ratio(
                a["start_offset"], a["end_offset"],
                b["start_offset"], b["end_offset"],
            )
            if ov >= self.dedup_span_overlap:
                return True
        # 跨文档或同文档 span 不重叠：Jaccard 兜底
        sim = _jaccard_ngram(a["content"], b["content"], self.ngram_size)
        return sim >= self.dedup_jaccard

    # ---- 邻接合并 ----

    def merge_neighbors(self, results: List[Dict]) -> List[Dict]:
        """邻接合并：同 document_id 且 chunk_index 连续或 span 端点贴合 → 合并。

        合并后 content 拼接、span 扩展为 [min(start), max(end)]、score 取较大。
        """
        # 按 document_id 分组
        groups: Dict[str, List[Dict]] = {}
        for r in results:
            doc = r.get("metadata", {}).get("document_id", "_unknown")
            groups.setdefault(doc, []).append(r)

        merged_all: List[Dict] = []
        for _, items in groups.items():
            merged_all.extend(self._merge_group(items))

        # 按"组代表 score"重新降序，保持高分在前
        merged_all.sort(key=_get_score, reverse=True)
        return merged_all

    def _merge_group(self, items: List[Dict]) -> List[Dict]:
        """对同一文档内的候选执行邻接合并。"""
        # 组内按 chunk_index 优先，否则按 start_offset 排序
        def sort_key(r: Dict):
            meta = r.get("metadata", {})
            idx = meta.get("chunk_index")
            return (idx if idx is not None else 10**9, r["start_offset"])

        ordered = sorted(items, key=sort_key)

        merged: List[Dict] = []
        for r in ordered:
            if not merged:
                merged.append(self._clone(r))
                continue
            last = merged[-1]
            if self._are_neighbors(last, r):
                self._merge_into(last, r)
            else:
                merged.append(self._clone(r))
        return merged

    def _are_neighbors(self, a: Dict, b: Dict) -> bool:
        """判定 a 与 b 是否邻接可合并（假设组内已排序，b 在 a 之后）。"""
        # chunk_index 连续
        a_idx = a.get("metadata", {}).get("chunk_index")
        b_idx = b.get("metadata", {}).get("chunk_index")
        if a_idx is not None and b_idx is not None:
            if abs(a_idx - b_idx) == 1:
                return True
        # span 端点贴合/小重叠兜底（fixed metadata 无 chunk_index）
        # 允许小范围重叠（负 gap）或贴合（小正 gap）
        gap = b["start_offset"] - a["end_offset"]
        return -self.merge_span_gap <= gap <= self.merge_span_gap

    def _clone(self, r: Dict) -> Dict:
        """复制为可变的合并容器。"""
        return {
            **r,
            "chunk_ids": [r.get("chunk_id", "")],
            "merged": False,
        }

    def _merge_into(self, target: Dict, src: Dict) -> None:
        """将 src 合并进 target（target 已 clone）。"""
        # content 拼接（组内已按 offset 排序，src 在后）
        target["content"] = target["content"] + src["content"]
        # span 扩展
        target["start_offset"] = min(target["start_offset"], src["start_offset"])
        target["end_offset"] = max(target["end_offset"], src["end_offset"])
        # score 取较大（写入 target 的主 score 字段）
        merged_score = max(_get_score(target), _get_score(src))
        if "rerank_score" in target:
            target["rerank_score"] = merged_score
        elif "score" in target:
            target["score"] = merged_score
        else:
            target["score"] = merged_score
        # chunk_id 累积
        target["chunk_ids"].append(src.get("chunk_id", ""))
        target["merged"] = True

    # ---- 排序 ----

    def order(self, results: List[Dict], strategy: str = "score") -> List[Dict]:
        """排序：score / document / interleaved。"""
        if strategy == "score":
            return sorted(results, key=_get_score, reverse=True)
        if strategy == "document":
            return sorted(results, key=lambda r: (
                r.get("metadata", {}).get("document_id", ""),
                r["start_offset"],
            ))
        if strategy == "interleaved":
            return self._interleave(results)
        raise ValueError(f"Unknown order strategy: {strategy}")

    def _interleave(self, results: List[Dict]) -> List[Dict]:
        """最高分放首位，次高分放末位，依次交替，缓解 lost-in-the-middle。"""
        ordered = sorted(results, key=_get_score, reverse=True)
        front, back = [], []
        for i, r in enumerate(ordered):
            if i % 2 == 0:
                front.append(r)
            else:
                back.append(r)
        # back 逆序，使次高分位于最末
        return front + back[::-1]
