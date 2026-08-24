r"""Context 模块单元测试：Builder 去重/合并/排序 + Manager 预算控制 + Compressor 压缩。

覆盖:
  - ContextBuilder.deduplicate: 同文档 span 重叠、跨文档 Jaccard
  - ContextBuilder.merge_neighbors: chunk_index 连续、span 贴合
  - ContextBuilder.order: score / document / interleaved
  - ContextCompressor.compress: 句子窗口截断
  - ContextManager.build: 端到端流程 + 预算控制

运行:
  .venv\Scripts\python.exe -m pytest tests\context\test_context.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.context.builder import ContextBuilder
from app.context.compressor import ContextCompressor
from app.context.manager import ContextManager
from app.ingestion.tokenizer import CharLengthCounter


# ---- 辅助 ----

def _chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    start: int,
    end: int,
    score: float = 0.5,
    chunk_index: int = 0,
):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "start_offset": start,
        "end_offset": end,
        "score": score,
        "metadata": {
            "document_id": doc_id,
            "chunk_index": chunk_index,
            "file_name": doc_id + ".txt",
        },
    }


# ============================================================
# ContextBuilder.deduplicate
# ============================================================

class TestDeduplicate:
    def test_same_doc_span_overlap_removed(self):
        """同文档 span 重叠超阈值应去重，保留高分项。"""
        builder = ContextBuilder(dedup_span_overlap=0.5)
        a = _chunk("a", "d1", "内容A", 0, 100, score=0.9, chunk_index=0)
        b = _chunk("b", "d1", "内容B", 50, 150, score=0.5, chunk_index=1)  # 与 a 重叠 50%
        result = builder.deduplicate([a, b])
        assert len(result) == 1
        assert result[0]["chunk_id"] == "a"  # 保留高分项

    def test_same_doc_no_overlap_kept(self):
        """同文档 span 不重叠应保留两者。"""
        builder = ContextBuilder(dedup_span_overlap=0.5)
        a = _chunk("a", "d1", "内容A", 0, 100, score=0.9, chunk_index=0)
        b = _chunk("b", "d1", "内容B", 200, 300, score=0.5, chunk_index=2)
        result = builder.deduplicate([a, b])
        assert len(result) == 2

    def test_cross_doc_jaccard_duplicate_removed(self):
        """跨文档内容高度相似（Jaccard）应去重。"""
        builder = ContextBuilder(dedup_jaccard=0.85)
        content = "这是一段完全相同的中文内容用于测试去重"
        a = _chunk("a", "d1", content, 0, 20, score=0.9)
        b = _chunk("b", "d2", content, 0, 20, score=0.5)  # 跨文档但内容相同
        result = builder.deduplicate([a, b])
        assert len(result) == 1
        assert result[0]["chunk_id"] == "a"

    def test_cross_doc_different_kept(self):
        """跨文档内容不同应保留。"""
        builder = ContextBuilder()
        a = _chunk("a", "d1", "完全不同的内容甲", 0, 10, score=0.9)
        b = _chunk("b", "d2", "完全不同的内容乙", 0, 10, score=0.5)
        result = builder.deduplicate([a, b])
        assert len(result) == 2

    def test_empty_input(self):
        builder = ContextBuilder()
        assert builder.deduplicate([]) == []


# ============================================================
# ContextBuilder.merge_neighbors
# ============================================================

class TestMergeNeighbors:
    def test_consecutive_chunk_index_merged(self):
        """同文档 chunk_index 连续应合并。"""
        builder = ContextBuilder()
        a = _chunk("a", "d1", "前半段", 0, 50, score=0.8, chunk_index=0)
        b = _chunk("b", "d1", "后半段", 50, 100, score=0.6, chunk_index=1)
        result = builder.merge_neighbors([a, b])
        assert len(result) == 1
        assert result[0]["merged"] is True
        assert result[0]["content"] == "前半段后半段"
        assert result[0]["start_offset"] == 0
        assert result[0]["end_offset"] == 100
        assert set(result[0]["chunk_ids"]) == {"a", "b"}

    def test_non_consecutive_not_merged(self):
        """chunk_index 不连续不应合并。"""
        builder = ContextBuilder()
        a = _chunk("a", "d1", "段一", 0, 50, score=0.8, chunk_index=0)
        b = _chunk("b", "d1", "段三", 100, 150, score=0.6, chunk_index=2)
        result = builder.merge_neighbors([a, b])
        assert len(result) == 2

    def test_span_gap_merged(self):
        """无 chunk_index 但 span 端点贴合应合并。"""
        builder = ContextBuilder(merge_span_gap=5)
        a = _chunk("a", "d1", "前", 0, 50, score=0.8, chunk_index=0)
        b = _chunk("b", "d1", "后", 53, 100, score=0.6, chunk_index=5)  # gap=3
        result = builder.merge_neighbors([a, b])
        assert len(result) == 1

    def test_different_docs_not_merged(self):
        """不同文档不应合并。"""
        builder = ContextBuilder()
        a = _chunk("a", "d1", "前", 0, 50, chunk_index=0)
        b = _chunk("b", "d2", "后", 50, 100, chunk_index=1)
        result = builder.merge_neighbors([a, b])
        assert len(result) == 2


# ============================================================
# ContextBuilder.order
# ============================================================

class TestOrder:
    def test_score_order(self):
        builder = ContextBuilder()
        a = _chunk("a", "d1", "A", 0, 1, score=0.3)
        b = _chunk("b", "d1", "B", 1, 2, score=0.9)
        c = _chunk("c", "d1", "C", 2, 3, score=0.6)
        result = builder.order([a, b, c], "score")
        assert [r["chunk_id"] for r in result] == ["b", "c", "a"]

    def test_document_order(self):
        builder = ContextBuilder()
        a = _chunk("a", "d2", "A", 5, 10)
        b = _chunk("b", "d1", "B", 0, 5)
        c = _chunk("c", "d1", "C", 10, 15)
        result = builder.order([a, b, c], "document")
        # d1 在前，d1 内按 start_offset 排序
        assert [r["metadata"]["document_id"] for r in result] == ["d1", "d1", "d2"]

    def test_interleaved_order(self):
        """interleaved: 最高分首位，次高分末位。"""
        builder = ContextBuilder()
        a = _chunk("a", "d1", "A", 0, 1, score=0.9)
        b = _chunk("b", "d1", "B", 1, 2, score=0.6)
        c = _chunk("c", "d1", "C", 2, 3, score=0.3)
        result = builder.order([a, b, c], "interleaved")
        # 排序后 [a(0.9), b(0.6), c(0.3)]，偶数位 front=[a,c]，奇数位 back=[b]
        # 最终 = front + back[::-1] = [a, c] + [b] = [a, c, b]
        assert result[0]["chunk_id"] == "a"  # 最高分首位
        assert result[-1]["chunk_id"] == "b"  # 次高分末位

    def test_unknown_strategy_raises(self):
        builder = ContextBuilder()
        with pytest.raises(ValueError):
            builder.order([], "unknown")


# ============================================================
# ContextCompressor
# ============================================================

class TestCompressor:
    def test_no_compression_when_under_limit(self):
        counter = CharLengthCounter()
        comp = ContextCompressor(counter)
        chunk = _chunk("a", "d1", "短内容", 0, 3)
        result = comp.compress(chunk, max_tokens=100)
        assert result is chunk  # 未超限，原样返回
        assert "compressed" not in result

    def test_compression_truncates(self):
        counter = CharLengthCounter()
        comp = ContextCompressor(counter)
        content = "第一句。第二句。第三句。"
        chunk = _chunk("a", "d1", content, 0, len(content))
        result = comp.compress(chunk, max_tokens=6)  # 只能装 6 字符
        assert result.get("compressed") is True
        assert counter.count(result["content"]) <= 6
        assert result["original_tokens"] == len(content)

    def test_compress_all(self):
        counter = CharLengthCounter()
        comp = ContextCompressor(counter)
        chunks = [
            _chunk("a", "d1", "短", 0, 1),
            _chunk("b", "d1", "超长内容超长内容超长内容", 1, 12),
        ]
        results = comp.compress_all(chunks, max_tokens_per_chunk=5)
        assert results[0] is chunks[0]  # 短的不压缩
        assert results[1].get("compressed") is True

    def test_truncate_to_tokens(self):
        counter = CharLengthCounter()
        comp = ContextCompressor(counter)
        assert comp._truncate_to_tokens("abcdef", 3) == "abc"
        assert comp._truncate_to_tokens("abcdef", 10) == "abcdef"
        assert comp._truncate_to_tokens("abcdef", 0) == ""


# ============================================================
# ContextManager.build 端到端
# ============================================================

class TestContextManager:
    def test_build_full_pipeline(self):
        """端到端：去重 → 合并 → 压缩 → 排序 → 预算装填。"""
        counter = CharLengthCounter()
        manager = ContextManager(
            token_counter=counter,
            max_context_tokens=1000,
            reserved_tokens=100,
            order_strategy="score",
        )
        chunks = [
            _chunk("a", "d1", "内容A", 0, 50, score=0.9, chunk_index=0),
            _chunk("b", "d1", "内容B", 50, 100, score=0.6, chunk_index=1),
            _chunk("c", "d2", "内容C", 0, 50, score=0.3, chunk_index=0),
        ]
        result = manager.build("query", chunks)
        assert "context" in result
        assert "chunks" in result
        assert "stats" in result
        assert result["stats"]["input_count"] == 3
        # a 和 b 邻接合并
        assert result["stats"]["after_merge"] <= 2
        # context 带编号
        assert "[1]" in result["context"]

    def test_budget_truncation(self):
        """预算不足时应截断。"""
        counter = CharLengthCounter()
        manager = ContextManager(
            token_counter=counter,
            max_context_tokens=20,  # 极小预算
            reserved_tokens=5,
            order_strategy="score",
        )
        chunks = [
            _chunk("a", "d1", "很长很长的内容" * 10, 0, 80, score=0.9),
            _chunk("b", "d2", "另一段长内容" * 10, 0, 80, score=0.5),
        ]
        result = manager.build("q", chunks)
        # 预算 20 - 5 - 1(query) = 14，只能装很少内容
        assert result["stats"]["used_tokens"] <= 14
        assert result["stats"]["budget_utilization"] <= 1.0

    def test_empty_input(self):
        counter = CharLengthCounter()
        manager = ContextManager(token_counter=counter)
        result = manager.build("query", [])
        assert result["context"] == ""
        assert result["chunks"] == []
        assert result["stats"]["input_count"] == 0
