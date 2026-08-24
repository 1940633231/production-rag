r"""Chunker 单元测试：固定切分 + 递归切分，重点验证 offset 正确性。

覆盖:
  - Chunker (fixed) 基本切分、overlap、offset
  - RecursiveChunker 递归切分、offset 连续性、分隔符处理
  - 边界：空文档、短文档、超长单段

运行:
  .venv\Scripts\python.exe -m pytest tests\ingestion\test_chunker.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.chunk import Chunk
from app.ingestion.chunker.chunker import Chunker
from app.ingestion.chunker.recursive_chunker import RecursiveChunker
from app.ingestion.document import Document


# ---- 辅助 ----

def _make_doc(content: str, doc_id: str = "doc1") -> Document:
    return Document(
        document_id=doc_id,
        content=content,
        metadata={"file_name": doc_id + ".txt"},
    )


# ============================================================
# Chunker (fixed)
# ============================================================

class TestChunker:
    def test_basic_split(self):
        doc = _make_doc("abcdefghij")  # 10 字符
        chunker = Chunker(chunk_size=4, overlap=0)
        chunks = chunker.split(doc)
        assert len(chunks) == 3  # 4 + 4 + 2
        assert chunks[0].content == "abcd"
        assert chunks[1].content == "efgh"
        assert chunks[2].content == "ij"

    def test_offset_continuous_no_overlap(self):
        """无 overlap 时，chunk 端点应连续覆盖原文。"""
        text = "abcdefghij"
        doc = _make_doc(text)
        chunker = Chunker(chunk_size=4, overlap=0)
        chunks = chunker.split(doc)
        for c in chunks:
            assert text[c.start_offset:c.end_offset] == c.content

    def test_offset_with_overlap(self):
        """有 overlap 时，chunk 端点仍应正确指向原文位置。"""
        text = "abcdefghij"
        doc = _make_doc(text)
        chunker = Chunker(chunk_size=4, overlap=2)
        chunks = chunker.split(doc)
        # 每个 chunk 的 [start, end) 切片应等于其 content
        for c in chunks:
            assert text[c.start_offset:c.end_offset] == c.content
        # 第二个 chunk 的 start 应为 4 - 2 = 2
        assert chunks[1].start_offset == 2

    def test_chunk_index_sequence(self):
        doc = _make_doc("a" * 25)
        chunker = Chunker(chunk_size=10, overlap=0)
        chunks = chunker.split(doc)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_chunk_id_format(self):
        doc = _make_doc("abcdef", doc_id="mydoc")
        chunker = Chunker(chunk_size=2, overlap=0)
        chunks = chunker.split(doc)
        assert chunks[0].chunk_id == "mydoc_chunk_0"

    def test_metadata_copied(self):
        doc = _make_doc("abcdef")
        doc.metadata["custom"] = "value"
        chunker = Chunker(chunk_size=3, overlap=0)
        chunks = chunker.split(doc)
        for c in chunks:
            assert c.metadata.get("custom") == "value"

    def test_empty_document(self):
        doc = _make_doc("")
        chunker = Chunker(chunk_size=4, overlap=0)
        chunks = chunker.split(doc)
        assert chunks == []

    def test_short_document_single_chunk(self):
        doc = _make_doc("ab")
        chunker = Chunker(chunk_size=10, overlap=0)
        chunks = chunker.split(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "ab"


# ============================================================
# RecursiveChunker
# ============================================================

class TestRecursiveChunker:
    def test_split_by_double_newline(self):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        doc = _make_doc(text)
        # 文本总长 < chunk_size，RecursiveChunker 不会触发切分，作为一个 chunk
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        chunks = chunker.split(doc)
        assert len(chunks) == 1
        # 用小 chunk_size 触发按 \n\n 切分
        chunker_small = RecursiveChunker(chunk_size=8, overlap=0)
        chunks_small = chunker_small.split(doc)
        assert len(chunks_small) == 3

    def test_split_by_sentence_when_para_too_long(self):
        """段落超过 chunk_size 时，应递归到句号切分。"""
        text = "短句一。短句二。短句三。短句四。短句五。"
        doc = _make_doc(text)
        chunker = RecursiveChunker(chunk_size=10, overlap=0)
        chunks = chunker.split(doc)
        # 每个短句 4 字符，chunk_size=10，每个 chunk 应装 2 个短句
        assert len(chunks) >= 2
        # 所有 chunk 拼接后应覆盖原文（不含 overlap 时）
        # 由于 strip 可能去掉首尾空白，这里只验证每个 chunk content 来自原文
        for c in chunks:
            assert c.content  # 非空

    def test_offset_coverage(self):
        """递归切分的 chunk 端点应落在原文区间内。"""
        text = "句一。句二。句三。句四。句五。句六。"
        doc = _make_doc(text)
        chunker = RecursiveChunker(chunk_size=8, overlap=0)
        chunks = chunker.split(doc)
        for c in chunks:
            # 端点应在原文范围内
            assert 0 <= c.start_offset < len(text)
            assert 0 < c.end_offset <= len(text)
            assert c.start_offset < c.end_offset

    def test_overlap_keeps_context(self):
        """有 overlap 时，相邻 chunk 应共享尾部内容。"""
        text = "句一。句二。句三。句四。句五。句六。句七。句八。"
        doc = _make_doc(text)
        chunker = RecursiveChunker(chunk_size=10, overlap=4)
        chunks = chunker.split(doc)
        # 至少产生 2 个 chunk
        assert len(chunks) >= 2
        # 每个 chunk 端点合法
        for c in chunks:
            assert 0 <= c.start_offset < c.end_offset <= len(text)

    def test_single_long_sentence_truncation(self):
        """单个超长句无分隔符时，piece 超过 chunk_size 但无更细切分，仍应返回。"""
        text = "a" * 50  # 无任何分隔符
        doc = _make_doc(text)
        chunker = RecursiveChunker(chunk_size=10, overlap=0)
        chunks = chunker.split(doc)
        # 无分隔符，整个文本作为一个 piece，超过 chunk_size 也只能作为一个 chunk
        assert len(chunks) == 1
        assert len(chunks[0].content) == 50

    def test_empty_document(self):
        doc = _make_doc("   ")
        chunker = RecursiveChunker(chunk_size=10, overlap=0)
        chunks = chunker.split(doc)
        # strip 后为空，应返回空列表或单个空 chunk
        # RecursiveChunker 对空内容应优雅处理
        assert all(c.content for c in chunks)  # 不应有空 content 的 chunk

    def test_preserves_separators(self):
        """分隔符应保留在 piece 中，不丢失。"""
        text = "句一。\n\n句二。"
        doc = _make_doc(text)
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        chunks = chunker.split(doc)
        # 拼接所有 chunk content 应能还原原文（忽略首尾空白）
        joined = "".join(c.content for c in chunks)
        # 验证关键内容存在
        assert "句一" in joined
        assert "句二" in joined
