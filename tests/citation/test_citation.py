r"""CitationExtractor 单元测试：引用格式鲁棒化。

覆盖:
  - 基本格式 [1] [2]
  - 逗号/顿号/中文括号分隔 [1,2] 【1、2】
  - 混合格式与去重保序
  - 空输入 / 编号超范围

运行:
  .venv\Scripts\python.exe -m pytest tests\citation\test_citation.py -v
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.citation.citation import CitationExtractor


def _chunk(chunk_id="c1", content="铁矿石供应端增产，需求端走弱。", score=0.9):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "start_offset": 0,
        "end_offset": len(content),
        "score": score,
        "metadata": {
            "document_id": "doc1",
            "file_name": "铁矿周报.txt",
        },
    }


def _extract(answer, n=3):
    chunks = [_chunk("c{}".format(i)) for i in range(n)]
    return CitationExtractor().extract(answer, chunks)


class TestCitationBasic:
    def test_single_bracket(self):
        citations = _extract("根据上下文[1]，铁矿石供给增加。")
        assert [c["number"] for c in citations] == [1]
        assert citations[0]["chunk_id"] == "c0"

    def test_adjacent_brackets(self):
        citations = _extract("供给增加[1]需求下降[2]。")
        assert [c["number"] for c in citations] == [1, 2]

    def test_no_citation(self):
        assert _extract("回答中没有任何引用标记。") == []

    def test_empty_answer(self):
        assert _extract("") == []


class TestCitationFormats:
    def test_comma_separated(self):
        """[1,2] 半角逗号分隔。"""
        citations = _extract("供需格局[1,2]偏宽松。")
        assert [c["number"] for c in citations] == [1, 2]

    def test_fullwidth_comma_separated(self):
        """[1，2] 全角逗号分隔。"""
        citations = _extract("供需格局[1，2]偏宽松。")
        assert [c["number"] for c in citations] == [1, 2]

    def test_dunhao_separated(self):
        """【1、2】顿号分隔 + 中文括号。"""
        citations = _extract("供需格局【1、2】偏宽松。")
        assert [c["number"] for c in citations] == [1, 2]

    def test_chinese_brackets(self):
        """【1】中文方括号。"""
        citations = _extract("根据【1】供需变化。")
        assert [c["number"] for c in citations] == [1]

    def test_mixed_formats(self):
        """混合格式 [1] 和【2】以及 [3,4]。"""
        citations = _extract("A[1]与B【2】以及C[3,4]。", n=5)
        assert [c["number"] for c in citations] == [1, 2, 3, 4]


class TestCitationDedupAndRange:
    def test_dedup_preserves_first_order(self):
        """[2][1][2] → 去重后 [2, 1]。"""
        citations = _extract("先引[2]，后引[1]，再引[2]。")
        assert [c["number"] for c in citations] == [2, 1]

    def test_out_of_range_skipped(self):
        """编号超出 chunks 范围应跳过，不影响其他引用。"""
        citations = _extract("有效[1]，无效[9]。", n=3)
        assert [c["number"] for c in citations] == [1]

    def test_citation_fields(self):
        """citation 应含 file_name/offset/content_preview 等字段。"""
        citations = _extract("内容[1]。")
        c = citations[0]
        assert c["file_name"] == "铁矿周报.txt"
        assert c["document_id"] == "doc1"
        assert c["content_preview"] == "铁矿石供应端增产，需求端走弱。"
