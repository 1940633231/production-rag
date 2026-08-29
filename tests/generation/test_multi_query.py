r"""MultiQueryExpander 单元测试：LLM 输出解析 + 回退策略。

运行:
  .venv\Scripts\python.exe -m pytest tests\generation\test_multi_query.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.multi_query import MultiQueryExpander
from app.generation.generator import StubGenerator


class _MockGenerator:
    """返回预设输出的 mock 生成器。"""

    def __init__(self, output):
        self.output = output

    def generate(self, messages):
        return self.output


class _FailingGenerator:
    def generate(self, messages):
        raise RuntimeError("llm down")


class TestMultiQueryParse:
    def test_parse_strips_numbering_and_bullets(self):
        raw = "1. 铁矿近期供需情况\n2. 铁矿价格走势分析\n- 进口铁矿渠道"
        assert MultiQueryExpander._parse(raw) == [
            "铁矿近期供需情况", "铁矿价格走势分析", "进口铁矿渠道",
        ]

    def test_parse_skips_empty_and_duplicates(self):
        raw = "\n铁矿供需\n\n1. 铁矿供需\n（空白行已跳过）"
        result = MultiQueryExpander._parse(raw)
        assert result == ["铁矿供需", "（空白行已跳过）"]

    def test_parse_strips_quotes(self):
        raw = '"铁矿供需如何"'
        assert MultiQueryExpander._parse(raw) == ["铁矿供需如何"]


class TestMultiQueryExpand:
    def test_expand_includes_original_query(self):
        exp = MultiQueryExpander(_MockGenerator("近期供需\n价格走势"), num_queries=3)
        result = exp.expand("铁矿供需")
        assert result[0] == "铁矿供需"  # 原始 query 占首路
        assert len(result) == 3
        assert "近期供需" in result and "价格走势" in result

    def test_stub_generator_falls_back_to_single(self):
        exp = MultiQueryExpander(StubGenerator(), num_queries=3)
        assert exp.expand("铁矿供需") == ["铁矿供需"]

    def test_llm_failure_falls_back_to_single(self):
        exp = MultiQueryExpander(_FailingGenerator(), num_queries=3)
        assert exp.expand("铁矿供需") == ["铁矿供需"]

    def test_empty_llm_output_falls_back_to_single(self):
        exp = MultiQueryExpander(_MockGenerator("   \n\n  "), num_queries=3)
        assert exp.expand("铁矿供需") == ["铁矿供需"]

    def test_dedup_limits_to_num_queries(self):
        # LLM 输出 5 个，限到 num_queries 个（含原始）
        exp = MultiQueryExpander(_MockGenerator("a\nb\nc\nd\ne"), num_queries=3)
        result = exp.expand("原问题")
        assert len(result) == 3
        assert result == ["原问题", "a", "b"]

    def test_num_queries_one_disabled(self):
        exp = MultiQueryExpander(_MockGenerator("a\nb"), num_queries=1)
        assert exp.expand("原问题") == ["原问题"]
