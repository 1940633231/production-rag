r"""评估模块单元测试：metrics.py (Precision/MRR/NDCG) + generation_eval。

覆盖:
  - metrics.py: chunk-id 版 + span 版各指标
  - generation_eval.py: FaithfulnessEvaluator / RelevanceEvaluator（用 mock judge）

运行:
  .venv\Scripts\python.exe -m pytest tests\evaluation\test_eval.py -v
"""
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.metrics import (
    evaluate_retrieval,
    evaluate_retrieval_spans,
    mrr,
    mrr_spans,
    ndcg_at_k,
    ndcg_at_k_spans,
    precision_at_k,
    precision_at_k_spans,
)


# ============================================================
# Chunk-ID 版指标
# ============================================================

class TestChunkIdMetrics:
    def test_precision_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        relevant = ["c2", "c3"]
        # top-5 中有 2 个相关，precision = 2/5
        assert precision_at_k(retrieved, relevant, k=5) == 0.4

    def test_precision_at_k_smaller_k(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c3"]
        # top-2 中 0 个相关
        assert precision_at_k(retrieved, relevant, k=2) == 0.0
        # top-3 中 1 个相关
        assert precision_at_k(retrieved, relevant, k=3) == 1 / 3

    def test_precision_at_k_empty_relevant(self):
        # 无相关项时，precision = 0/3 = 0（分母为 retrieved 数）
        assert precision_at_k(["c1", "c2"], [], k=2) == 0.0

    def test_precision_at_k_k_zero(self):
        assert precision_at_k(["c1"], ["c1"], k=0) == 0.0

    def test_mrr_first_hit_at_position_1(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c1"]
        assert mrr(retrieved, relevant) == 1.0

    def test_mrr_first_hit_at_position_3(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c3"]
        assert mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_mrr_no_hit(self):
        retrieved = ["c1", "c2"]
        relevant = ["c9"]
        assert mrr(retrieved, relevant) == 0.0

    def test_ndcg_ideal_order(self):
        """理想排序（相关项在最前）NDCG=1.0。"""
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c1", "c2"]
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_ndcg_suboptimal_order(self):
        """非理想排序 NDCG < 1.0。"""
        retrieved = ["c3", "c2", "c1"]  # 相关项在第 2、3 位
        relevant = ["c1", "c2"]
        # DCG = 1/log2(3) + 1/log2(4) = 0.6309 + 0.5 = 1.1309
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        # NDCG = 1.1309 / 1.6309 ≈ 0.693
        result = ndcg_at_k(retrieved, relevant, k=3)
        assert 0 < result < 1.0
        assert result == pytest.approx(1.1309 / 1.6309, rel=1e-3)

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k(["c1"], [], k=1) == 0.0

    def test_evaluate_retrieval_summary(self):
        retrieved = ["c1", "c2", "c3"]
        relevant = ["c1", "c3"]
        result = evaluate_retrieval(retrieved, relevant, k=3)
        assert set(result.keys()) == {"recall@k", "precision@k", "mrr", "ndcg@k"}
        assert result["recall@k"] == 1.0
        assert result["precision@k"] == pytest.approx(2 / 3)
        assert result["mrr"] == 1.0


# ============================================================
# Span 版指标
# ============================================================

class TestSpanMetrics:
    def _chunks(self):
        """构造检索 chunks，每个含 start_offset/end_offset。"""
        return [
            {"chunk_id": "c1", "content": "A", "start_offset": 0, "end_offset": 10},
            {"chunk_id": "c2", "content": "B", "start_offset": 50, "end_offset": 60},
            {"chunk_id": "c3", "content": "C", "start_offset": 100, "end_offset": 110},
        ]

    def _spans(self):
        return [{"start": 5, "end": 15}, {"start": 105, "end": 115}]

    def test_precision_at_k_spans(self):
        chunks = self._chunks()
        spans = self._spans()
        # c1 [0,10) 与 span1 [5,15) 相交 → 命中
        # c2 [50,60) 与两 span 不相交
        # c3 [100,110) 与 span2 [105,115) 相交 → 命中
        # top-3 precision = 2/3
        assert precision_at_k_spans(chunks, spans, k=3) == pytest.approx(2 / 3)

    def test_precision_at_k_spans_smaller_k(self):
        chunks = self._chunks()
        spans = self._spans()
        # top-1 只看 c1，命中 span1，precision = 1/1
        assert precision_at_k_spans(chunks, spans, k=1) == 1.0

    def test_mrr_spans(self):
        chunks = self._chunks()
        spans = self._spans()
        # c1 在第 1 位即命中 span1
        assert mrr_spans(chunks, spans) == 1.0

    def test_mrr_spans_second_position(self):
        chunks = [
            {"start_offset": 200, "end_offset": 210},  # 不命中
            {"start_offset": 5, "end_offset": 15},     # 命中 span1
        ]
        spans = [{"start": 5, "end": 15}]
        assert mrr_spans(chunks, spans) == 0.5

    def test_ndcg_at_k_spans_ideal(self):
        """理想排序：每个 chunk 都命中，且按命中顺序排列。"""
        # 两个 chunk 命中两个 span，顺序与 span 一致
        chunks = [
            {"start_offset": 0, "end_offset": 10},    # 命中 span1
            {"start_offset": 100, "end_offset": 110},  # 命中 span2
        ]
        spans = [{"start": 0, "end": 10}, {"start": 100, "end": 110}]
        result = ndcg_at_k_spans(chunks, spans, k=2)
        assert result == 1.0

    def test_ndcg_at_k_spans_suboptimal(self):
        chunks = [
            {"start_offset": 50, "end_offset": 60},    # 不命中
            {"start_offset": 0, "end_offset": 10},     # 命中 span1
            {"start_offset": 100, "end_offset": 110},  # 命中 span2
        ]
        spans = [{"start": 0, "end": 10}, {"start": 100, "end": 110}]
        # 相关项在第 2、3 位，NDCG < 1
        result = ndcg_at_k_spans(chunks, spans, k=3)
        assert 0 < result < 1.0

    def test_empty_spans(self):
        chunks = self._chunks()
        assert precision_at_k_spans(chunks, [], k=3) == 0.0
        assert mrr_spans(chunks, []) == 0.0
        assert ndcg_at_k_spans(chunks, [], k=3) == 0.0

    def test_evaluate_retrieval_spans_summary(self):
        chunks = self._chunks()
        spans = self._spans()
        result = evaluate_retrieval_spans(chunks, spans, k=3)
        assert set(result.keys()) == {"recall@k", "precision@k", "mrr", "ndcg@k"}
        # span1 由 c1 命中，span2 由 c3 命中 → recall = 2/2 = 1.0
        assert result["recall@k"] == 1.0


# ============================================================
# generation_eval（用 mock judge，不调真实 LLM）
# ============================================================

class TestFaithfulnessEvaluator:
    def _mock_judge(self, extraction_response, verification_response):
        """构造 mock judge，按调用顺序返回不同响应。"""
        from app.generation.generator import BaseGenerator

        class MockJudge(BaseGenerator):
            def __init__(self):
                self.calls = 0
                self._ext = extraction_response
                self._ver = verification_response

            def generate(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return self._ext
                return self._ver
        return MockJudge()

    def test_all_supported(self):
        import json
        ext = json.dumps(["铁矿石供应增加", "铁矿石需求下降"], ensure_ascii=False)
        ver = json.dumps([
            {"claim": "铁矿石供应增加", "verdict": "supported", "evidence": "上下文提及"},
            {"claim": "铁矿石需求下降", "verdict": "supported", "evidence": "上下文提及"},
        ], ensure_ascii=False)
        judge = self._mock_judge(ext, ver)
        from app.evaluation.generation_eval import FaithfulnessEvaluator
        evaluator = FaithfulnessEvaluator(judge)
        result = evaluator.evaluate("query", "answer", "context")
        assert result["score"] == 1.0
        assert result["total_claims"] == 2
        assert result["supported_claims"] == 2

    def test_partial_supported(self):
        import json
        ext = json.dumps(["声明一", "声明二"], ensure_ascii=False)
        ver = json.dumps([
            {"claim": "声明一", "verdict": "supported", "evidence": "有"},
            {"claim": "声明二", "verdict": "partial", "evidence": "部分"},
        ], ensure_ascii=False)
        judge = self._mock_judge(ext, ver)
        from app.evaluation.generation_eval import FaithfulnessEvaluator
        evaluator = FaithfulnessEvaluator(judge)
        result = evaluator.evaluate("query", "answer", "context")
        # partial 算 0.5：score = (1 + 0.5) / 2 = 0.75
        assert result["score"] == 0.75

    def test_all_unsupported(self):
        import json
        ext = json.dumps(["声明一", "声明二"], ensure_ascii=False)
        ver = json.dumps([
            {"claim": "声明一", "verdict": "unsupported", "evidence": "无依据"},
            {"claim": "声明二", "verdict": "unsupported", "evidence": "无依据"},
        ], ensure_ascii=False)
        judge = self._mock_judge(ext, ver)
        from app.evaluation.generation_eval import FaithfulnessEvaluator
        evaluator = FaithfulnessEvaluator(judge)
        result = evaluator.evaluate("query", "answer", "context")
        assert result["score"] == 0.0
        assert result["supported_claims"] == 0

    def test_empty_answer(self):
        from app.evaluation.generation_eval import FaithfulnessEvaluator
        from app.generation.generator import StubGenerator
        evaluator = FaithfulnessEvaluator(StubGenerator())
        result = evaluator.evaluate("query", "", "context")
        assert result["score"] == 0.0

    def test_no_claims_extracted(self):
        """LLM 返回空数组时，score=1.0（无可验证声明）。"""
        judge = self._mock_judge("[]", "[]")
        from app.evaluation.generation_eval import FaithfulnessEvaluator
        evaluator = FaithfulnessEvaluator(judge)
        result = evaluator.evaluate("query", "answer", "context")
        assert result["score"] == 1.0
        assert result["total_claims"] == 0


class TestRelevanceEvaluator:
    def _mock_judge(self, response):
        from app.generation.generator import BaseGenerator

        class MockJudge(BaseGenerator):
            def generate(self, messages):
                return response
        return MockJudge()

    def test_high_relevance(self):
        import json
        resp = json.dumps({"score": 0.9, "reasoning": "完全切题"}, ensure_ascii=False)
        judge = self._mock_judge(resp)
        from app.evaluation.generation_eval import RelevanceEvaluator
        evaluator = RelevanceEvaluator(judge)
        result = evaluator.evaluate("问题", "答案")
        assert result["score"] == 0.9
        assert "完全切题" in result["reasoning"]

    def test_score_clamped_to_range(self):
        """分数超 [0,1] 应被截断。"""
        import json
        resp = json.dumps({"score": 1.5, "reasoning": "超出"}, ensure_ascii=False)
        judge = self._mock_judge(resp)
        from app.evaluation.generation_eval import RelevanceEvaluator
        evaluator = RelevanceEvaluator(judge)
        result = evaluator.evaluate("q", "a")
        assert result["score"] == 1.0

    def test_empty_answer(self):
        from app.evaluation.generation_eval import RelevanceEvaluator
        from app.generation.generator import StubGenerator
        evaluator = RelevanceEvaluator(StubGenerator())
        result = evaluator.evaluate("query", "")
        assert result["score"] == 0.0

    def test_markdown_code_block_stripped(self):
        """LLM 返回带 ```json 包裹的内容应能正确解析。"""
        judge = self._mock_judge('```json\n{"score": 0.7, "reasoning": "ok"}\n```')
        from app.evaluation.generation_eval import RelevanceEvaluator
        evaluator = RelevanceEvaluator(judge)
        result = evaluator.evaluate("q", "a")
        assert result["score"] == 0.7
