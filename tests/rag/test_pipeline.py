r"""RAG Pipeline 单元测试：编排流程 + 流式输出 + 可选组件。

用 mock retriever/reranker/generator 避免依赖 faiss/embedding/LLM，专注验证编排逻辑。

覆盖:
  - RAGPipeline.run: 完整链路、无 reranker、无 generator
  - RAGPipeline.run_stream: 流式事件序列
  - Citation 提取
  - 异常处理

运行:
  .venv\Scripts\python.exe -m pytest tests\rag\test_pipeline.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.pipeline import RAGPipeline
from app.rag.result import RAGResponse


# ---- Mock 组件 ----

class MockRetriever:
    """Mock 检索器：返回固定结果。"""
    def __init__(self, results):
        self._results = results
        self.search_called = 0

    def search(self, query, top_k=5):
        self.search_called += 1
        return self._results[:top_k]


class MockReranker:
    """Mock 重排器：按 score 降序排列。"""
    def rerank(self, query, candidates, top_k=5):
        return sorted(candidates, key=lambda r: r.get("score", 0), reverse=True)[:top_k]


class MockGenerator:
    """Mock 生成器：返回固定 answer，含引用编号。"""
    def generate(self, messages):
        return "根据上下文[1]，答案是示例内容[2]。"

    def stream_generate(self, messages):
        for chunk in ["根据上下文[1]", "，答案是示例内容[2]。"]:
            yield chunk


class FailingGenerator:
    """总是抛异常的生成器。"""
    def generate(self, messages):
        raise RuntimeError("LLM 生成失败")

    def stream_generate(self, messages):
        raise RuntimeError("LLM 流式失败")
        yield  # 让 Python 识别为生成器


# ---- 辅助 ----

def _chunk(cid, doc_id, content, start, end, score=0.5, idx=0):
    return {
        "chunk_id": cid,
        "content": content,
        "start_offset": start,
        "end_offset": end,
        "score": score,
        "metadata": {
            "document_id": doc_id,
            "chunk_index": idx,
            "file_name": doc_id + ".txt",
        },
    }


def _mock_chunks():
    return [
        _chunk("c1", "d1", "铁矿石供应增加", 0, 10, score=0.9, idx=0),
        _chunk("c2", "d1", "铁矿石需求下降", 10, 20, score=0.7, idx=1),
        _chunk("c3", "d2", "钢材市场波动", 0, 8, score=0.5, idx=0),
    ]


# ============================================================
# RAGPipeline.run
# ============================================================

class TestPipelineRun:
    def test_full_pipeline_with_reranker_and_generator(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=MockReranker(),
            context_manager=None,  # 用 fallback
            generator=MockGenerator(),
            top_k=2,
            rerank_candidate_pool=3,
        )
        resp = pipeline.run("铁矿供需")
        assert isinstance(resp, RAGResponse)
        assert resp.query == "铁矿供需"
        assert resp.answer == "根据上下文[1]，答案是示例内容[2]。"
        assert len(resp.citations) == 2  # [1] 和 [2]
        assert resp.citations[0]["number"] == 1
        assert resp.citations[1]["number"] == 2
        # reranker 启用时应取宽候选池
        assert retriever.search_called == 1

    def test_pipeline_without_reranker(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=None,
            generator=MockGenerator(),
            top_k=2,
        )
        resp = pipeline.run("query")
        assert resp.answer is not None
        assert len(resp.chunks) == 2

    def test_pipeline_without_generator(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=None,
            generator=None,
            top_k=2,
        )
        resp = pipeline.run("query")
        assert resp.answer is None
        assert resp.citations == []
        # 但 context 和 chunks 应有值
        assert resp.context
        assert len(resp.chunks) == 2

    def test_pipeline_with_context_manager(self):
        from app.context.manager import ContextManager
        from app.ingestion.tokenizer import CharLengthCounter
        retriever = MockRetriever(_mock_chunks())
        ctx_mgr = ContextManager(
            token_counter=CharLengthCounter(),
            max_context_tokens=1000,
            reserved_tokens=100,
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            context_manager=ctx_mgr,
            generator=MockGenerator(),
            top_k=3,
        )
        resp = pipeline.run("query")
        assert resp.stats.get("after_dedup") is not None
        assert "[1]" in resp.context

    def test_retriever_failure_propagates(self):
        class FailingRetriever:
            def search(self, query, top_k=5):
                raise RuntimeError("检索失败")
        pipeline = RAGPipeline(retriever=FailingRetriever())
        with pytest.raises(RuntimeError, match="检索失败"):
            pipeline.run("query")


# ============================================================
# RAGPipeline.run_stream
# ============================================================

class TestPipelineRunStream:
    def test_stream_event_sequence(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            reranker=None,
            generator=MockGenerator(),
            top_k=2,
        )
        events = list(pipeline.run_stream("query"))
        # 应包含 meta → delta(多次) → citations → done
        types = [e["type"] for e in events]
        assert types[0] == "meta"
        assert "delta" in types
        assert "citations" in types
        assert types[-1] == "done"

    def test_stream_meta_has_chunks_and_stats(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=MockGenerator(),
            top_k=2,
        )
        events = list(pipeline.run_stream("query"))
        meta = events[0]
        assert meta["type"] == "meta"
        assert "chunks" in meta
        assert "stats" in meta

    def test_stream_without_generator(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=None,
            top_k=2,
        )
        events = list(pipeline.run_stream("query"))
        types = [e["type"] for e in events]
        # 无 generator 时应直接 done，无 delta
        assert "meta" in types
        assert "delta" not in types
        assert types[-1] == "done"

    def test_stream_citations_extracted(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=MockGenerator(),
            top_k=2,
        )
        events = list(pipeline.run_stream("query"))
        cit_events = [e for e in events if e["type"] == "citations"]
        assert len(cit_events) == 1
        assert len(cit_events[0]["citations"]) == 2

    def test_stream_generator_failure_yields_error(self):
        retriever = MockRetriever(_mock_chunks())
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=FailingGenerator(),
            top_k=2,
        )
        events = list(pipeline.run_stream("query"))
        # 应有 meta（检索成功）→ error（生成失败）
        assert events[0]["type"] == "meta"
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) >= 1
        assert "LLM" in error_events[0]["error"]

    def test_stream_retriever_failure_yields_error(self):
        class FailingRetriever:
            def search(self, query, top_k=5):
                raise RuntimeError("检索失败")
        pipeline = RAGPipeline(retriever=FailingRetriever())
        events = list(pipeline.run_stream("query"))
        # 检索失败应直接 error 事件
        assert events[0]["type"] == "error"
        assert "检索" in events[0]["error"] or "pipeline" in events[0]["error"]


# ============================================================
# 多轮对话：history 透传
# ============================================================

class TestPipelineMultiTurn:
    """history 应作为独立 user/assistant 消息插入 system 与当前 user 之间。"""

    @staticmethod
    def _capturing_pipeline(history=None):
        """构建捕获 messages 的 pipeline，返回 (pipeline, captured dict)。"""
        captured = {}

        class CapturingGenerator(MockGenerator):
            def generate(self, messages):
                captured["messages"] = messages
                return "回答"

        pipeline = RAGPipeline(
            retriever=MockRetriever(_mock_chunks()),
            generator=CapturingGenerator(),
            top_k=2,
        )
        return pipeline, captured

    def test_history_inserted_between_system_and_user(self):
        pipeline, captured = self._capturing_pipeline()
        history = [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
        ]
        pipeline.run("第三问", history=history)
        msgs = captured["messages"]
        # system + 4 条历史 + 当前 user
        assert [m["role"] for m in msgs] == [
            "system", "user", "assistant", "user", "assistant", "user",
        ]
        assert msgs[1] == {"role": "user", "content": "第一问"}
        assert msgs[-1]["role"] == "user"
        assert "第三问" in msgs[-1]["content"]
        assert "上下文" in msgs[-1]["content"]

    def test_history_truncated_to_last_5_turns(self):
        pipeline, captured = self._capturing_pipeline()
        history = []
        for i in range(8):  # 8 轮 = 16 条消息，超出 MAX_HISTORY_TURNS=5
            history.append({"role": "user", "content": "问{}".format(i)})
            history.append({"role": "assistant", "content": "答{}".format(i)})
        pipeline.run("当前问", history=history)
        msgs = captured["messages"]
        # system + 最近 5 轮(10 条) + 当前 user = 12 条
        assert len(msgs) == 12
        assert msgs[1]["content"] == "问3"  # 前 3 轮被截断

    def test_history_filters_invalid_entries(self):
        pipeline, captured = self._capturing_pipeline()
        history = [
            {"role": "system", "content": "非法角色"},
            {"role": "user", "content": "   "},  # 空白内容
            {"role": "user", "content": "合法问题"},
            {"role": "assistant", "content": "合法回答"},
        ]
        pipeline.run("当前问", history=history)
        msgs = captured["messages"]
        # 非法条目被过滤：system + 2 条合法历史 + 当前 user
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[1]["content"] == "合法问题"

    def test_history_default_empty(self):
        pipeline, captured = self._capturing_pipeline()
        pipeline.run("当前问")  # 不传 history
        msgs = captured["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
