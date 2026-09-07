"""Retriever.search_batch 单元测试（批量多路检索）。

覆盖:
  - 一次 batch encode 所有 query（而非 N 次单条调用）
  - 每路独立检索，返回与 queries 一一对应的结果列表
  - document_ids → vector_ids 预过滤（各路共用一次翻译）
  - ACL 兜底过滤（防御性校验）

运行:
  .venv\\Scripts\\python.exe -m pytest tests\\rag\\test_retriever_batch.py -v
"""
import numpy as np
import pytest

from app.rag.retriever import Retriever


class FakeEmbedding:
    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def encode(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        # 每条 query 一个独立向量（按输入序号区分）
        arr = np.arange(len(texts)).reshape(-1, 1) * 10
        return np.tile(arr, (1, self.dim)).astype("float32")


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def search(self, qv, top_k, vector_ids=None):
        self.calls.append({"top_k": top_k, "vector_ids": vector_ids})
        n = min(top_k, 3)
        scores = np.zeros((1, top_k), dtype="float32")
        ids = np.full((1, top_k), -1, dtype="int64")
        ids[0, :n] = np.arange(n)   # 返回 vector_id 0,1,2
        return scores, ids


class FakeChunkRepo:
    def list_all(self):
        return [
            {"document_id": "doc-a", "vector_id": 0},
            {"document_id": "doc-a", "vector_id": 1},
            {"document_id": "doc-b", "vector_id": 2},
        ]

    def get_by_id(self, idx):
        return {
            "chunk_id": "c{}".format(idx),
            "document_id": "doc-a" if idx < 2 else "doc-b",
            "vector_id": idx,
            "content": "内容{}".format(idx),
            "metadata": {},
            "start_offset": 0,
            "end_offset": 0,
        }

    def vector_ids_by_documents(self, document_ids):
        allowed = set(document_ids)
        m = {}
        for c in self.list_all():
            doc = c.get("document_id")
            if doc and doc in allowed:
                m.setdefault(doc, set()).add(int(c["vector_id"]))
        return m


@pytest.fixture
def retriever():
    emb = FakeEmbedding()
    store = FakeVectorStore()
    return Retriever(emb, store, FakeChunkRepo())


class TestSearchBatch:
    def test_encodes_all_queries_once(self, retriever):
        """一次 batch encode 全部 query（非 N 次调用）。"""
        retriever.search_batch(["q1", "q2", "q3"], per_query=5)
        assert retriever.embedding_model.calls == [["q1", "q2", "q3"]]

    def test_returns_per_route_results(self, retriever):
        """返回与 queries 一一对应的结果列表。"""
        results = retriever.search_batch(["q1", "q2"], per_query=5)
        assert len(results) == 2
        for route in results:
            assert len(route) <= 5
            assert all("chunk_id" in r and "score" in r for r in route)

    def test_prefilters_document_ids(self, retriever):
        """document_ids → vector_ids 预过滤（各路共用一次翻译）。"""
        retriever.search_batch(["q1", "q2"], per_query=5, document_ids={"doc-a"})
        # 只有 doc-a 的 vector_ids {0,1}
        for call in retriever.vector_store.calls:
            assert call["vector_ids"] == {0, 1}

    def test_acl_fallback_filters_invisible(self, retriever):
        """ACL 兜底：不可读文档（doc-b, chunk c2）的结果被过滤。"""
        results = retriever.search_batch(
            ["q1"], per_query=5, document_ids={"doc-a"}
        )
        assert len(results) == 1
        chunk_ids = [r["chunk_id"] for r in results[0]]
        assert "c2" not in chunk_ids   # doc-b 的 chunk 被兜底过滤
        assert all("c2" != c for c in chunk_ids)

    def test_no_document_ids_no_prefilter(self, retriever):
        """document_ids=None：不设预过滤。"""
        retriever.search_batch(["q1"], per_query=5, document_ids=None)
        for call in retriever.vector_store.calls:
            assert call["vector_ids"] is None
