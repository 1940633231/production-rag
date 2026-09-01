r"""先过滤后检索（permission-aware retrieval）单元测试。

验证四个检索链路在传入 document_ids 时都在"过滤后的集合内"检索，
不可读文档不会占用 top_k 名额，也不会出现在结果中:

  1. Retriever（向量）: document_ids → vector_ids 翻译并下传
  2. FAISSStore: flat 暴力重算 / ivf / hnsw IDSelector 预过滤
  3. BM25Search: get_batch_scores 只对可读文档打分
  4. MilvusStore: expr="id in [...]" 主键预过滤（mock client 验证 query 构造）
  5. ESClient / ESFulltextSearch: terms 文档过滤下传

无需外部服务（FAISS 本地 / 其余 mock），可独立运行:
  .venv\Scripts\python.exe -m pytest tests\search\test_prefilter.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retriever import Retriever  # noqa: E402
from app.search.bm25_search import BM25Search  # noqa: E402
from app.search.es_fulltext_search import ESFulltextSearch  # noqa: E402
from app.storage.es_client import ESClient, _ES_AVAILABLE  # noqa: E402
from app.vector.faiss_store import FAISSStore  # noqa: E402


# ============================================================
# 通用 chunk 数据：doc-a → vector_id {10,20}, doc-b → {30}
# ============================================================

def _chunk(vector_id, doc_id, content, chunk_id):
    return {
        "vector_id": vector_id,
        "document_id": doc_id,
        "chunk_id": chunk_id,
        "content": content,
        "start_offset": 0,
        "end_offset": len(content),
        "metadata": {"source": "test"},
    }


CHUNKS = [
    _chunk(10, "doc-a", "铁矿石 铁矿石", "a1"),
    _chunk(20, "doc-a", "价格", "a2"),
    _chunk(30, "doc-b", "钢材", "b1"),
]


# ============================================================
# 1. Retriever：document_ids → vector_ids 翻译并下传
# ============================================================

class _FakeEmbedding:
    def encode(self, texts):
        return np.zeros((len(texts), 4), dtype="float32")


class _FakeVectorStore:
    """记录 search 收到的 vector_ids 参数，返回预置结果。"""

    def __init__(self, scores, ids):
        self.scores = np.asarray([scores], dtype="float32")
        self.ids = np.asarray([ids], dtype="int64")
        self.called_vector_ids = None

    def search(self, query_vector, top_k, vector_ids=None):
        self.called_vector_ids = vector_ids
        return self.scores, self.ids


class _FakeChunkRepo:
    def __init__(self, chunks):
        self._chunks = chunks

    def list_all(self):
        return self._chunks

    def get_by_id(self, idx):
        for c in self._chunks:
            if c["vector_id"] == idx:
                return c
        return None


class TestRetrieverPrefilter:
    def _make(self, ids=(10, 30, 20), scores=(0.9, 0.8, 0.7)):
        vs = _FakeVectorStore(list(scores), list(ids))
        retriever = Retriever(
            embedding_model=_FakeEmbedding(),
            vector_store=vs,
            chunk_repo=_FakeChunkRepo(CHUNKS),
        )
        return retriever, vs

    def test_translates_document_ids_to_vector_ids(self):
        """document_ids={doc-a} 必须下传 vector_ids={10,20}（先过滤后检索）。"""
        retriever, vs = self._make()
        retriever.search("铁矿石", top_k=5, document_ids={"doc-a"})
        assert vs.called_vector_ids == {10, 20}

    def test_no_document_ids_means_no_filter(self):
        """document_ids=None 时 vector_ids 必须为 None（全量检索）。"""
        retriever, vs = self._make()
        retriever.search("铁矿石", top_k=5, document_ids=None)
        assert vs.called_vector_ids is None

    def test_results_only_contain_readable_docs(self):
        """防御性 ACL：即使后端返回不可读向量，结果也必须剔除。"""
        retriever, _ = self._make()
        results = retriever.search("铁矿石", top_k=5, document_ids={"doc-a"})
        assert {r["vector_id"] for r in results} == {10, 20}

    def test_empty_document_ids_builds_empty_vector_ids(self):
        """可读文档为空集时翻译出的 vector_ids 也是空集（后端返回空）。"""
        retriever, vs = self._make(ids=(), scores=())
        results = retriever.search("铁矿石", top_k=5, document_ids=set())
        assert vs.called_vector_ids == set()
        assert results == []


# ============================================================
# 2. FAISSStore：flat 暴力重算 / ivf / hnsw IDSelector 预过滤
# ============================================================

class TestFAISSPrefilter:
    DIM = 8
    # doc-a → id {10,20}, doc-b → {30,40}, doc-c → {50,60}
    IDS = [10, 20, 30, 40, 50, 60]

    @staticmethod
    def _vectors():
        rng = np.random.default_rng(0)
        v = rng.random((6, TestFAISSPrefilter.DIM), dtype="float32")
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(norms == 0, 1, norms)

    @pytest.mark.parametrize("index_type", ["flat", "ivf", "hnsw"])
    def test_only_allowed_ids_returned(self, index_type):
        """查询与不可读向量最相似时，结果仍只能来自可读集合。"""
        vectors = self._vectors()
        store = FAISSStore(dimension=self.DIM, index_type=index_type)
        store.add(vectors, ids=self.IDS)

        # 用 doc-b 的向量（id=30）做查询：它本身不可读，但 id=30 也不允许
        qv = vectors[2:3]  # id=30
        allowed = {10, 20}  # 只允许 doc-a
        scores, ids = store.search(qv, top_k=5, vector_ids=allowed)
        returned = [int(i) for i in ids[0] if int(i) != -1]
        assert set(returned) <= allowed, (
            "先过滤后检索失败: index_type={}, 返回={}".format(index_type, returned)
        )
        assert 30 not in returned

    @pytest.mark.parametrize("index_type", ["flat", "ivf", "hnsw"])
    def test_top_k_fully_filled_by_allowed(self, index_type):
        """top_k 名额由可读向量填满（不浪费在不可读向量上），不足补 -1。"""
        vectors = self._vectors()
        store = FAISSStore(dimension=self.DIM, index_type=index_type)
        store.add(vectors, ids=self.IDS)

        qv = vectors[0:1]  # 与 id=10 最相似（不可读）
        allowed = {30, 40}  # 只允许 doc-b
        scores, ids = store.search(qv, top_k=5, vector_ids=allowed)
        valid = [int(i) for i in ids[0] if int(i) != -1]
        # 可读集合内只有 2 个 → 返回这 2 个，其余补 -1，绝不含 id=10
        assert set(valid) == {30, 40}
        assert 10 not in valid

    @pytest.mark.parametrize("index_type", ["flat", "ivf", "hnsw"])
    def test_empty_allowed_returns_padded_empty(self, index_type):
        """vector_ids 为空集时返回全 -1 填充（与 FAISS 返回格式一致）。"""
        vectors = self._vectors()
        store = FAISSStore(dimension=self.DIM, index_type=index_type)
        store.add(vectors, ids=self.IDS)

        scores, ids = store.search(vectors[0:1], top_k=3, vector_ids=set())
        assert list(ids[0]) == [-1, -1, -1]
        assert list(scores[0]) == [-1.0, -1.0, -1.0]

    def test_no_vector_ids_returns_unfiltered(self):
        """不传 vector_ids 时与原有全量检索行为一致。"""
        vectors = self._vectors()
        store = FAISSStore(dimension=self.DIM, index_type="flat")
        store.add(vectors, ids=self.IDS)
        qv = vectors[0:1]
        scores, ids = store.search(qv, top_k=3)
        assert int(ids[0][0]) == 10  # 与自身最相似


# ============================================================
# 3. BM25Search：只对可读文档打分排序
# ============================================================

class TestBM25Prefilter:
    @pytest.fixture
    def bm25(self):
        return BM25Search(_FakeChunkRepo(CHUNKS))

    def test_only_readable_documents_ranked(self, bm25):
        """document_ids={doc-b} 时，即使 doc-a 得分更高也不能进入结果。"""
        # 查询 "铁矿石 钢材"：doc-a(chunk10) 得分 0.956 > doc-b(chunk30) 0.623；
        # 只允许 doc-b 时，结果必须只来自 doc-b（先过滤后检索，doc-a 不占名额）
        results = bm25.search("铁矿石 钢材", top_k=2, document_ids={"doc-b"})
        assert [r["vector_id"] for r in results] == [30]

    def test_top_k_filled_by_readable_without_wasting_quota(self, bm25):
        """top_k 名额由可读文档填满：只允许 doc-a（2 个 chunk）时返回全部 2 个。"""
        results = bm25.search("铁矿石 价格", top_k=5, document_ids={"doc-a"})
        returned = {r["vector_id"] for r in results}
        assert returned <= {10, 20}
        assert 30 not in returned

    def test_empty_document_ids_returns_empty(self, bm25):
        results = bm25.search("钢材", top_k=5, document_ids=set())
        assert results == []

    def test_no_document_ids_returns_all(self, bm25):
        results = bm25.search("铁矿石 钢材", top_k=10, document_ids=None)
        assert {r["vector_id"] for r in results} <= {10, 20, 30}


# ============================================================
# 4. MilvusStore：expr="id in [...]" 主键预过滤（mock client）
# ============================================================

class _FakeMilvusClient:
    """记录 search 调用，返回可读集合内的命中。"""

    def __init__(self):
        self.calls = []  # 每次 search 的 kwargs
        self.loaded = []

    def load_collection(self, name):
        self.loaded.append(name)

    def search(self, collection_name, **kwargs):
        self.calls.append(kwargs)
        # pymilvus 3.x 主键过滤参数名为 filter（2.x 为 expr）
        filter_expr = kwargs.get("filter") or kwargs.get("expr")
        ids = [30, 40] if filter_expr else [10, 30, 40, 50]
        return [[{"id": i, "distance": 1.0 - i * 0.001} for i in ids]]


@pytest.fixture
def milvus_store():
    from app.vector.milvus_store import MilvusStore, _PYMILVUS_AVAILABLE

    if not _PYMILVUS_AVAILABLE:
        pytest.skip("pymilvus 未安装")
    store = MilvusStore(dimension=8, collection_name="pytest_prefilter")
    store._collection_name = "pytest_prefilter"
    fake = _FakeMilvusClient()
    store._client = fake
    return store, fake


class TestMilvusPrefilter:
    def test_expr_contains_allowed_ids(self, milvus_store):
        store, fake = milvus_store
        qv = np.zeros((1, 8), dtype="float32")
        store.search(qv, top_k=5, vector_ids={40, 30})
        assert len(fake.calls) == 1
        # pymilvus 3.x 参数名为 filter
        assert fake.calls[0]["filter"] == "id in [30,40]"

    def test_no_vector_ids_no_filter_kwarg(self, milvus_store):
        store, fake = milvus_store
        qv = np.zeros((1, 8), dtype="float32")
        store.search(qv, top_k=5, vector_ids=None)
        assert "filter" not in fake.calls[0]
        assert "expr" not in fake.calls[0]

    def test_empty_allowed_returns_padded_empty_without_search(self, milvus_store):
        store, fake = milvus_store
        qv = np.zeros((1, 8), dtype="float32")
        scores, ids = store.search(qv, top_k=3, vector_ids=set())
        assert fake.calls == []  # 未发起真实检索
        assert list(ids[0]) == [-1, -1, -1]

    def test_search_with_expr_only_returns_allowed(self, milvus_store):
        store, fake = milvus_store
        qv = np.zeros((1, 8), dtype="float32")
        # fake 在 expr 存在时只返回 {30,40}，模拟 Milvus 端预过滤
        _scores, ids = store.search(qv, top_k=5, vector_ids={30, 40})
        valid = [int(i) for i in ids[0] if int(i) != -1]
        assert set(valid) == {30, 40}


# ============================================================
# 5. ESClient / ESFulltextSearch：terms 文档过滤下传
# ============================================================

class _FakeESClient:
    """记录 search 调用，并模拟 ES 端 terms 预过滤（只返回可读命中）。

    兼容两种调用风格:
      - ESClient.search → self._client.search(index=..., body=...)
      - ESFulltextSearch.search → es.search(strategy=..., query=..., document_ids=...)
    """

    def __init__(self, hits):
        self._hits = hits
        self.calls = []  # 每次 search 的 kwargs

    def _index_name(self, strategy):
        return "pytest_pre_{}".format(strategy)

    def search(self, **kwargs):
        self.calls.append(kwargs)
        doc_filter = kwargs.get("document_ids")
        if doc_filter is None:
            body = kwargs.get("body") or {}
            query = body.get("query")
            if isinstance(query, dict) and "bool" in query:
                for clause in query["bool"].get("filter", []):
                    if "terms" in clause:
                        doc_filter = clause["terms"]["document_id"]
        if doc_filter is not None:
            doc_filter = set(doc_filter)
            allowed = [
                h for h in self._hits
                if h["_source"]["document_id"] in doc_filter
            ]
        else:
            allowed = self._hits
        return {"hits": {"hits": allowed}}


def _es_hit(chunk_id, doc_id, vector_id):
    return {
        "_source": {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "strategy": "test",
            "chunk_index": 0,
            "vector_id": vector_id,
            "content": "内容",
            "start_offset": 0,
            "end_offset": 2,
            "metadata": {},
        },
        "_score": 0.9,
    }


@pytest.fixture
def es_client():
    if not _ES_AVAILABLE:
        pytest.skip("elasticsearch-py 未安装")
    client = ESClient(hosts=["http://localhost:9999"], index_prefix="pytest_pre")
    return client


class TestESClientPrefilter:
    def _client_with_fake(self, es_client, hits):
        fake = _FakeESClient(hits)
        es_client._client = fake
        return es_client, fake

    def test_terms_filter_in_body(self, es_client):
        es_client, fake = self._client_with_fake(es_client, [])
        es_client.search("test", query="全文", top_k=5,
                         document_ids=["doc-a", "doc-b"])
        body = fake.calls[0]["body"]
        filt = body["query"]["bool"]["filter"]
        assert {"terms": {"document_id": ["doc-a", "doc-b"]}} in filt

    def test_match_clause_preserved_in_must(self, es_client):
        es_client, fake = self._client_with_fake(es_client, [])
        es_client.search("test", query="铁矿石", top_k=5, document_ids=["doc-a"])
        body = fake.calls[0]["body"]
        assert body["query"]["bool"]["must"] == [{"match": {"content": "铁矿石"}}]

    def test_no_document_ids_plain_query(self, es_client):
        es_client, fake = self._client_with_fake(es_client, [])
        es_client.search("test", query="全文", top_k=5, document_ids=None)
        body = fake.calls[0]["body"]
        assert body["query"] == {"match": {"content": "全文"}}

    def test_empty_document_ids_returns_empty_without_query(self, es_client):
        es_client, fake = self._client_with_fake(es_client, [])
        results = es_client.search("test", query="全文", top_k=5,
                                   document_ids=[])
        assert results == []
        assert fake.calls == []  # 未发起查询

    def test_returns_only_filtered_hits(self, es_client):
        es_client, fake = self._client_with_fake(
            es_client,
            [_es_hit("a1", "doc-a", 10), _es_hit("b1", "doc-b", 30)],
        )
        results = es_client.search("test", query="全文", top_k=5,
                                   document_ids=["doc-a"])
        assert {r["document_id"] for r in results} == {"doc-a"}


class _FakeESFulltext:
    """模拟 ESClient 层（ESFulltextSearch 的注入对象）：返回扁平结果字典。"""

    def __init__(self, hits):
        self._hits = hits
        self.calls = []  # 每次 search 收到的 document_ids

    def _index_name(self, strategy):
        return "pytest_pre_{}".format(strategy)

    def ping(self):
        return True

    def count(self, strategy):
        return len(self._hits)

    def search(self, strategy, query, top_k=10, sort_by_vector_id=False,
               document_ids=None):
        self.calls.append(document_ids)
        if document_ids is not None:
            return [h for h in self._hits if h["document_id"] in document_ids][:top_k]
        return self._hits[:top_k]


def _flat_hit(chunk_id, doc_id, vector_id):
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "strategy": "test",
        "chunk_index": 0,
        "vector_id": vector_id,
        "content": "内容",
        "start_offset": 0,
        "end_offset": 2,
        "metadata": {},
        "score": 0.9,
    }


class TestESFulltextPrefilter:
    def test_passes_document_ids_down(self):
        hits = [_flat_hit("a1", "doc-a", 10), _flat_hit("b1", "doc-b", 30)]
        fake_es = _FakeESFulltext(hits)
        search = ESFulltextSearch(strategy="test", es_client=fake_es)
        results = search.search("全文", top_k=5, document_ids={"doc-a"})
        # terms 已下传到 ES，结果只含可读文档（无后置过滤）
        assert fake_es.calls[-1] == {"doc-a"}
        assert [r["vector_id"] for r in results] == [10]

    def test_no_document_ids_returns_all(self):
        hits = [_flat_hit("a1", "doc-a", 10), _flat_hit("b1", "doc-b", 30)]
        fake_es = _FakeESFulltext(hits)
        search = ESFulltextSearch(strategy="test", es_client=fake_es)
        results = search.search("全文", top_k=5, document_ids=None)
        assert fake_es.calls[-1] is None
        assert {r["vector_id"] for r in results} == {10, 30}
