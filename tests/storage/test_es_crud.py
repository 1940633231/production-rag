"""ES 存储层集成测试。

验证 ESClient / ChunkESRepository 的核心接口，重点验证
vector_id 与 FAISS 位置 ID 的对齐（修复 ID 错位隐患）。

前置条件:
  1. ES 服务运行中（默认 http://127.0.0.1:9200）
  2. .env 中已配置 ES_HOSTS / ES_USER / ES_PASSWORD / ES_INDEX_PREFIX / ES_TIMEOUT
  3. elasticsearch-py 已安装（>=8.9,<9）

运行方式:
  .venv\\Scripts\\python.exe -m pytest tests/storage/test_es_crud.py -v

  如果 ES 不可用，测试会自动 skip。
"""
import sys
from pathlib import Path

import pytest

# 注入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env（pytest 不走 FastAPI 启动流程，需手动加载）
from app.core.env import load_env
load_env()

from app.ingestion.chunk import Chunk
from app.storage.es_client import ESClient, _ES_AVAILABLE
from app.storage.es_repository import ChunkESRepository


STRATEGY = "test_smoke"


@pytest.fixture(scope="module")
def es_client():
    """ESClient fixture，ES 不可用时跳过全部测试。"""
    if not _ES_AVAILABLE:
        pytest.skip("elasticsearch-py 未安装")
    try:
        client = ESClient()
    except Exception as e:
        pytest.skip("ESClient 构造失败: {}".format(e))
    if not client.ping():
        pytest.skip("ES 不可达，请检查 ES 服务与 .env 配置")
    # 清理上次残留
    client.drop_index(STRATEGY)
    yield client
    # 模块结束清理
    try:
        client.drop_index(STRATEGY)
    except Exception:
        pass


@pytest.fixture
def es_repo(es_client):
    """每个测试用例独立的 ChunkESRepository，开始前清空索引。"""
    es_client.drop_index(STRATEGY)
    return ChunkESRepository(strategy=STRATEGY, es_client=es_client)


def _make_chunk(chunk_id: str, document_id: str, chunk_index: int,
                content: str, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=start,
        end_offset=end,
        metadata={"source": "test"},
    )


# ---- 基础接口 ----

def test_ping(es_client):
    assert es_client.ping() is True


def test_create_and_drop_index(es_client):
    es_client.drop_index(STRATEGY)
    es_client.create_index(STRATEGY)
    idx = es_client._index_name(STRATEGY)
    # ES 8.x 的 indices.exists 返回 HeadApiResponse（truthy），不是纯 True
    assert bool(es_client._client.indices.exists(index=idx)) is True
    es_client.drop_index(STRATEGY)
    assert bool(es_client._client.indices.exists(index=idx)) is False


# ---- vector_id 对齐（核心修复点）----

def test_batch_insert_writes_vector_id(es_repo, es_client):
    """batch_insert 写入的 ES 文档必须包含 vector_id 字段，且与 chunks 顺序对齐。"""
    chunks = [
        _make_chunk("c-0", "doc-1", 0, "第一段内容", 0, 10),
        _make_chunk("c-1", "doc-1", 1, "第二段内容", 10, 20),
        _make_chunk("c-2", "doc-2", 0, "第三段内容", 0, 10),
    ]
    es_repo.batch_insert(chunks)
    es_client._client.indices.refresh(
        index=es_client._index_name(STRATEGY)
    )

    # 拉取全量，按 vector_id 升序
    results = es_client.search(STRATEGY, query="*", top_k=100,
                                sort_by_vector_id=True)
    assert len(results) == 3
    # 关键断言：vector_id 必须存在且与 chunks 顺序对齐
    assert [r["vector_id"] for r in results] == [0, 1, 2]
    assert [r["chunk_id"] for r in results] == ["c-0", "c-1", "c-2"]


def test_count_returns_correct_number(es_repo, es_client):
    """count() 必须返回正确数量（修复 top_k=0 返回 0 的 bug）。"""
    chunks = [_make_chunk("c-{}".format(i), "doc-1", i, "内容{}".format(i),
                          i * 10, i * 10 + 10) for i in range(5)]
    es_repo.batch_insert(chunks)
    es_client._client.indices.refresh(
        index=es_client._index_name(STRATEGY)
    )
    # 未加载缓存时走 ES count API
    assert es_repo.count() == 5
    # 加载缓存后走 len(_cache_list)
    es_repo.list_all()
    assert es_repo.count() == 5


def test_get_by_id_aligned_with_faiss_position(es_repo, es_client):
    """get_by_id(int) 必须返回 vector_id == int 的 chunk（不是 enumerate 顺序）。

    模拟 Retriever.search 的调用方式：FAISS 返回 ids=[1, 0, 2]，
    用 get_by_id(1/0/2) 必须拿到对应 chunks[1/0/2]。
    """
    chunks = [
        _make_chunk("alpha", "doc-1", 0, "alpha 内容", 0, 10),
        _make_chunk("beta", "doc-1", 1, "beta 内容", 10, 20),
        _make_chunk("gamma", "doc-2", 0, "gamma 内容", 0, 10),
    ]
    es_repo.batch_insert(chunks)
    es_client._client.indices.refresh(
        index=es_client._index_name(STRATEGY)
    )

    # 模拟 FAISS 返回 ids=[1, 0, 2]（任意顺序）
    retrieved = [es_repo.get_by_id(i) for i in [1, 0, 2]]
    assert retrieved[0]["chunk_id"] == "beta"   # vector_id=1
    assert retrieved[1]["chunk_id"] == "alpha"  # vector_id=0
    assert retrieved[2]["chunk_id"] == "gamma"  # vector_id=2
    # 越界返回 None
    assert es_repo.get_by_id(99) is None


def test_list_all_sorted_by_vector_id(es_repo, es_client):
    """list_all() 按 vector_id 升序返回，顺序与写入一致。"""
    chunks = [
        _make_chunk("c-{}".format(i), "doc-1", i, "内容{}".format(i),
                    i * 10, i * 10 + 10)
        for i in range(4)
    ]
    es_repo.batch_insert(chunks)
    es_client._client.indices.refresh(
        index=es_client._index_name(STRATEGY)
    )
    all_chunks = es_repo.list_all()
    assert [c["vector_id"] for c in all_chunks] == [0, 1, 2, 3]


# ---- search 全文检索 ----

def test_search_match_returns_vector_id(es_repo, es_client):
    """match 查询结果必须包含 vector_id 字段。"""
    chunks = [
        _make_chunk("c-0", "doc-1", 0,
                    "Elasticsearch 是一个分布式的全文检索引擎", 0, 30),
        _make_chunk("c-1", "doc-1", 1,
                    "RAG 检索增强生成结合了检索与生成模型", 30, 60),
    ]
    es_repo.batch_insert(chunks)
    es_client._client.indices.refresh(
        index=es_client._index_name(STRATEGY)
    )
    results = es_client.search(STRATEGY, query="全文检索", top_k=5)
    assert len(results) >= 1
    assert "vector_id" in results[0]
    assert results[0]["vector_id"] in (0, 1)
