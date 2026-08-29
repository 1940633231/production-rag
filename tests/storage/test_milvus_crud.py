"""Milvus 向量存储集成测试。

验证 MilvusStore 的核心接口（add/search/save/load/drop/count），
重点验证 Milvus 返回的 int ids 与 FAISSStore 格式兼容（不足 top_k 补 -1），
以及写入顺序与 auto_id 主键的一致性（索引 i 的向量对应主键 i）。

前置条件:
  1. Milvus 服务运行中（默认 127.0.0.1:19530 standalone 或 cluster）
  2. .env 中可配置 MILVUS_HOST / MILVUS_PORT / MILVUS_COLLECTION_PREFIX 覆盖默认值
  3. pymilvus 已安装（pip install pymilvus）

运行方式:
  .venv\\Scripts\\python.exe -m pytest tests/storage/test_milvus_crud.py -v

  如果 pymilvus 未安装或 Milvus 不可达，测试会自动 skip。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# 注入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env（pytest 不走 FastAPI 启动流程，需手动加载）
from app.core.env import load_env  # noqa: E402

load_env()

from app.vector.milvus_store import MilvusStore, _PYMILVUS_AVAILABLE  # noqa: E402
from app.core.config import Config  # noqa: E402


TEST_DIM = 16
TEST_COLLECTION = "pytest_milvus_smoke"  # 与生产前缀隔离，避免污染


def _milvus_kwargs():
    """读取 Config 中的 Milvus 连接参数（支持 MILVUS_* 环境变量覆盖）。"""
    cfg = Config()
    return {
        "host": cfg.milvus_host,
        "port": cfg.milvus_port,
    }


@pytest.fixture(scope="module")
def milvus_conn():
    """模块级 MilvusStore 连接 fixture。

    pymilvus 未安装 → skip 全部。
    服务不可达（ping 失败）→ skip 全部。
    测试前后清理 TEST_COLLECTION。
    """
    if not _PYMILVUS_AVAILABLE:
        pytest.skip("pymilvus 未安装，请先 pip install pymilvus")

    kwargs = _milvus_kwargs()
    try:
        store = MilvusStore(
            dimension=TEST_DIM,
            collection_name=TEST_COLLECTION,
            **kwargs,
        )
    except Exception as e:
        pytest.skip("MilvusStore 构造失败: {}".format(e))

    # 验证连通性：尝试连接 + 建测试用临时 collection 判断可达
    try:
        store._connect()
    except Exception as e:
        pytest.skip("Milvus 不可达，请检查服务或 .env 配置: {}".format(e))

    # 先清理（上次残留）
    try:
        store.drop(TEST_COLLECTION)
    except Exception:
        pass

    yield store

    # 模块收尾清理
    try:
        store.drop(TEST_COLLECTION)
    except Exception:
        pass
    try:
        store.close()
    except Exception:
        pass


@pytest.fixture
def clean_store(milvus_conn):
    """每个用例前清空测试 collection，保持用例独立。"""
    try:
        milvus_conn.drop(TEST_COLLECTION)
    except Exception:
        pass
    # 重置内部状态（便于重新创建）
    milvus_conn._collection_name = None
    milvus_conn.collection_name = TEST_COLLECTION
    return milvus_conn


# ---------------------------------------------------------------------------
# 基础接口
# ---------------------------------------------------------------------------


def test_add_writes_vectors_and_increments_count(clean_store):
    """add(vectors) 后 count() 应匹配。"""
    vectors = np.random.rand(8, TEST_DIM).astype("float32")
    clean_store.add(vectors)
    clean_store._client.flush(TEST_COLLECTION)
    assert clean_store.count() == 8


def test_search_returns_sorted_results_in_faiss_compatible_format(clean_store):
    """search(qv, k) 必须返回 (scores_2d, ids_2d)，id 与写入顺序一致。

    - ids 类型为 int，与 enumerate 顺序对齐（vector_id 兼容 FAISS position）
    - 结果按相似度从高到低排序（Milvus IP = 内积，归一化向量等价 cosine）
    """
    n = 10
    rng = np.random.default_rng(42)
    vectors = rng.random((n, TEST_DIM), dtype="float32")
    # 先归一化，保证 IP = cosine 相似度便于验证
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)

    clean_store.add(vectors)
    clean_store._client.flush(TEST_COLLECTION)

    # 用第 3 条向量做查询，期望最佳命中 id==3
    qv = vectors[3:4]
    scores, ids = clean_store.search(qv, top_k=3)

    # FAISS 兼容：二维数组
    assert scores.shape == (1, 3), "scores 应为 (1, k) 二维 ndarray"
    assert ids.shape == (1, 3), "ids 应为 (1, k) 二维 ndarray"
    # 最佳命中必须是它自己 (id==3)
    assert int(ids[0][0]) == 3, "最相似向量应为 id=3（自身），实际 ids[0]={}".format(ids[0])
    # 分数按降序
    scored = list(scores[0])
    assert scored == sorted(scored, reverse=True), "scores 未按降序: {}".format(scored)


def test_search_pads_with_minus_one_when_fewer_results(clean_store):
    """数据量小于 top_k 时，ids/scores 尾部补 -1（FAISS 兼容）。"""
    vectors = np.random.rand(2, TEST_DIM).astype("float32")
    clean_store.add(vectors)
    clean_store._client.flush(TEST_COLLECTION)

    scores, ids = clean_store.search(vectors[0:1], top_k=10)
    assert ids.shape == (1, 10)
    valid_ids = [int(i) for i in ids[0] if int(i) != -1]
    assert len(valid_ids) == 2
    # 末尾必须是 -1 填充
    assert int(ids[0][-1]) == -1
    assert float(scores[0][-1]) == -1.0


def test_save_and_load_round_trip(clean_store):
    """save(collection_name) 之后重新 load() 可正常检索。"""
    n = 6
    vectors = np.random.rand(n, TEST_DIM).astype("float32")
    clean_store.add(vectors)
    # save(path) 中 path 作为 collection 名称（这里保持与 TEST_COLLECTION 一致）
    clean_store.save(TEST_COLLECTION)

    # 新建 store 实例并 load，模拟下次服务重启
    kwargs = _milvus_kwargs()
    new_store = MilvusStore(
        dimension=TEST_DIM,
        collection_name="should_be_overridden",
        **kwargs,
    )
    new_store.load(TEST_COLLECTION)
    assert new_store.count() == n

    # 随机一条检索自查询，确认数据可被取回
    qv = vectors[2:3]
    _scores, ids = new_store.search(qv, top_k=1)
    assert int(ids[0][0]) == 2
    new_store.close()


def test_drop_collection_removes_data(clean_store):
    """drop() 后再 search 应返回空/抛错，且 collection 已不存在。"""
    vectors = np.random.rand(4, TEST_DIM).astype("float32")
    clean_store.add(vectors)
    clean_store.save(TEST_COLLECTION)
    assert clean_store.count() == 4

    clean_store.drop()
    # 再次查询：因 collection 不存在，search 前 _ensure_collection 会新建空 collection
    # count 应返回 0
    clean_store._collection_name = None  # 强制重新走 ensure 路径
    clean_store._ensure_collection(TEST_COLLECTION)
    assert clean_store.count() == 0


def test_count_without_loading_returns_zero(clean_store):
    """未 add/load 时 count 返回 0，不抛异常。"""
    assert clean_store.count() == 0


# ---------------------------------------------------------------------------
# 维度 / 数据校验
# ---------------------------------------------------------------------------


def test_add_rejects_dimension_mismatch(clean_store):
    """向量维度不匹配必须抛 ValueError。"""
    wrong = np.random.rand(3, TEST_DIM + 2).astype("float32")
    with pytest.raises(ValueError, match="维度不匹配"):
        clean_store.add(wrong)


def test_add_rejects_non_2d_array(clean_store):
    """一维向量必须抛 ValueError。"""
    with pytest.raises(ValueError, match="二维数组"):
        clean_store.add(np.random.rand(TEST_DIM).astype("float32"))


# ---------------------------------------------------------------------------
# vector_id 一致性：auto_id 从 0 开始（前提是 collection 新建且 drop 过）
# ---------------------------------------------------------------------------


def test_auto_id_matches_enumerate_order(clean_store):
    """N 条写入后按顺序查询自匹配，id[i] 必须 == i。"""
    n = 12
    rng = np.random.default_rng(0)
    vectors = rng.random((n, TEST_DIM), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)

    clean_store.add(vectors)
    clean_store._client.flush(TEST_COLLECTION)
    assert clean_store.count() == n

    for i in range(n):
        _scores, ids = clean_store.search(vectors[i : i + 1], top_k=1)
        best_id = int(ids[0][0])
        assert best_id == i, (
            "第 {} 条向量自查询应命中 id={}，实际={}".format(i, i, best_id)
        )
