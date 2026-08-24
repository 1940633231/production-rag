"""Milvus 向量存储：基于 pymilvus 的分布式向量数据库后端。

依赖:
    pip install pymilvus
    需运行 Milvus 服务（ standalone 或 cluster ）

与 FAISSStore 接口完全一致，可无缝替换:
    - add(vectors):    批量写入向量，自动建 collection（IP 内积，与 FAISS IndexFlatIP 对齐）
    - search(qv, k):   返回 (scores, ids)，格式与 FAISSStore 一致
    - save(path):      path 作为 collection name（Milvus 持久化在远端，无需本地文件）
    - load(path):      连接已有 collection

设计要点:
    - 维度从 embedding 模型动态获取（不硬编码）
    - metric_type=IP（内积），与 FAISSStore 的 IndexFlatIP 对齐
    - embedding 向量需归一化（EmbeddingModel.encode 已 normalize_embeddings=True）
    - 向量主键用自增 int id，与 metadata.json 的 key（"0"/"1"/...）一一对应
    - pymilvus 未安装时抛出明确 ImportError，不强制依赖

使用示例:
    from app.vector.milvus_store import MilvusStore
    store = MilvusStore(dimension=512, host="127.0.0.1", port=19530)
    store.add(vectors)
    scores, ids = store.search(query_vector, top_k=5)
"""
import time
from typing import Any, List, Optional

from app.core.logger import get_logger
from app.vector.base import BaseVectorStore

logger = get_logger(__name__)

# 尝试导入 pymilvus，不可用时降级
try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
    _PYMILVUS_AVAILABLE = True
except ImportError:
    _PYMILVUS_AVAILABLE = False


class MilvusStore(BaseVectorStore):
    """Milvus 向量存储后端。

    支持两种 Milvus 索引类型:
      - ivf:  IVF_FLAT，倒排索引（默认）
      - hnsw: HNSW，分层近邻图索引
    """

    def __init__(
        self,
        dimension: int,
        host: str = "127.0.0.1",
        port: int = 19530,
        collection_name: str = "rag_vectors",
        index_type: str = "ivf",
        ivf_nlist: int = 128,
        ivf_nprobe: int = 16,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        **kwargs,
    ):
        """初始化 Milvus 存储。

        参数:
            dimension: 向量维度（必须与 embedding 模型输出一致）
            host: Milvus 服务地址
            port: Milvus 服务端口
            collection_name: collection 名称（save/load 的 path 参数会覆盖此值）
            index_type: 索引类型 "ivf"(IVF_FLAT) / "hnsw"(HNSW)
            ivf_nlist / ivf_nprobe: IVF 参数
            hnsw_m / hnsw_ef_construction / hnsw_ef_search: HNSW 参数
        """
        if not _PYMILVUS_AVAILABLE:
            raise ImportError(
                "MilvusStore 需要 pymilvus: pip install pymilvus"
            )

        self.dimension = dimension
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.index_type = index_type.lower()
        # 索引参数
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self._collection: Optional[Any] = None
        self._connected = False

    # ---- 连接管理 ----

    def _connect(self):
        """连接 Milvus 服务（延迟连接，避免初始化时阻塞）。"""
        if self._connected:
            return
        t = time.time()
        alias = "rag_{}".format(id(self))
        logger.info("连接 Milvus: %s:%s (alias=%s)", self.host, self.port, alias)
        connections.connect(
            alias=alias,
            host=self.host,
            port=str(self.port),
        )
        self._alias = alias
        self._connected = True
        logger.info("Milvus 连接成功: %.3fs", time.time() - t)

    def _ensure_collection(self, collection_name: str):
        """确保 collection 存在，不存在则创建。

        主键 id 使用 INT64 + auto_id=False，由 add() 显式传入 vector_id=0..N-1，
        保证与 chunks enumerate 顺序、FAISS store 的 position 严格对齐。
        """
        self._connect()

        if utility.has_collection(collection_name, using=self._alias):
            self._collection = Collection(
                collection_name, using=self._alias
            )
            logger.info(
                "加载已有 collection: %s, rows=%d",
                collection_name, self._collection.num_entities,
            )
            return

        # 创建新 collection
        logger.info(
            "创建 collection: %s, dim=%d, metric=IP",
            collection_name, self.dimension,
        )
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,  # 关键：由 add() 显式写入 vector_id=0..N-1，保证对齐
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            ),
        ]
        schema = CollectionSchema(fields, description="RAG production vectors")
        self._collection = Collection(
            collection_name, schema=schema, using=self._alias
        )
        # 根据索引类型创建索引
        if self.index_type == "hnsw":
            index_params = {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {
                    "M": self.hnsw_m,
                    "efConstruction": self.hnsw_ef_construction,
                },
            }
        else:  # 默认 ivf
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": self.ivf_nlist},
            }
        self._collection.create_index(
            field_name="embedding", index_params=index_params
        )
        logger.info(
            "collection 创建并建索引完成: %s, index_type=%s",
            collection_name, self.index_type,
        )

    # ---- 接口实现 ----

    def add(self, vectors):
        """批量写入向量。

        参数:
            vectors: numpy.ndarray 或 list，shape=(n, dimension)
        """
        import numpy as np

        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError(
                "vectors 必须是二维数组, got shape={}".format(vectors.shape)
            )
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                "向量维度不匹配: expected={}, got={}".format(
                    self.dimension, vectors.shape[1]
                )
            )

        self._ensure_collection(self.collection_name)
        t = time.time()
        logger.info("Milvus 写入向量: count=%d", vectors.shape[0])

        # 显式生成 vector_id=0..N-1 并作为主键写入（auto_id=False）
        # 保证 Milvus 返回的 id 与 chunks enumerate 顺序、FAISS position 严格对齐
        n = vectors.shape[0]
        vector_ids = list(range(n))
        self._collection.insert([vector_ids, vectors.tolist()])
        self._collection.flush()
        logger.info(
            "Milvus 写入完成: %.3fs, total_rows=%d",
            time.time() - t, self._collection.num_entities,
        )

    def search(self, query_vector, top_k):
        """检索最相似的 top_k 个向量。

        返回格式与 FAISSStore 完全一致:
            (scores, ids) 二元组
            scores[0] / ids[0] 是第一批结果
            ids[0][i] 为 int 索引（对应 metadata 的 key），-1 表示无结果

        参数:
            query_vector: 单条查询向量，shape=(1, dimension) 或 (dimension,)
            top_k: 返回结果数
        """
        import numpy as np

        if self._collection is None:
            raise RuntimeError(
                "collection 未加载，请先调用 load(path) 或 add(vectors)"
            )

        query_vector = np.asarray(query_vector, dtype="float32")
        # 统一为二维
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        t = time.time()
        self._collection.load()

        # 按索引类型设置搜索参数
        if self.index_type == "hnsw":
            search_params = {"metric_type": "IP", "params": {"ef": self.hnsw_ef_search}}
        else:  # ivf
            search_params = {"metric_type": "IP", "params": {"nprobe": self.ivf_nprobe}}

        results = self._collection.search(
            data=query_vector.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=None,
        )
        logger.info(
            "Milvus 检索: %.3fs, top_k=%d, 返回=%d, index_type=%s",
            time.time() - t, top_k, len(results[0]) if results else 0,
            self.index_type,
        )

        # 转换为 FAISSStore 兼容格式: (scores_2d, ids_2d)
        scores_list: List[float] = []
        ids_list: List[int] = []
        if results and len(results) > 0:
            for hit in results[0]:
                scores_list.append(float(hit.score))
                ids_list.append(int(hit.id))

        # 不足 top_k 时补 -1（与 FAISS 行为一致）
        while len(ids_list) < top_k:
            scores_list.append(-1.0)
            ids_list.append(-1)

        scores = np.array([scores_list], dtype="float32")
        ids = np.array([ids_list], dtype="int64")
        return scores, ids

    def save(self, path):
        """持久化索引。

        Milvus 数据存储在远端服务，无需本地文件。
        path 参数作为 collection name（覆盖初始化时的 collection_name），
        保持与 FAISSStore.save(path) 接口一致。
        """
        if self._collection is None:
            raise RuntimeError("无数据可保存，请先 add(vectors)")

        # path 作为 collection name
        collection_name = str(path)
        if collection_name != self.collection_name:
            # 重命名 collection（Milvus 不直接支持重命名，这里用新名字重建）
            logger.info(
                "save: collection name 从 %s 切换到 %s",
                self.collection_name, collection_name,
            )
            self.collection_name = collection_name
            self._ensure_collection(collection_name)
        # Milvus 数据已在远端持久化，flush 确保写入磁盘
        self._collection.flush()
        logger.info("Milvus 索引已保存: collection=%s", collection_name)

    def load(self, path):
        """加载索引。

        连接 Milvus 服务并加载指定 collection。
        path 参数作为 collection name，保持与 FAISSStore.load(path) 接口一致。
        """
        collection_name = str(path)
        self.collection_name = collection_name
        self._ensure_collection(collection_name)
        self._collection.load()
        logger.info(
            "Milvus 索引已加载: collection=%s, rows=%d",
            collection_name, self._collection.num_entities,
        )

    # ---- 辅助方法 ----

    def drop(self, collection_name: Optional[str] = None):
        """删除 collection（慎用，数据不可恢复）。

        参数:
            collection_name: 指定 collection，默认用当前 collection_name
        """
        self._connect()
        name = collection_name or self.collection_name
        if utility.has_collection(name, using=self._alias):
            utility.drop_collection(name, using=self._alias)
            logger.warning("已删除 collection: %s", name)
        else:
            logger.info("collection 不存在，无需删除: %s", name)

    def count(self) -> int:
        """返回当前 collection 中的向量数。"""
        if self._collection is None:
            return 0
        return self._collection.num_entities

    def close(self):
        """断开 Milvus 连接。"""
        if self._connected:
            try:
                connections.disconnect(self._alias)
            except Exception:
                pass
            self._connected = False
            logger.info("Milvus 连接已断开")
