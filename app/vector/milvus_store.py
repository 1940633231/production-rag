"""Milvus 向量存储：基于 pymilvus MilvusClient 的分布式向量数据库后端。

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
    - 使用 MilvusClient（非弃用的 ORM Collection API，兼容 pymilvus 3.1+）
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
    from pymilvus import MilvusClient, DataType
    from pymilvus.milvus_client.index import IndexParams

    _PYMILVUS_AVAILABLE = True
except ImportError:
    _PYMILVUS_AVAILABLE = False


class MilvusStore(BaseVectorStore):
    """Milvus 向量存储后端（基于 MilvusClient API）。

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
        self._client: Optional[Any] = None
        self._connected = False
        # 当前生效的 collection（add/load 后非 None，search 依赖它）
        self._collection_name: Optional[str] = None

    # ---- 连接管理 ----

    def _connect(self):
        """连接 Milvus 服务（延迟连接，避免初始化时阻塞）。"""
        if self._connected:
            return
        t = time.time()
        uri = "http://{}:{}".format(self.host, self.port)
        logger.info("连接 Milvus: %s", uri)
        self._client = MilvusClient(uri=uri)
        self._connected = True
        logger.info("Milvus 连接成功: %.3fs", time.time() - t)

    def _build_index_params(self) -> IndexParams:
        """按索引类型构建 IndexParams（IVF_FLAT / HNSW）。"""
        index_params = IndexParams()
        if self.index_type == "hnsw":
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="IP",
                params={"M": self.hnsw_m, "efConstruction": self.hnsw_ef_construction},
            )
        else:  # 默认 ivf
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": self.ivf_nlist},
            )
        return index_params

    def _ensure_collection(self, collection_name: str):
        """确保 collection 存在，不存在则创建。

        主键 id 使用 INT64 + auto_id=False，由 add() 显式传入 vector_id=0..N-1，
        保证与 chunks enumerate 顺序、FAISS store 的 position 严格对齐。
        """
        self._connect()
        self._collection_name = collection_name

        if self._client.has_collection(collection_name):
            logger.info(
                "加载已有 collection: %s, rows=%d",
                collection_name, self._row_count(collection_name),
            )
            return

        # 创建新 collection（schema + 指定索引类型，create_collection 会建索引并加载）
        logger.info(
            "创建 collection: %s, dim=%d, metric=IP, index=%s",
            collection_name, self.dimension, self.index_type,
        )
        schema = self._client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)
        self._client.create_collection(
            collection_name,
            schema=schema,
            index_params=self._build_index_params(),
        )
        logger.info(
            "collection 创建并建索引完成: %s, index_type=%s",
            collection_name, self.index_type,
        )

    def _row_count(self, collection_name: str) -> int:
        """返回 collection 的持久化行数。"""
        try:
            stats = self._client.get_collection_stats(collection_name)
            return int(stats.get("row_count", 0))
        except Exception as e:
            logger.debug("获取 collection 行数失败: %s", e)
            return 0

    # ---- 接口实现 ----

    def add(self, vectors, ids=None):
        """批量写入向量。

        参数:
            vectors: numpy.ndarray 或 list，shape=(n, dimension)
            ids: 可选，与 vectors 一一对应的显式 int64 主键；
                 不传时用末尾序号 0..N-1（向后兼容）
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

        # 显式主键：优先用传入 ids（稳定 ID），否则用末尾序号 0..N-1（向后兼容）
        if ids is not None:
            ids_list = [int(i) for i in ids]
            if len(ids_list) != vectors.shape[0]:
                raise ValueError(
                    "ids 长度必须与 vectors 一致: ids={}, vectors={}".format(
                        len(ids_list), vectors.shape[0]
                    )
                )
        else:
            current = self._row_count(self._collection_name)
            ids_list = list(range(current, current + vectors.shape[0]))

        data = [
            {"id": ids_list[i], "embedding": vectors[i].tolist()}
            for i in range(vectors.shape[0])
        ]
        self._client.insert(self.collection_name, data)
        self._client.flush(self.collection_name)
        logger.info(
            "Milvus 写入完成: %.3fs, total_rows=%d",
            time.time() - t, self._row_count(self.collection_name),
        )

    def remove(self, ids):
        """按 id 删除向量（其余向量 id 不变，无需重建）。

        参数:
            ids: 待删除的 int64 id 列表
        """
        if self._collection_name is None:
            raise RuntimeError(
                "collection 未加载，请先调用 load(path) 或 add(vectors)"
            )
        t = time.time()
        ids_list = [int(i) for i in ids]
        self._connect()
        self._client.delete(self._collection_name, ids=ids_list)
        self._client.flush(self._collection_name)
        logger.info(
            "Milvus 删除向量: ids=%s, %.3fs",
            ids_list, time.time() - t,
        )
        return len(ids_list)

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

        if self._collection_name is None:
            raise RuntimeError(
                "collection 未加载，请先调用 load(path) 或 add(vectors)"
            )

        query_vector = np.asarray(query_vector, dtype="float32")
        # 统一为二维
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        t = time.time()
        self._client.load_collection(self._collection_name)

        # 按索引类型设置搜索参数
        if self.index_type == "hnsw":
            search_params = {"metric_type": "IP", "params": {"ef": self.hnsw_ef_search}}
        else:  # ivf
            search_params = {"metric_type": "IP", "params": {"nprobe": self.ivf_nprobe}}

        results = self._client.search(
            self._collection_name,
            data=query_vector.tolist(),
            anns_field="embedding",
            search_params=search_params,
            limit=top_k,
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
                scores_list.append(float(hit.get("distance", -1.0)))
                ids_list.append(int(hit.get("id", -1)))

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
        if self._collection_name is None:
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
        self._client.flush(self.collection_name)
        logger.info("Milvus 索引已保存: collection=%s", collection_name)

    def load(self, path):
        """加载索引。

        连接 Milvus 服务并加载指定 collection。
        path 参数作为 collection name，保持与 FAISSStore.load(path) 接口一致。
        """
        collection_name = str(path)
        self.collection_name = collection_name
        self._ensure_collection(collection_name)
        self._client.load_collection(collection_name)
        logger.info(
            "Milvus 索引已加载: collection=%s, rows=%d",
            collection_name, self._row_count(collection_name),
        )

    # ---- 辅助方法 ----

    def drop(self, collection_name: Optional[str] = None):
        """删除 collection（慎用，数据不可恢复）。

        参数:
            collection_name: 指定 collection，默认用当前 collection_name
        """
        self._connect()
        name = collection_name or self.collection_name
        if self._client.has_collection(name):
            self._client.drop_collection(name)
            logger.warning("已删除 collection: %s", name)
        else:
            logger.info("collection 不存在，无需删除: %s", name)

    def count(self) -> int:
        """返回当前 collection 中的向量数。"""
        if self._client is None or self._collection_name is None:
            return 0
        return self._row_count(self._collection_name)

    def close(self):
        """断开 Milvus 连接。"""
        if self._connected:
            try:
                self._client.close()
            except Exception:
                pass
            self._connected = False
            self._client = None
            logger.info("Milvus 连接已断开")
