import time

import faiss
import numpy as np

from app.core.logger import get_logger
from app.vector.base import BaseVectorStore

logger = get_logger(__name__)


class FAISSStore(BaseVectorStore):
    """FAISS 向量存储后端（本地文件索引，内积相似度）。

    支持三种索引类型:
      - flat:  IndexFlatIP，暴力精确检索（默认，适合 <10 万向量）
      - ivf:   IndexIVFFlat，倒排索引 + nprobe 近似检索（需先 train）
      - hnsw:  IndexHNSWFlat，分层近邻图索引（无需训练，高召回低延迟）
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        ivf_nlist: int = 128,
        ivf_nprobe: int = 16,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
        **kwargs,
    ):
        self.dimension = dimension
        self.index_type = index_type.lower()

        # 保存索引参数（save/load 后恢复搜索参数用）
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # 标记 IVF 是否已训练（add 时若未训练则先 train）
        self._ivf_trained = False

        self._build_index()

    def _build_index(self):
        """根据 index_type 创建 FAISS 索引实例。"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("FAISS 索引创建: type=flat(FlatIP), dim=%d", self.dimension)

        elif self.index_type == "ivf":
            # 量化器: 内积 + nlist 聚类中心
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT
            )
            # 设置搜索时的 nprobe
            self.index.nprobe = self.ivf_nprobe
            logger.info(
                "FAISS 索引创建: type=ivf(IVFFlat), dim=%d, nlist=%d, nprobe=%d",
                self.dimension, self.ivf_nlist, self.ivf_nprobe,
            )

        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(
                self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efConstruction = self.hnsw_ef_construction
            self.index.hnsw.efSearch = self.hnsw_ef_search
            logger.info(
                "FAISS 索引创建: type=hnsw(HNSWFlat), dim=%d, M=%d, efConstruction=%d, efSearch=%d",
                self.dimension, self.hnsw_m,
                self.hnsw_ef_construction, self.hnsw_ef_search,
            )

        else:
            raise ValueError(
                "不支持的索引类型: {}（可选: flat / ivf / hnsw）".format(self.index_type)
            )

    def add(self, vectors):
        """批量写入向量。

        IVF 索引首次 add 前需要 train（用前 min(nlist, n) 条向量训练聚类中心）。
        """
        vectors = np.asarray(vectors, dtype="float32")

        # IVF: 首次写入前需要训练
        if self.index_type == "ivf" and not self._ivf_trained:
            n = vectors.shape[0]
            train_n = min(self.ivf_nlist, n)
            if train_n < self.ivf_nlist:
                logger.info(
                    "IVF 训练: 向量数=%d < nlist=%d，自动调低 nlist 到 %d",
                    n, self.ivf_nlist, train_n,
                )
                # 重建索引用调低后的 nlist
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, train_n, faiss.METRIC_INNER_PRODUCT
                )
                self.index.nprobe = min(self.ivf_nprobe, train_n)

            logger.info("IVF 训练开始: train_n=%d", train_n)
            self.index.train(vectors[:train_n])
            self._ivf_trained = True
            logger.info("IVF 训练完成")

        self.index.add(vectors)
        logger.info(
            "FAISS 写入向量: count=%d, total=%d",
            vectors.shape[0], self.index.ntotal,
        )

    def search(self, query_vector, top_k):
        """检索最相似的 top_k 个向量。"""
        t = time.time()
        query_vector = np.asarray(query_vector, dtype="float32")

        # HNSW: 设置搜索时的 ef（需要 >= top_k）
        if self.index_type == "hnsw":
            self.index.hnsw.efSearch = max(self.hnsw_ef_search, top_k)

        scores, ids = self.index.search(query_vector, top_k)

        logger.info(
            "FAISS 检索: %.3fs, top_k=%d, 返回=%d, index_type=%s",
            time.time() - t, top_k, len(ids[0]) if len(ids) > 0 else 0,
            self.index_type,
        )
        return scores, ids

    def save(self, path):
        faiss.write_index(self.index, path)
        logger.info("FAISS 索引保存: path=%s, type=%s", path, self.index_type)

    def load(self, path):
        t = time.time()
        logger.info("加载 FAISS 索引: %s", path)
        self.index = faiss.read_index(path)
        # 恢复搜索参数（save 时不会序列化 nprobe/efSearch）
        if self.index_type == "ivf" and hasattr(self.index, "nprobe"):
            self.index.nprobe = self.ivf_nprobe
        if self.index_type == "hnsw" and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.hnsw_ef_search
        # IVF 索引从文件加载后已训练
        self._ivf_trained = True
        logger.info(
            "FAISS 索引加载完成: %.3fs, type=%s, total=%d",
            time.time() - t, self.index_type, self.index.ntotal,
        )
