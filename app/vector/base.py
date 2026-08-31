"""向量存储抽象基类：统一 FAISS / Milvus 等后端的接口契约。

接口契约（从 FAISSStore 和 Retriever/IngestionPipeline 的调用推导）:
    - __init__(dimension, **kwargs)
    - add(vectors)                批量写入向量
    - search(query_vector, top_k) → (scores, ids)
        scores/ids 均为二维数组，scores[0]/ids[0] 取第一批结果
        ids[0][i] 为 int 索引（对应 metadata 的 key），-1 表示无结果
    - save(path)                  持久化索引到文件/远端
    - load(path)                  从文件/远端加载索引

子类实现:
    - FAISSStore: 本地文件索引（IndexFlatIP，内积相似度）
    - MilvusStore: 分布式向量数据库（需 Milvus 服务）
"""
from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """向量存储抽象基类。

    id 语义（稳定 ID 索引）:
      - add 可传入显式 ids；不传时用自动 id（0..N-1）
      - search 返回的 ids 即写入时的显式 id（用于映射回 chunk）
      - remove(ids) 按 id 删除向量，其余向量 id 不变（无需重建）
    """

    @abstractmethod
    def add(self, vectors, ids=None):
        """批量写入向量。

        参数:
            vectors: numpy.ndarray 或 list，shape=(n, dimension)
            ids: 可选，与 vectors 一一对应的显式 int64 id 列表；
                 不传时使用自动 id（当前末尾序号 +0..n-1）
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector, top_k):
        """检索最相似的 top_k 个向量。

        参数:
            query_vector: 单条查询向量，shape=(1, dimension) 或 (dimension,)
            top_k: 返回结果数

        返回:
            (scores, ids) 二元组:
                scores: 二维数组，scores[0] 是第一批结果的相似度分数
                ids: 二维数组，ids[0] 是对应向量写入时的显式 id（int），-1 表示无结果
        """
        raise NotImplementedError

    @abstractmethod
    def remove(self, ids):
        """按 id 删除向量（稳定 ID 索引：删除不影响其余向量 id）。

        参数:
            ids: 待删除的 id 列表
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        """持久化索引。

        FAISSStore 保存到本地文件；MilvusStore 持久化到远端 collection（path 作为 collection name）。
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        """加载索引。

        FAISSStore 从本地文件加载；MilvusStore 连接远端 collection（path 作为 collection name）。
        """
        raise NotImplementedError
