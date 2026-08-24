"""向量存储层：统一接口 + 可插拔后端。

后端:
    - FAISSStore: 本地文件索引（默认，零外部服务依赖）
    - MilvusStore: 分布式向量数据库（需 Milvus 服务 + pymilvus）

工厂函数:
    from app.vector import create_vector_store
    store = create_vector_store(backend="faiss", dimension=512)
    store = create_vector_store(backend="milvus", dimension=512, host="127.0.0.1", port=19530)
"""
from typing import Any

from app.vector.base import BaseVectorStore


def create_vector_store(backend: str = "faiss", dimension: int = 512, **kwargs) -> BaseVectorStore:
    """工厂函数：按 backend 创建向量存储实例。

    参数:
        backend: "faiss"（默认）或 "milvus"
        dimension: 向量维度（必须与 embedding 模型一致）
        **kwargs: 传给具体后端的额外参数（如 MilvusStore 的 host/port）

    返回:
        BaseVectorStore 子类实例
    """
    if backend == "faiss":
        from app.vector.faiss_store import FAISSStore
        return FAISSStore(dimension=dimension)
    if backend == "milvus":
        from app.vector.milvus_store import MilvusStore
        return MilvusStore(dimension=dimension, **kwargs)
    raise ValueError("Unknown vector store backend: {}".format(backend))


__all__ = ["BaseVectorStore", "create_vector_store"]
