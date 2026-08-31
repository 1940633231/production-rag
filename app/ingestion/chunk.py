from dataclasses import dataclass, field
from typing import Dict


def vector_id_for(chunk_id: str) -> int:
    """由 chunk_id 派生稳定的 int64 向量 ID（确定性、可复现）。

    稳定 ID 索引：删除某向量后其余向量 id 不变，无需重建索引。
    使用 sha256 前 15 位十六进制转 int（< 2^60，落在 int64 范围内），
    冲突概率可忽略。
    """
    import hashlib
    return int(hashlib.sha256(str(chunk_id).encode("utf-8")).hexdigest()[:15], 16)


@dataclass
class Chunk:

    chunk_id: str

    document_id: str

    chunk_index: int

    content: str

    # chunk 在所属文档（clean 后的 content）中的字符偏移区间 [start_offset, end_offset)
    # 用于 span 级检索评测，与切片策略解耦
    start_offset: int = 0

    end_offset: int = 0

    metadata: Dict = field(default_factory=dict)

    # 稳定向量 ID（写入链路分配，= vector_id_for(chunk_id)），作为 FAISS/Milvus 显式主键
    vector_id: int = 0
