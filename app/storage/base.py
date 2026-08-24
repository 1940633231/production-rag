"""分块仓库抽象层：统一 MySQL / ES / 文件后端的读接口。

职责：
  - 定义 BaseChunkRepository 抽象基类，供 Retriever / BM25Search 依赖
  - 提供 MetadataChunkRepository 适配器（metadata.json 文件降级）

ES 后端实现见 app/storage/es_repository.py（ChunkESRepository）。

设计说明：
  - id 为向量库位置索引（int），与 metadata.json 的 key 语义一致
  - 各后端实现需保证 list_all() 返回顺序与向量入库顺序一致
  - 写入接口（insert/batch_insert）仍由具体后端自行定义，不在基类约束
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


class BaseChunkRepository(ABC):
    """分块仓库抽象基类：统一读接口。

    id 语义：向量库位置索引（int），对应 metadata.json 的 string key。
    Retriever 拿到 vector_store.search 返回的 int index 后，调用 get_by_id
    获取 chunk 详细信息（content / offsets / metadata）。
    """

    @abstractmethod
    def get_by_id(self, id: int) -> Optional[Dict]:
        """按向量位置 ID 查询单个 chunk。

        返回 dict 格式与 metadata.json 条目一致：
            {"chunk_id", "content", "start_offset", "end_offset", "metadata"}
        不存在时返回 None。
        """

    @abstractmethod
    def batch_get_by_ids(self, ids: List[int]) -> List[Dict]:
        """批量按向量位置 ID 查询 chunks。

        返回列表顺序与输入 ids 顺序一致；不存在的 id 跳过。
        """

    @abstractmethod
    def list_all(self) -> List[Dict]:
        """返回所有 chunks，按向量位置顺序排列。

        供 BM25 等需要全量语料的检索器使用。列表 index 即为向量位置 ID。
        """

    @abstractmethod
    def count(self) -> int:
        """chunk 总数。"""


class MetadataChunkRepository(BaseChunkRepository):
    """基于 metadata.json 的文件后端适配器（降级方案）。

    将 MetadataStore.load() 返回的 {str(id): chunk_dict} 包装为
    BaseChunkRepository 接口，使 Retriever / BM25 无感知后端切换。
    """

    def __init__(self, metadata: Dict[str, Dict]):
        # 预排序：按 int(key) 排列，保证 list_all 顺序与向量位置一致
        self._sorted_ids = sorted(metadata.keys(), key=lambda x: int(x))
        self._metadata = metadata

    def get_by_id(self, id: int) -> Optional[Dict]:
        return self._metadata.get(str(id))

    def batch_get_by_ids(self, ids: List[int]) -> List[Dict]:
        result = []
        for i in ids:
            doc = self._metadata.get(str(i))
            if doc is not None:
                result.append(doc)
        return result

    def list_all(self) -> List[Dict]:
        return [self._metadata[k] for k in self._sorted_ids]

    def count(self) -> int:
        return len(self._metadata)
