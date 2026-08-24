from dataclasses import dataclass, field
from typing import Dict


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
