from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RetrievalResult:
    """单条检索结果（原始字段，供文档化使用）。"""
    vector_id: int
    score: float
    chunk_id: str
    content: str
    metadata: Dict
    start_offset: int = 0
    end_offset: int = 0


@dataclass
class RAGResponse:
    """RAG 流水线最终响应。

    字段:
        query: 用户查询
        context: 最终拼装的上下文文本（带 [1][2] 编号，送 LLM）
        chunks: 最终保留的 chunks 列表（含 merged/compressed 等标记，供 citation）
        stats: ContextManager 各阶段统计（input/after_dedup/after_merge/预算等）
        answer: LLM 生成的答案，generation 模块补齐后填充；None 表示未生成
        citations: 从 answer 提取的引用列表（含 number/file_name/offset），由 CitationExtractor 填充
    """
    query: str
    context: str
    chunks: List[Dict] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    answer: Optional[str] = None
    citations: List[Dict] = field(default_factory=list)
