"""Citation 提取：从 LLM answer 中解析 [1][2] 引用编号，映射到具体 chunk 源。

工作流：
  1. 正则提取 answer 中的 [1][2] 编号（保留出现顺序，去重）
  2. 编号映射到 RAGResponse.chunks[index-1]（1-based → 0-based）
  3. 返回 citation 列表，每条含 number/file_name/offset/content_preview

前置条件（已由其他模块保证）：
  - PromptBuilder.SYSTEM_PROMPT 约束 LLM 用 [1][2] 格式引用
  - ContextManager._format_context 输出的 context 带 [1][2] 编号
  - RAGResponse.chunks 保留完整元信息（chunk_id/offset/metadata）
"""
import re
from typing import Dict, List

from app.core.logger import get_logger

logger = get_logger(__name__)


class CitationExtractor:
    """从 LLM answer 提取 [1][2] 引用并映射到 chunks。"""

    # 匹配 [1] [12] 等单个数字编号
    REF_PATTERN = re.compile(r"\[(\d+)\]")

    def extract(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """从 answer 提取引用编号，映射到 chunks。

        参数:
            answer: LLM 生成的答案（含 [1][2] 编号）
            chunks: RAGResponse.chunks（1-based 编号对应 index+1）

        返回:
            [
                {
                    "number": 1,
                    "chunk_id": "doc1_chunk_0",
                    "file_name": "铁矿周报.txt",
                    "document_id": "doc1",
                    "start_offset": 0,
                    "end_offset": 48,
                    "content_preview": "铁矿石供应端..."（前 50 字符）
                },
                ...
            ]
        """
        if not answer or not chunks:
            return []

        # 提取所有 [数字] 标记
        matches = self.REF_PATTERN.findall(answer)
        if not matches:
            logger.info("answer 中未找到引用编号")
            return []

        # 去重并保留首次出现顺序
        seen = set()
        ordered_numbers = []
        for m in matches:
            n = int(m)
            if n not in seen:
                ordered_numbers.append(n)
                seen.add(n)

        # 映射编号到 chunks（1-based → 0-based）
        citations = []
        for n in ordered_numbers:
            idx = n - 1
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                metadata = chunk.get("metadata", {})
                citations.append({
                    "number": n,
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "file_name": metadata.get("file_name", "unknown"),
                    "document_id": metadata.get("document_id", "unknown"),
                    "start_offset": chunk.get("start_offset", 0),
                    "end_offset": chunk.get("end_offset", 0),
                    "content_preview": chunk.get("content", "")[:50],
                })
            else:
                logger.warning(
                    "引用编号 %d 超出 chunks 范围 (1-%d)", n, len(chunks)
                )

        logger.info(
            "Citation 提取完成: answer 中有 %d 个引用标记, 映射到 %d 条 citation",
            len(matches), len(citations),
        )
        return citations
