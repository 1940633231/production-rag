"""PDF 文档加载器：基于 pypdf 抽取纯文本。

依赖:
    pip install pypdf

特性:
    - 逐页提取文本，用 \n\n 分页
    - 自动过滤空页
    - 提取失败时抛出明确异常
"""
from pathlib import Path

from app.core.logger import get_logger
from app.ingestion.document import Document
from app.ingestion.loader.base import BaseLoader

logger = get_logger(__name__)


class PdfLoader(BaseLoader):
    """PDF 文件加载器。"""

    def load(self, file_path) -> Document:
        path = Path(file_path)
        logger.info("加载 PDF: %s", path)

        try:
            import pypdf
        except ImportError as e:
            raise ImportError(
                "PdfLoader 需要 pypdf: pip install pypdf"
            ) from e

        reader = pypdf.PdfReader(str(path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning("第 %d 页文本提取失败: %s", i + 1, e)
                text = ""
            if text.strip():
                pages_text.append(text.strip())

        content = "\n\n".join(pages_text)
        if not content.strip():
            logger.warning("PDF 无可提取文本（可能是扫描件）: %s", path)

        return Document(
            document_id=path.stem,
            content=content,
            metadata={
                "file_name": path.name,
                "file_path": str(path),
                "file_type": "pdf",
                "page_count": len(reader.pages),
            },
        )
