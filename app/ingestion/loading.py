"""data/raw 文档加载 mixin（rebuild / 增量重建共用）。"""
from pathlib import Path
from typing import List

from app.core.logger import get_logger

logger = get_logger(__name__)


class LoadingMixin:
    """文档加载 mixin（由 IndexWriter 组合）。"""

    # ---- 内部：文档加载 ----
    def _load_all_documents(self, tenant_id: str = "default") -> List:
        """加载某租户 data/raw/ 目录下所有支持的文档。"""
        import importlib

        from app.ingestion.loader.registry import LOADER_MAP as loader_map

        raw_dir = self.config.raw_dir_for(tenant_id)
        if not raw_dir.exists():
            logger.error("data/raw 目录不存在 (tenant=%s): %s", tenant_id, raw_dir)
            raise FileNotFoundError("data/raw 目录不存在 (tenant={})".format(tenant_id))

        files = [f for f in raw_dir.iterdir() if f.suffix.lower() in loader_map]
        logger.info(
            "扫描 data/raw/: 找到 %d 个可处理文件: %s",
            len(files), [f.name for f in files],
        )
        if not files:
            logger.error("data/raw/ 下无可处理的文件")
            raise FileNotFoundError("data/raw/ 下无可处理的文件")

        documents = []
        for f in files:
            loader_cls_path = loader_map[f.suffix.lower()]
            module_path, cls_name = loader_cls_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            loader_cls = getattr(module, cls_name)
            loader = loader_cls()
            doc = loader.load(str(f))
            documents.append(doc)
            logger.info(
                "文档加载: file=%s, doc_id=%s, content_len=%d",
                f.name, doc.document_id, len(doc.content),
            )

        logger.info("全部文档加载完成: %d 个", len(documents))
        return documents

    def _load_single_document(self, file_path: Path):
        """加载单个文档。"""
        import importlib

        from app.ingestion.loader.registry import LOADER_MAP as loader_map

        loader_cls_path = loader_map.get(file_path.suffix.lower())
        if not loader_cls_path:
            logger.error("不支持的文件类型: %s", file_path.suffix)
            raise ValueError("不支持的文件类型: {}".format(file_path.suffix))
        module_path, cls_name = loader_cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        loader_cls = getattr(module, cls_name)
        loader = loader_cls()
        doc = loader.load(str(file_path))
        logger.info(
            "单文档加载: file=%s, doc_id=%s, content_len=%d",
            file_path.name, doc.document_id, len(doc.content),
        )
        return doc
