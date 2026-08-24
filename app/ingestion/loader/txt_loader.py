from pathlib import Path

from app.ingestion.document import Document
from app.ingestion.loader.base import BaseLoader


class TxtLoader(BaseLoader):

    def load(self, file_path):

        path = Path(file_path)

        content = path.read_text(encoding="utf-8")

        return Document(
            document_id=path.stem,
            content=content,
            metadata={
                "file_name": path.name,
                "file_path": str(path),
                "file_type": "txt",
            },
        )
