import re

from app.ingestion.document import Document


class DocumentCleaner:

    def clean(self, document: Document) -> Document:

        content = document.content

        # 1. 去除 HTML 空白实体
        content = content.replace("\xa0", " ")

        # 2. 统一换行
        content = content.replace("\r\n", "\n")

        content = content.replace("\r", "\n")

        # 3. 删除多余空格
        content = re.sub(r"[ \t]+", " ", content)

        # 4. 删除连续空行
        content = re.sub(r"\n{3,}", "\n\n", content)

        # 5. 去除首尾空白
        content = content.strip()

        return Document(
            document_id=document.document_id,
            content=content,
            metadata=document.metadata.copy(),
        )
