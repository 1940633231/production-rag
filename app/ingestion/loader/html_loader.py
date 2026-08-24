from pathlib import Path

from bs4 import BeautifulSoup

from app.ingestion.document import Document
from app.ingestion.loader.base import BaseLoader


class HtmlLoader(BaseLoader):

    def load(self, file_path):

        path = Path(file_path)

        html = path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html, "html.parser")

        # 删除无意义标签
        for tag in soup(["script", "style", "noscript"]):

            tag.decompose()

        # 获取正文文本
        content = soup.get_text(separator="\n")

        # 获取 title
        title = ""

        if soup.title:

            title = soup.title.get_text(strip=True)

        return Document(
            document_id=path.stem,
            content=content,
            metadata={
                "file_name": path.name,
                "file_path": str(path),
                "file_type": "html",
                "title": title,
            },
        )
