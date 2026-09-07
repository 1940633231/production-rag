"""加载器注册表：文件后缀 → loader 类路径（统一配置，避免多文件重复定义）。

knowledge.py 路由与 writer.py 均据此映射加载文档，保证支持的文件类型单一可信源。
"""
LOADER_MAP = {
    ".txt": "app.ingestion.loader.txt_loader.TxtLoader",
    ".html": "app.ingestion.loader.html_loader.HtmlLoader",
    ".pdf": "app.ingestion.loader.pdf_loader.PdfLoader",
    ".docx": "app.ingestion.loader.word_loader.WordLoader",
}