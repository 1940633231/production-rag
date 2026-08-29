"""自动构建检索评估数据集：基于 data/raw 现有文档 + 小节标题定位 spans。

背景:
  原 retrieval_dataset.json 的 relevant_spans 标注自一份"铁矿+焦煤综合报告"，
  该文档不在当前知识库，导致检索 Recall 恒为 0（数据错位）。

本脚本:
  1. 用与摄入链路一致的 loader + DocumentCleaner 处理 data/raw 文档
  2. 按"小节标题"在 clean 后的 content 中定位 spans（偏移与 chunk 对齐）
  3. 覆盖现有 data/evaluation/retrieval_dataset.json

用法:
  .venv\\Scripts\\python.exe scripts/build_eval_dataset.py
  .venv\\Scripts\\python.exe scripts/build_eval_dataset.py --output data/evaluation/retrieval_dataset.json
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.cleaner.cleaner import DocumentCleaner  # noqa: E402
from app.ingestion.loader.html_loader import HtmlLoader  # noqa: E402
from app.ingestion.loader.pdf_loader import PdfLoader  # noqa: E402
from app.ingestion.loader.txt_loader import TxtLoader  # noqa: E402

DEFAULT_OUTPUT = str(PROJECT_ROOT / "data/evaluation/retrieval_dataset.json")

# query 定义：目标文档 + 相关小节标题（标题需与 clean 后 content 完全一致）
QUERY_SPECS = [
    {
        "query": "铁矿近期供需和价格走势如何？",
        "file": "data/raw/test.txt",
        "sections": ["二、当前市场核心运行特征", "三、短期行情预判"],
    },
    {
        "query": "云创科技 2026 年第三季度营收和利润表现如何？",
        "file": "data/raw/financial_report_q3.txt",
        "sections": ["1. 营收概况", "2. 利润表现"],
    },
    {
        "query": "人工智能的发展经历了哪些阶段？",
        "file": "data/raw/ai_history.txt",
        "sections": [
            "1. 早期阶段", "2. 黄金时代", "3. 第一次 AI 冬天",
            "4. 专家系统的兴起", "5. 第二次 AI 冬天", "6. 现代 AI",
        ],
    },
    {
        "query": "公司员工的年假和病假制度是怎样的？",
        "file": "data/raw/company_handbook.html",
        "sections": ["3. 带薪休假"],
    },
]

_LOADERS = {".txt": TxtLoader, ".html": HtmlLoader, ".pdf": PdfLoader}


def load_and_clean(file: str):
    """加载文档并 clean，返回与摄入链路一致的 content。"""
    path = Path(file)
    loader_cls = _LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError("不支持的文件类型: {}".format(path.suffix))
    doc = loader_cls().load(str(path))
    return DocumentCleaner().clean(doc).content


def locate_sections(content: str, sections):
    """按小节标题在 content 中定位偏移，返回相关 span 列表。"""
    positions = []
    for title in sections:
        idx = content.find(title)
        if idx < 0:
            print("[WARN] 未找到小节标题: {}".format(title))
            continue
        positions.append((idx, title))
    positions.sort()
    if not positions:
        return []

    spans = []
    for i, (start, _title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        if end > start:
            spans.append({"start": start, "end": end})
    return spans


def build():
    parser = argparse.ArgumentParser(description="构建检索评估数据集")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    samples = []
    for spec in QUERY_SPECS:
        content = load_and_clean(spec["file"])
        spans = locate_sections(content, spec["sections"])
        if not spans:
            print("[WARN] 未生成任何 span: query={}, file={}".format(
                spec["query"], spec["file"]
            ))
            continue
        samples.append({
            "query": spec["query"],
            "relevant_spans": spans,
        })
        print("query: {}".format(spec["query"]))
        print("  file: {}, content_len={}, spans={}".format(
            spec["file"], len(content),
            [(s["start"], s["end"]) for s in spans],
        ))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n数据集已生成: {} ({} 条 query)".format(out, len(samples)))


if __name__ == "__main__":
    build()
