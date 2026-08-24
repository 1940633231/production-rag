"""查询脚本：通过 RAGService 走主流程检索（自动选 ES/MySQL/metadata 后端）。

替换旧的硬编码 MetadataChunkRepository 实现，统一走 create_chunk_repo 工厂，
保证 ES 后端启用时使用 vector_id 对齐的新逻辑。

用法:
  .venv\\Scripts\\python.exe scripts\\query.py
  .venv\\Scripts\\python.exe scripts\\query.py --strategy recursive
  .venv\\Scripts\\python.exe scripts\\query.py --query "你的问题" --no-rerank
  .venv\\Scripts\\python.exe scripts\\query.py --mode bm25
"""
import argparse
import sys
from pathlib import Path

# 注入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env
from app.core.env import load_env
load_env()

from app.rag.service import get_service


def main():
    parser = argparse.ArgumentParser(description="RAG 查询（走主流程 RAGService）")
    parser.add_argument(
        "--strategy", choices=["fixed", "recursive"], default="recursive",
        help="分块策略（默认 recursive）",
    )
    parser.add_argument(
        "--mode", choices=["vector", "bm25", "hybrid"], default="vector",
        help="检索模式（默认 vector）",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="单次查询，省略则进入交互模式",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="禁用 reranker（加速启动）",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="检索 top_k（默认走 config）",
    )
    args = parser.parse_args()

    use_rerank = not args.no_rerank
    service = get_service(
        strategy=args.strategy,
        mode=args.mode,
        use_rerank=use_rerank,
    )

    # 单次查询模式
    if args.query:
        _run_query(service, args.query, top_k=args.top_k)
        return 0

    # 交互模式
    print("RAG 查询已就绪（输入 exit 退出）")
    print("strategy={}, mode={}, use_rerank={}".format(
        args.strategy, args.mode, use_rerank,
    ))
    while True:
        try:
            query = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break
        _run_query(service, query, top_k=args.top_k)
    return 0


def _run_query(service, query: str, top_k=None):
    """执行一次查询并打印结果。

    直接调用 RAGPipeline 的内部组件展示检索结果（含 vector_id），
    然后跑完整 RAG 拿 answer。
    """
    print("\n========== 检索结果 ==========")
    pipeline = service._pipeline
    effective_top_k = top_k or pipeline.top_k

    # 1. 检索阶段（展示 vector_id 用于验证对齐）
    retriever = pipeline.retriever
    if hasattr(retriever, "search"):
        results = retriever.search(query, top_k=effective_top_k)
        for i, r in enumerate(results, start=1):
            print("[{}] score={:.4f} vector_id={} chunk_id={}".format(
                i, r.get("score", 0), r.get("vector_id"), r.get("chunk_id"),
            ))
            print("    content: {}".format(r.get("content", "")[:120]))
    else:
        print("(检索器不支持 .search 接口，跳过检索结果展示)")

    # 2. 完整 RAG 流程
    print("\n========== RAG 回答 ==========")
    response = service.query(query)
    answer = response.answer or "(空)"
    print(answer)
    if response.chunks:
        print("\n引用 ({} 条):".format(len(response.chunks)))
        for c in response.chunks:
            print("  - [{}] {} ({}-{})".format(
                c.get("number"), c.get("file_name"),
                c.get("start_offset"), c.get("end_offset"),
            ))


if __name__ == "__main__":
    sys.exit(main())
