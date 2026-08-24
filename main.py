"""RAG 主程序：交互式问答入口。

从 config.yaml 读取配置，初始化 RAGService，提供交互式查询。
支持命令行单次查询和交互式循环查询。

前置条件:
  - 索引文件已生成：data/index/{strategy}/faiss.index + metadata.json
  - 环境变量 DASHSCOPE_API_KEY 已设置（generation.backend=qwen 时）

使用示例:
  # 交互式（默认）
  python main.py

  # 单次查询
  python main.py --query "铁矿近期供需如何？"

  # 指定策略和模式
  python main.py --strategy recursive --mode hybrid --query "铁矿供需"

  # 跳过 rerank（更快，精度略降）
  python main.py --no-rerank --query "铁矿供需"
"""
import argparse
import os
import sys
from pathlib import Path

# 注入项目根目录到 sys.path（保证从任意 cwd 运行都能 import app）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import Config
from app.core.env import load_env
from app.core.logger import get_logger
from app.rag.service import get_service, reset_service_cache

logger = get_logger(__name__)


def print_response(response):
    """打印 RAGResponse 的完整结果。"""
    print()
    print("=" * 60)
    print("Answer")
    print("=" * 60)
    print(response.answer or "(未生成)")

    print()
    print("=" * 60)
    print("Context ({} chars)".format(len(response.context)))
    print("=" * 60)
    print(response.context)

    print()
    print("=" * 60)
    print("Stats")
    print("=" * 60)
    stats = response.stats
    for k, v in stats.items():
        if k == "budget_utilization":
            print("  {}: {:.2%}".format(k, v))
        else:
            print("  {}: {}".format(k, v))

    print()
    print("-" * 60)
    print("Chunks ({} 条)".format(len(response.chunks)))
    for i, c in enumerate(response.chunks, 1):
        tags = []
        if c.get("merged"):
            tags.append("merged:{}".format("+".join(c.get("chunk_ids", []))))
        if c.get("compressed"):
            tags.append("compressed({}->{})".format(
                c.get("original_tokens"), c.get("compressed_tokens")
            ))
        if c.get("budget_truncated"):
            tags.append("truncated")
        tag_str = "  [{}]".format(", ".join(tags)) if tags else ""
        score = c.get("rerank_score", c.get("score", 0))
        print("  [{}] score={:.4f} {}{}".format(
            i, score, c.get("chunk_id", "unknown"), tag_str
        ))

    if response.citations:
        print()
        print("-" * 60)
        print("Citations ({} 条引用)".format(len(response.citations)))
        for c in response.citations:
            print("  [{}] {} | {} | offset={}~{} | {}".format(
                c["number"],
                c["chunk_id"],
                c["file_name"],
                c["start_offset"],
                c["end_offset"],
                c["content_preview"],
            ))


def main():
    parser = argparse.ArgumentParser(description="RAG 主程序：交互式问答")
    parser.add_argument("--query", "-q", help="单次查询（不指定则进入交互式）")
    parser.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 rerank")
    args = parser.parse_args()

    # 加载 .env（设置 DASHSCOPE_API_KEY 等环境变量）
    load_env()

    # 读 config，如果 .env 指定了 DASHSCOPE_MODEL 则覆盖 yaml 的 model_name
    config = Config()
    env_model = os.getenv("DASHSCOPE_MODEL")
    if env_model:
        config.data.setdefault("generation", {})["model_name"] = env_model
        logger.info("用 .env 的 DASHSCOPE_MODEL 覆盖 model_name: %s", env_model)

    print("=" * 60)
    print("RAG 主程序")
    print("=" * 60)
    print("Strategy: {}".format(args.strategy))
    print("Mode: {}".format(args.mode))
    print("Rerank: {}".format("off" if args.no_rerank else "on"))
    print("LLM: {} (backend={})".format(
        config.generation_model, config.generation_backend
    ))
    print()

    # 初始化 service（从 config 读全部参数，含 LLM backend）
    logger.info("初始化 RAGService: strategy=%s, mode=%s, use_rerank=%s",
                args.strategy, args.mode, not args.no_rerank)
    service = get_service(
        config=config,
        strategy=args.strategy,
        mode=args.mode,
        use_rerank=not args.no_rerank,
    )

    if args.query:
        # 单次查询模式
        response = service.query(args.query)
        print_response(response)
    else:
        # 交互式模式
        print("进入交互式查询（输入 exit 退出）")
        print()
        while True:
            try:
                query = input("问> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "退出"):
                break
            try:
                response = service.query(query)
                print_response(response)
            except Exception as e:
                print("查询失败: {}".format(e))
                logger.error("查询异常: %s", e, exc_info=True)
                continue
        print("再见")


if __name__ == "__main__":
    main()
