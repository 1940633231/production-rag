"""Context Manager 端到端演示脚本。

串联完整流水线：
  Retriever（mock/vector/bm25/hybrid）
    → Reranker（Cross-Encoder，可选）
    → ContextManager（去重/合并/压缩/排序/预算控制）
    → 最终 context 文本（暂不接 LLM）

使用示例：
  # 最简：mock 检索 + 跳过 rerank（零外部依赖，只需 pyyaml，最快验证）
  python scripts/context_demo.py --strategy recursive --mode mock --no-rerank

  # 真实 BM25 + 跳过 rerank（需 jieba + rank-bm25）
  python scripts/context_demo.py --mode bm25 --no-rerank

  # 完整：hybrid + rerank（需 faiss + sentence-transformers + jieba）
  python scripts/context_demo.py --mode hybrid

  # 自定义 query
  python scripts/context_demo.py --query "港口库存变化如何？" --mode mock --no-rerank

  # 小预算触发压缩（验证 TokenBudget 控制）
  python scripts/context_demo.py --mode mock --no-rerank --max-tokens 300
"""
import argparse
import sys
from pathlib import Path

# 注入项目根目录到 sys.path，保证从任意 cwd 运行都能 import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Config
from app.storage.metadata_store import MetadataStore
from app.storage.base import MetadataChunkRepository
from app.ingestion.tokenizer import create_token_counter
from app.context.builder import ContextBuilder
from app.context.compressor import ContextCompressor
from app.context.manager import ContextManager


def mock_retrieve(metadata, top_k):
    """从 metadata 直接取前 top_k 个 chunk，模拟检索结果。

    零外部依赖（不调用 numpy/jieba/faiss），分数按序号递减模拟。
    用于快速验证 ContextManager 流水线效果。
    """
    results = []
    for idx in sorted(metadata.keys(), key=lambda x: int(x))[:top_k]:
        doc = metadata[idx]
        results.append({
            "chunk_id": doc["chunk_id"],
            "content": doc["content"],
            "start_offset": doc.get("start_offset", 0),
            "end_offset": doc.get("end_offset", 0),
            "metadata": doc.get("metadata", {}),
            "score": 1.0 - int(idx) * 0.05,  # 模拟递减分数
        })
    return results


def main():
    # =========================
    # 1. Arguments
    # =========================
    parser = argparse.ArgumentParser(description="Context Manager 端到端演示")
    parser.add_argument("--strategy", choices=["fixed", "recursive"], default="recursive")
    parser.add_argument("--mode", choices=["mock", "vector", "bm25", "hybrid"], default="mock",
                        help="mock=零依赖模拟检索；vector/bm25/hybrid=真实检索")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 Cross-Encoder 重排")
    parser.add_argument(
        "--query",
        default="铁矿近期供需和价格走势如何？",
        help="查询文本（默认与文档语义对齐的 well-formed 问题）",
    )
    parser.add_argument("--top_k", type=int, default=None, help="检索 top_k，默认读 config")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="覆盖 max_context_tokens（设小值可触发压缩）")
    args = parser.parse_args()

    # =========================
    # 2. Config
    # =========================
    config = Config()
    top_k = args.top_k or config.retrieval_top_k

    # =========================
    # 3. Index Path（mock 模式只需 metadata，不需 faiss.index）
    # =========================
    index_dir = Path("data/index") / args.strategy
    metadata_path = index_dir / "metadata.json"

    if args.mode != "mock":
        index_path = index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError("FAISS index not found: {}".format(index_path))
    if not metadata_path.exists():
        raise FileNotFoundError("Metadata not found: {}".format(metadata_path))

    # =========================
    # 4. Metadata & Retriever（按 mode 切换）
    # =========================
    metadata = MetadataStore().load(str(metadata_path))
    chunk_repo = MetadataChunkRepository(metadata)

    if args.mode == "mock":
        # 零依赖模拟检索，不初始化任何检索器
        retriever = None
    else:
        if args.mode in ("vector", "hybrid"):
            from app.embedding.model import EmbeddingModel
            from app.vector import create_vector_store
            from app.rag.retriever import Retriever
            embedding_model = EmbeddingModel(config.embedding_model)
            vector_store = create_vector_store(
                backend="faiss", dimension=embedding_model.dimension
            )
            vector_store.load(str(index_path))
            dense = Retriever(embedding_model, vector_store, chunk_repo)

        if args.mode in ("bm25", "hybrid"):
            from app.search.bm25_search import BM25Search
            sparse = BM25Search(chunk_repo)

        if args.mode == "vector":
            retriever = dense
        elif args.mode == "bm25":
            retriever = sparse
        else:  # hybrid
            from app.search.hybrid_search import HybridSearch
            retriever = HybridSearch(dense, sparse)

    # =========================
    # 5. Reranker（可选）
    # =========================
    if not args.no_rerank:
        from app.rerank.reranker import Reranker
        reranker = Reranker(config.rerank_model)
    else:
        reranker = None

    # =========================
    # 6. ContextManager（从 config 注入全部参数，支持命令行覆盖）
    # =========================
    token_counter = create_token_counter(config.tokenizer_backend)
    builder = ContextBuilder(
        dedup_span_overlap=config.dedup_span_overlap,
        dedup_jaccard=config.dedup_jaccard,
        merge_span_gap=config.merge_span_gap,
    )
    compressor = ContextCompressor(token_counter)
    context_manager = ContextManager(
        token_counter=token_counter,
        max_context_tokens=args.max_tokens or config.max_context_tokens,
        reserved_tokens=config.reserved_tokens,
        builder=builder,
        compressor=compressor,
        order_strategy=config.context_order_strategy,
    )

    # =========================
    # 7. 检索 → 重排 → Context Manager
    # =========================
    query = args.query

    if args.mode == "mock":
        results = mock_retrieve(metadata, top_k)
        pipeline = "mock_retrieve → top_{}".format(top_k)
    elif reranker:
        candidates = retriever.search(query, top_k=config.rerank_candidate_pool)
        results = reranker.rerank(query, candidates, top_k=top_k)
        pipeline = "retrieve({}) → rerank → top_{}".format(
            config.rerank_candidate_pool, top_k
        )
    else:
        results = retriever.search(query, top_k=top_k)
        pipeline = "retrieve → top_{}".format(top_k)

    context_result = context_manager.build(query, results)

    # =========================
    # 8. 打印结果
    # =========================
    print()
    print("=" * 60)
    print("Context Manager 端到端演示")
    print("=" * 60)
    print("Strategy : {}".format(args.strategy))
    print("Mode     : {}".format(args.mode))
    print("Rerank   : {}".format("off" if args.no_rerank else "on"))
    print("Query    : {}".format(query))
    print("Pipeline : {}".format(pipeline))
    if args.max_tokens:
        print("Override : max_context_tokens={}".format(args.max_tokens))
    print()

    # 8.1 Stats
    print("-" * 30 + " Stats " + "-" * 30)
    stats = context_result["stats"]
    print("input_count       : {}".format(stats["input_count"]))
    print("input_tokens      : {}".format(stats["input_tokens"]))
    print("after_dedup        : {}".format(stats["after_dedup"]))
    print("after_merge        : {}".format(stats["after_merge"]))
    print("query_tokens      : {}".format(stats["query_tokens"]))
    print("available_tokens  : {}".format(stats["available_tokens"]))
    print("compressed_count  : {}".format(stats["compressed_count"]))
    print("final_count       : {}".format(stats["final_count"]))
    print("used_tokens       : {}".format(stats["used_tokens"]))
    print("budget_utilization: {:.2%}".format(stats["budget_utilization"]))
    print()

    # 8.2 Final chunks（带 merged/compressed/budget_truncated 标记）
    print("-" * 30 + " Final Chunks " + "-" * 30)
    for i, c in enumerate(context_result["chunks"], 1):
        tags = []
        if c.get("merged"):
            tags.append("merged:{}".format("+".join(c.get("chunk_ids", []))))
        if c.get("compressed"):
            tags.append("compressed({}->{})".format(
                c.get("original_tokens"), c.get("compressed_tokens")
            ))
        if c.get("budget_truncated"):
            tags.append("budget_truncated")
        tag_str = "  [{}]".format(", ".join(tags)) if tags else ""
        score = c.get("rerank_score", c.get("score", 0))
        print("[{}] score={:.4f} span=[{},{}]{}".format(
            i, score, c["start_offset"], c["end_offset"], tag_str
        ))
        print("    {}".format(c["content"]))
        print()

    # 8.3 Final context（送给 LLM 的实际文本）
    print("-" * 30 + " Final Context ({} chars) ".format(len(context_result["context"])) + "-" * 30)
    print(context_result["context"])


if __name__ == "__main__":
    main()
