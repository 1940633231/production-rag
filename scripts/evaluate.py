"""批量评估脚本：支持检索评估 + 生成质量评估。

用法示例:

  # 检索评估（Recall@K）
  python scripts/evaluate.py --mode retrieval

  # 生成质量评估（Faithfulness + Relevance），输出 CSV 报告
  python scripts/evaluate.py --mode generation

  # 指定策略和检索模式
  python scripts/evaluate.py --mode generation --strategy recursive --search-mode hybrid

  # 指定输出路径
  python scripts/evaluate.py --mode generation --output reports/gen_eval.csv

前置条件:
  - 索引文件已生成：data/index/{strategy}/
  - 环境变量 DASHSCOPE_API_KEY 已设置（生成评估需要 LLM）
  - data/evaluation/retrieval_dataset.json 存在（提供 query 列表）
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# 注入项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Config
from app.core.env import load_env
from app.core.logger import get_logger
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.retrieval_eval import recall_at_k_spans
from app.rag.service import get_service

logger = get_logger(__name__)

DATASET_PATH = "data/evaluation/retrieval_dataset.json"


def run_retrieval_eval(config, strategy, search_mode, use_rerank):
    """检索评估：计算各 query 的 Recall@K。

    复用已有逻辑：加载 dataset → 逐 query 检索 → span 级 Recall 计算
    """
    print("=" * 60)
    print("检索评估 (Retrieval Evaluation)")
    print("=" * 60)

    dataset = EvaluationDataset(DATASET_PATH).load()
    print("数据集: {} 条 query".format(len(dataset)))

    service = get_service(
        config=config,
        strategy=strategy,
        mode=search_mode,
        use_rerank=use_rerank,
    )

    top_k = config.retrieval_top_k
    results = []

    for i, item in enumerate(dataset):
        query = item["query"]
        relevant_spans = item.get("relevant_spans", [])

        print("\n[{}/{}] query: {}".format(i + 1, len(dataset), query[:60]))

        response = service.query(query)
        chunks = response.chunks

        # span 级 Recall@K
        for k in [1, 3, 5]:
            if k > top_k:
                break
            score = recall_at_k_spans(chunks, relevant_spans, k)
            print("  Recall@{}: {:.4f}".format(k, score))
            results.append({
                "query": query,
                "metric": "recall_at_k_spans",
                "k": k,
                "score": round(score, 4),
            })

    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for k in [1, 3, 5]:
        k_results = [r for r in results if r["k"] == k]
        if k_results:
            avg = sum(r["score"] for r in k_results) / len(k_results)
            print("  Avg Recall@{}: {:.4f}".format(k, avg))


def run_generation_eval(config, strategy, search_mode, use_rerank, output_path):
    """生成质量评估：Faithfulness + Relevance，输出 CSV 报告。

    流程：
      1. 加载 query 列表（复用 retrieval_dataset.json）
      2. 对每条 query 运行 RAG pipeline → 获取 context + answer
      3. 用 LLM-as-Judge 评估 faithfulness（answer vs context）和 relevance（answer vs query）
      4. 输出 CSV 报告
    """
    print("=" * 60)
    print("生成质量评估 (Generation Evaluation)")
    print("=" * 60)

    # 加载数据集
    dataset = EvaluationDataset(DATASET_PATH).load()
    print("数据集: {} 条 query".format(len(dataset)))

    # 初始化 RAG 服务（获取 context + answer）
    service = get_service(
        config=config,
        strategy=strategy,
        mode=search_mode,
        use_rerank=use_rerank,
    )

    # 初始化评估器（复用 generation backend 做 LLM-as-Judge）
    from app.generation.generator import create_generator
    from app.evaluation.generation_eval import (
        FaithfulnessEvaluator,
        RelevanceEvaluator,
    )

    judge_model = config.generation_model
    judge = create_generator(
        config.generation_backend,
        model=judge_model,
        api_key_env=config.generation_api_key_env,
        temperature=0.0,  # Judge 用 temperature=0 保证一致性
        max_tokens=2048,   # 评估需要更多 token 输出 JSON
    )

    faith_eval = FaithfulnessEvaluator(judge)
    rel_eval = RelevanceEvaluator(judge)

    # 评估每条 query
    csv_rows = []
    detailed_results = []

    for i, item in enumerate(dataset):
        query = item["query"]
        print("\n[{}/{}] query: {}".format(i + 1, len(dataset), query[:60]))

        # Step 1: RAG 生成
        t = time.time()
        response = service.query(query)
        rag_elapsed = time.time() - t
        context = response.context
        answer = response.answer or ""
        print("  RAG 生成完成: {:.1f}s, answer_len={}".format(rag_elapsed, len(answer)))

        if not answer:
            print("  [警告] 答案为空，跳过评估")
            csv_rows.append({
                "query": query,
                "answer": "",
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "relevance_reasoning": "答案为空",
                "rag_elapsed_s": round(rag_elapsed, 2),
            })
            continue

        # Step 2: Faithfulness 评估
        t = time.time()
        faith_result = faith_eval.evaluate(query, answer, context)
        faith_elapsed = time.time() - t
        print(
            "  Faithfulness: score={:.2f}, claims={}/{} supported ({:.1f}s)".format(
                faith_result["score"],
                faith_result["supported_claims"],
                faith_result["total_claims"],
                faith_elapsed,
            )
        )

        # Step 3: Relevance 评估
        t = time.time()
        rel_result = rel_eval.evaluate(query, answer)
        rel_elapsed = time.time() - t
        print(
            "  Relevance: score={:.2f} ({:.1f}s)".format(
                rel_result["score"], rel_elapsed,
            )
        )

        # 收集结果
        csv_rows.append({
            "query": query,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "faithfulness_score": faith_result["score"],
            "relevance_score": rel_result["score"],
            "total_claims": faith_result["total_claims"],
            "supported_claims": faith_result["supported_claims"],
            "relevance_reasoning": rel_result["reasoning"],
            "rag_elapsed_s": round(rag_elapsed, 2),
        })

        detailed_results.append({
            "query": query,
            "answer": answer,
            "context": context,
            "faithfulness": faith_result,
            "relevance": rel_result,
        })

    # 输出 CSV 报告
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query", "answer", "faithfulness_score", "relevance_score",
        "total_claims", "supported_claims", "relevance_reasoning", "rag_elapsed_s",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 60)
    print("CSV 报告已输出: {}".format(output_path))
    print("=" * 60)

    # 汇总统计
    if csv_rows:
        avg_faith = sum(r["faithfulness_score"] for r in csv_rows) / len(csv_rows)
        avg_rel = sum(r["relevance_score"] for r in csv_rows) / len(csv_rows)
        total_claims = sum(r["total_claims"] for r in csv_rows)
        total_supported = sum(r["supported_claims"] for r in csv_rows)

        print("\n汇总统计:")
        print("  Avg Faithfulness: {:.4f}".format(avg_faith))
        print("  Avg Relevance:    {:.4f}".format(avg_rel))
        print("  Total Claims:     {} (supported: {})".format(
            total_claims, total_supported
        ))

    # 输出详细 JSON（包含 claim 级证据）
    detail_path = output_path.with_suffix(".detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    print("  详细报告: {}".format(detail_path))


def main():
    parser = argparse.ArgumentParser(
        description="RAG 评估脚本：支持检索评估和生成质量评估"
    )
    parser.add_argument(
        "--mode", choices=["retrieval", "generation"],
        default="generation",
        help="评估模式: retrieval（检索评估）或 generation（生成质量评估）",
    )
    parser.add_argument(
        "--strategy", choices=["fixed", "recursive"],
        default="recursive",
        help="分块策略",
    )
    parser.add_argument(
        "--search-mode", choices=["vector", "bm25", "hybrid"],
        default="hybrid",
        help="检索模式",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="跳过 rerank",
    )
    parser.add_argument(
        "--output", default="reports/gen_eval.csv",
        help="CSV 输出路径（仅 generation 模式）",
    )
    args = parser.parse_args()

    load_env()
    config = Config()

    # .env 覆盖 model_name
    env_model = os.getenv("DASHSCOPE_MODEL")
    if env_model:
        config.data.setdefault("generation", {})["model_name"] = env_model
        logger.info("用 .env 的 DASHSCOPE_MODEL 覆盖 model_name: %s", env_model)

    print("配置: strategy={}, search_mode={}, rerank={}".format(
        args.strategy, args.search_mode, "off" if args.no_rerank else "on"
    ))
    print("LLM: {} (backend={})".format(
        config.generation_model, config.generation_backend
    ))
    print()

    if args.mode == "retrieval":
        run_retrieval_eval(
            config, args.strategy, args.search_mode,
            use_rerank=not args.no_rerank,
        )
    else:
        run_generation_eval(
            config, args.strategy, args.search_mode,
            use_rerank=not args.no_rerank,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
