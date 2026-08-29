"""批量评估脚本：支持检索评估 + 生成质量评估 + 评估回归闭环。

用法示例:

  # 检索评估（Recall@K）
  python scripts/evaluate.py --mode retrieval

  # 生成质量评估（Faithfulness + Relevance），输出 CSV 报告
  python scripts/evaluate.py --mode generation

  # 评估回归闭环：存档基线
  python scripts/evaluate.py --save-baseline reports/baseline.json

  # 评估回归闭环：与基线对比 + 门禁 + 趋势（指标跌破阈值退出码 1）
  python scripts/evaluate.py --compare reports/baseline.json

  # 检索评估回归（强制 stub 生成器，无需 LLM/API key，适合 CI）
  python scripts/evaluate.py --no-generation --compare reports/baseline.json

  # 指定策略和检索模式
  python scripts/evaluate.py --mode generation --strategy recursive --search-mode hybrid

前置条件:
  - 索引文件已生成：data/index/{strategy}/（或 Milvus collection 已写入）
  - 环境变量 DASHSCOPE_API_KEY 已设置（生成评估需要 LLM；--no-generation 除外）
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
from app.evaluation.regression import (
    all_passed,
    append_trend,
    compare_metrics,
    format_table,
    load_baseline,
    save_baseline,
)
from app.evaluation.retrieval_eval import recall_at_k_spans
from app.rag.service import get_service, reset_service_cache

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
    metrics = {}
    for k in [1, 3, 5]:
        k_results = [r for r in results if r["k"] == k]
        if k_results:
            avg = sum(r["score"] for r in k_results) / len(k_results)
            print("  Avg Recall@{}: {:.4f}".format(k, avg))
            metrics["recall@{}".format(k)] = round(avg, 4)
    return metrics


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

    # 返回结构化指标（供回归对比）
    metrics = {}
    if csv_rows:
        metrics = {
            "faithfulness": round(avg_faith, 4),
            "relevance": round(avg_rel, 4),
        }
    return metrics


def _config_snapshot(config, strategy, search_mode, use_rerank):
    """记录评估配置快照（写入基线供溯源）。"""
    return {
        "strategy": strategy,
        "mode": search_mode,
        "rerank": use_rerank,
        "model": config.generation_model,
        "top_k": config.retrieval_top_k,
        "dataset": DATASET_PATH,
    }


def _force_stub_generator(config):
    """强制使用 stub 生成器（检索评估无需 LLM/API key，适合 CI）。"""
    config.data.setdefault("generation", {})["backend"] = "stub"
    reset_service_cache()


def _resolve_thresholds(tolerance):
    """阈值解析：--tolerance 统一覆盖所有指标时使用，否则用各指标默认阈值。"""
    from app.evaluation.regression import DEFAULT_THRESHOLDS
    if tolerance is None:
        return None
    return {m: tolerance for m in DEFAULT_THRESHOLDS}


def run_regression(config, args):
    """评估闭环：跑完整评估 → 保存基线 或 对比基线 + 门禁 + 趋势。

    顺序：
      1. 先加载旧基线（若 --compare）
      2. 跑检索评估（必须）+ 生成评估（默认，--no-generation 跳过）
      3. --save-baseline: 存档本次结果为新基线
      4. --compare: 与旧基线对比，输出对比表 + 追加趋势，门禁失败退出码 1
    """
    strategy, search_mode, use_rerank = (
        args.strategy, args.search_mode, not args.no_rerank,
    )

    # --no-generation 时强制 stub 生成器：检索评估无需 LLM（CI 无 API key 场景）
    if args.no_generation:
        _force_stub_generator(config)

    old_baseline = None
    if args.compare:
        try:
            old_baseline = load_baseline(args.compare)
            print("已加载基线: {} (commit={}, 创建于 {})".format(
                args.compare, old_baseline.get("commit"), old_baseline.get("created_at"),
            ))
        except Exception as e:
            logger.error("加载基线失败: %s", e)
            sys.exit(2)

    metrics = run_retrieval_eval(config, strategy, search_mode, use_rerank)

    if not args.no_generation:
        try:
            gen_metrics = run_generation_eval(
                config, strategy, search_mode, use_rerank, args.output
            )
            metrics.update(gen_metrics)
        except Exception as e:
            logger.error("生成评估失败，跳过（仅以检索指标对比）: %s", e, exc_info=True)

    if args.save_baseline:
        p = save_baseline(args.save_baseline, metrics, _config_snapshot(
            config, strategy, search_mode, use_rerank
        ))
        print("\n基线已保存: {}".format(p))

    if args.compare:
        if not metrics:
            logger.error("本次评估无任何指标，无法对比")
            sys.exit(2)
        rows = compare_metrics(
            metrics, old_baseline["metrics"],
            thresholds=_resolve_thresholds(args.tolerance),
        )
        print("\n" + "=" * 60)
        print("对比基线: {}（commit={}）".format(args.compare, old_baseline.get("commit")))
        print(format_table(rows))
        append_trend(args.trend, metrics)
        ok = all_passed(rows)
        print("\n门禁结果: {}".format(
            "PASS" if ok else "FAIL（存在指标跌破阈值，不允许合入）"
        ))
        if not ok:
            sys.exit(1)


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
    parser.add_argument(
        "--save-baseline", metavar="PATH",
        help="回归模式：将本次评估指标存档为基线 JSON（如 reports/baseline.json）",
    )
    parser.add_argument(
        "--compare", metavar="PATH",
        help="回归模式：与基线 JSON 对比并执行门禁（任一指标跌破阈值则退出码 1）",
    )
    parser.add_argument(
        "--trend", default="reports/trend.csv",
        help="趋势 CSV 路径（--compare 时追加一行，默认 reports/trend.csv）",
    )
    parser.add_argument(
        "--no-generation", action="store_true",
        help="跳过生成质量评估（检索评估强制使用 stub 生成器，无需 LLM/API key）",
    )
    parser.add_argument(
        "--tolerance", type=float, default=None,
        help="统一覆盖所有指标阈值（默认按指标独立阈值，见 app/evaluation/regression.py）",
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

    if args.save_baseline or args.compare:
        run_regression(config, args)
        return

    # --no-generation 在普通检索模式下同样强制 stub（无需 LLM/API key）
    if args.no_generation and args.mode == "retrieval":
        _force_stub_generator(config)

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
