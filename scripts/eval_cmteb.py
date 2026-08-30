"""C-MTEB T2Retrieval 中文检索评测脚本。

数据（scripts/download_cmteb.py 下载到 data/evaluation/cmteb/）:
  - C-MTEB/T2Retrieval:      corpus（passage 语料，id/text）+ queries（id/text）
  - C-MTEB/T2Retrieval-qrels: qrels（qid/pid/score，dev 集）

评测方式:
  1. 用项目 embedding 模型编码 corpus 建 FAISS 索引（IndexFlatIP，向量已归一化）
  2. 对每条 query 编码检索 top-k passage
  3. 与 qrels 比对计算 Recall@1/3/5/10、MRR@10、NDCG@10
  4. 可选 --rerank 先用 embedding 粗筛再 Cross-Encoder 精排
  5. 可选 --save-baseline / --compare 接入回归门禁闭环

用法:
  .venv\\Scripts\\python.exe scripts\\eval_cmteb.py                          # 默认 500 query
  .venv\\Scripts\\python.exe scripts\\eval_cmteb.py --max-queries 2000
  .venv\\Scripts\\python.exe scripts\\eval_cmteb.py --max-corpus 50000 --rerank
  .venv\\Scripts\\python.exe scripts\\eval_cmteb.py --save-baseline reports/cmteb_baseline.json
  .venv\\Scripts\\python.exe scripts\\eval_cmteb.py --compare reports/cmteb_baseline.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from app.core.config import Config
from app.core.env import load_env
from app.core.logger import get_logger

logger = get_logger(__name__)

DATA_ROOT = PROJECT_ROOT / "data/evaluation/cmteb"
CORPUS_PATH = DATA_ROOT / "C-MTEB/T2Retrieval/data/corpus-00000-of-00001-8afe7b7a7eca49e3.parquet"
QUERIES_PATH = DATA_ROOT / "C-MTEB/T2Retrieval/data/queries-00000-of-00001-930bf3b805a80dd9.parquet"
QRELS_PATH = DATA_ROOT / "C-MTEB/T2Retrieval-qrels/data/dev-00000-of-00001-92ed0416056ff7e1.parquet"

CACHE_DEFAULT = PROJECT_ROOT / "data/evaluation/cmteb_cache"

KS = [1, 3, 5, 10]


def hash_corpus(corpus_df) -> str:
    """对采样后的 corpus 内容做 hash（缓存键：同一份数据 + 同一 seed/规模 → 同 hash → 可复用 embedding）。

    GitHub-hosted runner 每次都是全新机器，corpus embedding 是确定性产物
    （固定数据 + 固定模型），内容不变时应直接复用，避免每次全量重算。
    """
    payload = "\n".join(
        "{}|{}".format(i, t)
        for i, t in zip(corpus_df["id"].astype(str), corpus_df["text"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    """corpus embedding 缓存：meta.json + corpus_ids.json + embeddings.npy。

    命中条件（全部一致才复用）：
      - embedding 模型名 + normalize 开关
      - 采样 corpus 内容 hash（corpus_hash）
    """

    def __init__(self, path):
        self.path = Path(path)
        self.meta_path = self.path / "meta.json"
        self.ids_path = self.path / "corpus_ids.json"
        self.npy_path = self.path / "embeddings.npy"

    def _meta(self, model_name, normalize, corpus_hash):
        return {
            "model": model_name,
            "normalize": bool(normalize),
            "corpus_hash": corpus_hash,
        }

    def load(self, model_name, normalize, corpus_hash, dim):
        """命中返回 np.ndarray，未命中返回 None。"""
        if not all(p.exists() for p in (self.meta_path, self.ids_path, self.npy_path)):
            return None
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if meta != self._meta(model_name, normalize, corpus_hash):
            logger.info("embedding 缓存未命中（meta 不一致），重新编码")
            return None
        emb = np.load(self.npy_path)
        if emb.shape[1] != dim:
            logger.info("embedding 缓存未命中（维度 %d != %d），重新编码", emb.shape[1], dim)
            return None
        logger.info("embedding 缓存命中: %s (%s)", self.npy_path, emb.shape)
        return emb

    def save(self, model_name, normalize, corpus_hash, embeddings, corpus_ids):
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.npy_path, embeddings)
        self.ids_path.write_text(
            json.dumps(corpus_ids, ensure_ascii=False), encoding="utf-8"
        )
        self.meta_path.write_text(
            json.dumps(self._meta(model_name, normalize, corpus_hash), ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("embedding 缓存已保存: %s (%s)", self.npy_path, embeddings.shape)


def load_data(max_corpus, max_queries, fill_factor=5, seed=42):
    """加载 corpus/queries/qrels，构造"query-focused"受限语料。

    策略（解决子集采样覆盖度低 + 相关占比虚高的问题）:
      1. 随机采样 N 条 query（固定 seed）
      2. 从 qrels 反查这些 query 的相关 pid 文档
      3. 补充干扰文档 = max_corpus 与相关文档 * fill_factor 的较小者，
         使相关文档在语料中占少数，逼近真实检索场景（否则指标虚高）
      这样每条 query 的相关标注完整落在语料内，且语料规模可控（CPU 可编码）。

    返回 (corpus_df, queries_df, relevant_map)。
    """
    import pandas as pd

    corpus = pd.read_parquet(CORPUS_PATH)
    queries = pd.read_parquet(QUERIES_PATH)
    qrels = pd.read_parquet(QRELS_PATH)

    # 1. 采样 query
    queries = queries.sample(n=min(max_queries, len(queries)), random_state=seed)
    query_ids = set(queries["id"].astype(str))

    # 2. 从 qrels 反查这些 query 的相关 pid
    pid_to_qids = {}  # pid -> set(qid)
    for row in qrels.itertuples():
        qid, pid = str(row.qid), str(row.pid)
        if qid in query_ids:
            pid_to_qids.setdefault(pid, set()).add(qid)

    relevant_pids = set(pid_to_qids.keys())
    print("采样 query: {} 条，相关 pid（去重）: {} 条".format(
        len(query_ids), len(relevant_pids),
    ))

    # 3. 构造语料 = 相关 pid 文档 + 补充干扰项（默认 5 倍）
    relevant_df = corpus[corpus["id"].astype(str).isin(relevant_pids)]
    rest_df = corpus[~corpus["id"].astype(str).isin(relevant_pids)]
    fill_target = min(
        len(relevant_df) * fill_factor,          # 相关文档的 fill_factor 倍
        max(0, max_corpus - len(relevant_df)),   # 不超过 max_corpus 上限
        len(rest_df),
    )
    fill_df = (
        rest_df.sample(n=fill_target, random_state=seed) if fill_target else rest_df.head(0)
    )
    corpus_sample = pd.concat([relevant_df, fill_df])
    print("corpus: 相关 {} 条 + 干扰 {} 条 = {} 条（相关占比 {:.1f}%）".format(
        len(relevant_df), len(fill_df), len(corpus_sample),
        100.0 * len(relevant_df) / len(corpus_sample),
    ))

    # 4. relevant_map：只保留语料内的相关 pid（理论上全部都在）
    corpus_ids = set(corpus_sample["id"].astype(str))
    relevant_map = {}
    dropped = 0
    for pid, qids in pid_to_qids.items():
        if pid not in corpus_ids:
            dropped += 1
            continue
        for qid in qids:
            relevant_map.setdefault(qid, set()).add(pid)
    print("相关 pid 未落入语料的条数: {}（理论应为 0）".format(dropped))
    print("有完整标注的 query: {}/{}".format(len(relevant_map), len(queries)))
    return corpus_sample, queries, relevant_map


def build_index(embeddings):
    """由 corpus embeddings 构建 FAISS IndexFlatIP 索引（向量已归一化）。"""
    import faiss

    t = time.time()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    logger.info("corpus 索引构建完成: %.3fs, %d 条, dim=%d",
                time.time() - t, len(embeddings), embeddings.shape[1])
    return index


def retrieve(query_texts, index, embeddings, model, top_k, corpus_df, corpus_ids,
             reranker=None, candidate_k=100):
    """逐条 query 检索，返回每个 query 的 corpus id 列表。

    corpus_df: 采样后的 corpus DataFrame（id/text 列），供 rerank 取文本。
    candidate_k: rerank 前的粗筛候选池大小（越小重排越快）。
    """
    results = []
    batch = model.encode(list(query_texts))
    for i, qv in enumerate(batch):
        qv = np.asarray(qv, dtype="float32").reshape(1, -1)
        if reranker is None:
            scores, ids = index.search(qv, top_k)
            results.append([corpus_ids[j] for j in ids[0]])
        else:
            # 粗筛候选池 → rerank 精排（rerank 需要候选含 content 文本）
            candidate_k = min(candidate_k, len(corpus_ids))
            _s, ids = index.search(qv, candidate_k)
            candidate_rows = corpus_df.iloc[ids[0]]
            candidates = [
                {"chunk_id": cid, "content": text}
                for cid, text in zip(candidate_rows["id"], candidate_rows["text"])
            ]
            ranked = reranker.rerank(query_texts[i], candidates, top_k=top_k)
            results.append([c["chunk_id"] for c in ranked])
    return results


def evaluate(retrieved_list, relevant_list, query_ids):
    """计算 Recall@K / MRR@10 / NDCG@10。"""
    from app.evaluation.metrics import mrr, ndcg_at_k
    from app.evaluation.retrieval_eval import recall_at_k

    sums = {k: 0.0 for k in KS}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    evaluated = 0

    for i, qid in enumerate(query_ids):
        relevant = relevant_list[i]
        if not relevant:
            continue
        retrieved = retrieved_list[i]
        evaluated += 1
        for k in KS:
            sums[k] += recall_at_k(retrieved, relevant, k)
        mrr_sum += mrr(retrieved, relevant)
        ndcg_sum += ndcg_at_k(retrieved, relevant, 10)

    if evaluated == 0:
        print("[WARN] 没有任何 query 有可用标注（受限 corpus 可能过小）")
        return {}

    metrics = {}
    for k in KS:
        metrics["recall@{}".format(k)] = round(sums[k] / evaluated, 4)
    metrics["mrr@10"] = round(mrr_sum / evaluated, 4)
    metrics["ndcg@10"] = round(ndcg_sum / evaluated, 4)
    return metrics


def run():
    parser = argparse.ArgumentParser(description="C-MTEB T2Retrieval 检索评测")
    parser.add_argument("--max-corpus", type=int, default=20000,
                        help="语料总规模上限（相关文档 + 干扰项，默认 20000）")
    parser.add_argument("--fill-factor", type=int, default=5,
                        help="干扰项为相关文档的倍数（默认 5，越大越接近真实检索场景但编码越慢）")
    parser.add_argument("--max-queries", type=int, default=500,
                        help="限制评测 query 条数（默认 500，全量 22812 条）")
    parser.add_argument("--seed", type=int, default=42,
                        help="采样随机种子（默认 42，固定 seed 保证 CI 可复现）")
    parser.add_argument("--top-k", type=int, default=10, help="检索返回条数（默认 10）")
    parser.add_argument("--rerank", action="store_true", help="启用 Cross-Encoder 重排")
    parser.add_argument("--candidate-k", type=int, default=30,
                        help="rerank 前粗筛候选池大小（默认 30，越小重排越快）")
    parser.add_argument("--rerank-batch", type=int, default=8,
                        help="rerank CrossEncoder 推理 batch（默认 8）")
    parser.add_argument("--rerank-device", default=None,
                        help="rerank 设备：None 自动 / cpu / cuda（默认 None）")
    parser.add_argument("--save-baseline", metavar="PATH", help="保存评估结果为新基线")
    parser.add_argument("--compare", metavar="PATH", help="与基线对比并执行门禁")
    parser.add_argument("--trend", default="reports/cmteb_trend.csv", help="趋势 CSV")
    parser.add_argument("--tolerance", type=float, default=None, help="统一指标阈值")
    parser.add_argument("--embedding-cache", metavar="PATH", default=str(CACHE_DEFAULT),
                        help="corpus embedding 缓存目录（命中则跳过编码，默认 data/evaluation/cmteb_cache）")
    parser.add_argument("--no-embedding-cache", action="store_true",
                        help="禁用 embedding 缓存（总是重新编码）")
    args = parser.parse_args()

    load_env()
    config = Config()
    print("配置: corpus_limit={}, fill_factor={}, queries_limit={}, top_k={}, rerank={}, embedding_cache={}".format(
        args.max_corpus, args.fill_factor, args.max_queries, args.top_k, args.rerank,
        "off" if args.no_embedding_cache else args.embedding_cache,
    ))
    print()

    # ---- 1. 数据 ----
    corpus, queries, relevant_map = load_data(
        args.max_corpus, args.max_queries, args.fill_factor, seed=args.seed,
    )
    corpus_ids = corpus["id"].astype(str).tolist()
    corpus_hash = hash_corpus(corpus)
    logger.info("采样 corpus hash: %s（缓存键）", corpus_hash)

    # ---- 2. Embedding（缓存命中则跳过编码）+ 索引 ----
    from app.embedding.model import EmbeddingModel
    t = time.time()
    model = EmbeddingModel(config.embedding_model)
    logger.info("embedding 模型加载完成: %.3fs", time.time() - t)

    normalize = config.data.get("embedding", {}).get("normalize", True)
    cache = None
    if not args.no_embedding_cache:
        cache = EmbeddingCache(args.embedding_cache)
        embeddings = cache.load(config.embedding_model, normalize, corpus_hash, model.dimension)
    else:
        embeddings = None

    if embeddings is None:
        t = time.time()
        embeddings = model.encode(list(corpus["text"]))
        logger.info("corpus 编码完成: %.3fs, %d 条, dim=%d",
                    time.time() - t, len(corpus), embeddings.shape[1])
        if cache is not None:
            cache.save(config.embedding_model, normalize, corpus_hash, embeddings, corpus_ids)

    index = build_index(embeddings)

    # ---- 3. 检索 ----
    query_texts = queries["text"].tolist()
    query_ids = queries["id"].astype(str).tolist()
    relevant_list = [relevant_map.get(qid, set()) for qid in query_ids]

    reranker = None
    if args.rerank:
        from app.rerank.reranker import Reranker
        t = time.time()
        reranker = Reranker(
            config.rerank_model,
            device=args.rerank_device,
            batch_size=args.rerank_batch,
        )
        logger.info("reranker 加载完成: %.3fs", time.time() - t)

    t = time.time()
    retrieved_list = retrieve(
        query_texts, index, embeddings, model, args.top_k, corpus, corpus_ids,
        reranker, args.candidate_k,
    )
    print("检索完成: {} 条 query, 耗时 {:.1f}s".format(
        len(query_texts), time.time() - t,
    ))

    # ---- 4. 指标 ----
    metrics = evaluate(retrieved_list, relevant_list, query_ids)
    if not metrics:
        sys.exit(1)
    print("\n========== C-MTEB T2Retrieval 评测结果 ==========")
    for k in KS:
        print("  Recall@{}: {:.4f}".format(k, metrics["recall@{}".format(k)]))
    print("  MRR@10:   {:.4f}".format(metrics["mrr@10"]))
    print("  NDCG@10:  {:.4f}".format(metrics["ndcg@10"]))
    print("=================================================")

    # ---- 5. 回归闭环（可选） ----
    from app.evaluation.regression import (
        all_passed, append_trend, compare_metrics, format_table,
        load_baseline, save_baseline,
    )
    config_info = {
        "dataset": "C-MTEB/T2Retrieval (dev)",
        "corpus_limit": args.max_corpus,
        "queries_limit": args.max_queries,
        "top_k": args.top_k,
        "rerank": args.rerank,
        "embedding": config.embedding_model,
    }
    thresholds = None
    if args.tolerance is not None:
        thresholds = {m: args.tolerance for m in metrics}

    if args.save_baseline:
        p = save_baseline(args.save_baseline, metrics, config_info)
        print("\n基线已保存: {}".format(p))

    if args.compare:
        old = load_baseline(args.compare)
        print("\n对比基线: {} (commit={})".format(args.compare, old.get("commit")))
        rows = compare_metrics(metrics, old["metrics"], thresholds=thresholds)
        print(format_table(rows))
        append_trend(args.trend, metrics)
        ok = all_passed(rows)
        print("\n门禁结果: {}".format("PASS" if ok else "FAIL"))
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    run()
