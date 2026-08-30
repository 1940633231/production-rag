"""诊断：C-MTEB T2Retrieval 在候选池不同截断下的 Recall，定位 rerank 降分环节。

结论（2026-08 实测，20 query / 618 corpus，bge-small-zh + bge-reranker-base）:
  - 纯向量 top-10 与粗筛池 top-30/top-100 的 Recall 完全一致 → 相关文档都集中在
    向量检索前 10，扩大候选池对召回零贡献
  - rerank 精排后 Recall@10 反而下降（0.985 → 0.915）：短文本高重合场景下
    Cross-Encoder 打分引入噪声，是负优化 → 不要无脑叠加 rerank

用法:
  .venv\\Scripts\\python.exe scripts\\cmteb_rerank_diag.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from app.core.config import Config
from app.core.env import load_env
from app.evaluation.metrics import mrr, ndcg_at_k
from app.evaluation.retrieval_eval import recall_at_k

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_cmteb import load_data, build_index  # noqa: E402

KS = [1, 3, 5, 10]


def main():
    load_env()
    config = Config()

    corpus, queries, relevant_map = load_data(6000, 20, 5)

    from app.embedding.model import EmbeddingModel
    model = EmbeddingModel(config.embedding_model)
    index, emb = build_index(corpus["text"].tolist(), model)
    corpus_ids = corpus["id"].astype(str).tolist()

    query_texts = queries["text"].tolist()
    query_ids = queries["id"].astype(str).tolist()
    relevant_list = [relevant_map.get(qid, set()) for qid in query_ids]

    # 预取粗筛池（一次），供多路复用
    pool_size = 100
    pools = []
    qbatch = model.encode(query_texts)
    for qv in qbatch:
        qv = np.asarray(qv, dtype="float32").reshape(1, -1)
        _s, ids = index.search(qv, pool_size)
        pools.append([corpus_ids[j] for j in ids[0]])

    def agg(name, retrieved_list):
        sums = {k: 0.0 for k in KS}
        ms = ns = 0.0
        n = 0
        for i, qid in enumerate(query_ids):
            rel = relevant_list[i]
            if not rel:
                continue
            ret = retrieved_list[i]
            n += 1
            for k in KS:
                sums[k] += recall_at_k(ret, rel, k)
            ms += mrr(ret, rel)
            ns += ndcg_at_k(ret, rel, 10)
        print("== {} ==".format(name))
        for k in KS:
            print("  Recall@{}: {:.4f}".format(k, sums[k] / n))
        print("  MRR@10: {:.4f}  NDCG@10: {:.4f}".format(ms / n, ns / n))

    # 1. 纯向量 top-10
    agg("vector@10", [p[:10] for p in pools])

    # 2. 粗筛池自身（top-10/30/100 的召回能力）
    agg("pool@10", [p[:10] for p in pools])
    agg("pool@30", [p[:30] for p in pools])
    agg("pool@100", pools)

    # 3. rerank: pool@100 -> 精排 top-10
    from app.rerank.reranker import Reranker
    reranker = Reranker(config.rerank_model, batch_size=8)
    reranked = []
    for i, q in enumerate(query_texts):
        rows = corpus.iloc[[corpus_ids.index(cid) for cid in pools[i]]]
        candidates = [
            {"chunk_id": cid, "content": text}
            for cid, text in zip(rows["id"], rows["text"])
        ]
        ranked = reranker.rerank(q, candidates, top_k=10)
        reranked.append([c["chunk_id"] for c in ranked])
    agg("rerank@10 (pool100)", reranked)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("\n总耗时: {:.1f}s".format(time.time() - t0))
