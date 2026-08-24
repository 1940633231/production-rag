import argparse
from pathlib import Path

from app.core.config import Config

from app.embedding.model import EmbeddingModel
from app.vector import create_vector_store
from app.storage.metadata_store import MetadataStore
from app.storage.base import MetadataChunkRepository
from app.rag.retriever import Retriever

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.retrieval_eval import recall_at_k_spans


def main():

    # =========================
    # 1. Arguments
    # =========================

    parser = argparse.ArgumentParser()

    parser.add_argument("--strategy", choices=["fixed", "recursive"], required=True)

    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="vector")

    parser.add_argument("--rerank", action="store_true", help="启用 Cross-Encoder 重排")

    args = parser.parse_args()

    # =========================
    # 2. Config
    # =========================

    config = Config()

    # =========================
    # 3. Index Path
    # =========================

    index_dir = Path("data/index") / args.strategy

    index_path = index_dir / "faiss.index"

    metadata_path = index_dir / "metadata.json"

    if not index_path.exists():

        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not metadata_path.exists():

        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # =========================
    # 4. Metadata（vector / bm25 都需要）
    # =========================

    metadata = MetadataStore().load(str(metadata_path))
    chunk_repo = MetadataChunkRepository(metadata)

    # =========================
    # 5. Retriever（按 mode 切换）
    # =========================

    if args.mode in ("vector", "hybrid"):

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
    # 6. Reranker（可选）
    # =========================

    if args.rerank:

        from app.rerank.reranker import Reranker

        reranker = Reranker(config.rerank_model)

    else:

        reranker = None

    # =========================
    # 8. Dataset (span 标注，与切片策略解耦)
    # =========================

    dataset = EvaluationDataset("data/evaluation/retrieval_dataset.json")

    samples = dataset.load()

    # =========================
    # 9. Evaluation (span 级 Recall)
    # =========================

    ks = [1, 3, 5, 10]

    scores = {k: [] for k in ks}

    for sample in samples:

        query = sample["query"]

        relevant_spans = sample["relevant_spans"]

        if args.rerank:
            # 宽候选池 retrieve → cross-encoder 重排截断到 max(ks)
            candidates = retriever.search(query, top_k=config.rerank_candidate_pool)
            results = reranker.rerank(query, candidates, top_k=max(ks))
        else:
            results = retriever.search(query, top_k=max(ks))

        for k in ks:

            score = recall_at_k_spans(results, relevant_spans, k)

            scores[k].append(score)

    # =========================
    # 10. Result
    # =========================

    print()

    print("=" * 50)

    print("Retrieval Evaluation (span-based)")

    print("=" * 50)

    print(f"Strategy: {args.strategy}")

    print(f"Mode: {args.mode}")

    print(f"Rerank: {args.rerank}")

    print(f"Query Count: {len(samples)}")

    print()

    for k in ks:

        values = scores[k]

        avg = sum(values) / len(values) if values else 0.0

        print(f"Recall@{k:<2}: {avg:.4f}")


if __name__ == "__main__":
    main()
