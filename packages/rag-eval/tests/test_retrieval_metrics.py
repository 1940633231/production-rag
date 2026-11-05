from __future__ import annotations

import random

from rag_eval.contracts import EvalSample, Prediction
from rag_eval.metrics.retrieval import (
    evaluate_retrieval,
    evidence_hit_at_k,
    span_ndcg_at_k,
)


def _sample() -> EvalSample:
    return EvalSample.model_validate(
        {
            "id": "q1",
            "task": "retrieval",
            "query": "query",
            "gold": {
                "documents": [
                    {
                        "document_id": "doc-a",
                        "relevance": 1,
                        "spans": [{"start": 10, "end": 20}],
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk-a",
                        "document_id": "doc-a",
                        "relevance": 1,
                        "spans": [{"start": 10, "end": 20}],
                    }
                ],
            },
        }
    )


def test_perfect_retrieval_is_bounded() -> None:
    sample = _sample()
    prediction = Prediction.model_validate(
        {
            "sample_id": "q1",
            "documents": [{"document_id": "doc-a", "rank": 1}],
            "chunks": [
                {
                    "chunk_id": "chunk-a",
                    "document_id": "doc-a",
                    "rank": 1,
                    "span": {"start": 10, "end": 20},
                }
            ],
        }
    )
    summary = evaluate_retrieval([sample], [prediction], [1, 5])

    assert summary.values["recall@1"] == 1.0
    assert summary.values["precision@1"] == 1.0
    assert summary.values["mrr@1"] == 1.0
    assert summary.values["ndcg@1"] == 1.0
    assert summary.values["evidence_hit@1"] == 1.0
    assert all(0.0 <= value <= 1.0 for value in summary.values.values())


def test_duplicate_chunk_cannot_push_span_ndcg_above_one() -> None:
    sample = _sample()
    duplicate = {
        "chunk_id": "chunk-a",
        "document_id": "doc-a",
        "span": {"start": 10, "end": 20},
    }
    prediction = Prediction(sample_id="q1", chunks=[duplicate for _ in range(30)])

    score = span_ndcg_at_k(sample, prediction, 30)

    assert 0.0 <= score <= 1.0
    assert score == 1.0


def test_same_offsets_in_different_documents_do_not_match() -> None:
    sample = _sample()
    prediction = Prediction.model_validate(
        {
            "sample_id": "q1",
            "chunks": [
                {
                    "chunk_id": "other-chunk",
                    "document_id": "doc-b",
                    "span": {"start": 10, "end": 20},
                }
            ],
        }
    )

    assert span_ndcg_at_k(sample, prediction, 10) == 0.0
    assert evidence_hit_at_k(sample, prediction, 10) == 0.0


def test_empty_or_missing_results_return_zero_defaults() -> None:
    sample = _sample()
    empty = Prediction(sample_id="q1")
    summary = evaluate_retrieval([sample], [empty], [5])

    assert summary.values["recall@5"] == 0.0
    assert summary.values["span_ndcg@5"] == 0.0
    assert summary.values["evidence_hit@5"] == 0.0

    missing_summary = evaluate_retrieval([sample], [], [5])
    assert missing_summary.values["recall@5"] == 0.0
    assert missing_summary.values["span_ndcg@5"] == 0.0


def test_external_qrels_can_override_sample_document_gold() -> None:
    sample = _sample()
    prediction = Prediction.model_validate(
        {
            "sample_id": "q1",
            "documents": [{"document_id": "doc-b", "rank": 1}],
        }
    )

    summary = evaluate_retrieval(
        [sample],
        [prediction],
        [1],
        qrels={"q1": {"doc-b": 1}},
    )

    assert summary.values["recall@1"] == 1.0
    assert summary.values["precision@1"] == 1.0


def test_randomized_span_ndcg_stays_in_range() -> None:
    sample = _sample()
    randomizer = random.Random(17)
    for _ in range(200):
        chunks = []
        for index in range(randomizer.randint(0, 25)):
            start = randomizer.randint(0, 30)
            chunks.append(
                {
                    "chunk_id": randomizer.choice(["chunk-a", f"c-{index}"]),
                    "document_id": randomizer.choice(["doc-a", "doc-b"]),
                    "span": {"start": start, "end": start + randomizer.randint(1, 10)},
                }
            )
        prediction = Prediction.model_validate(
            {"sample_id": "q1", "chunks": chunks}
        )
        score = span_ndcg_at_k(sample, prediction, 10)
        assert 0.0 <= score <= 1.0
