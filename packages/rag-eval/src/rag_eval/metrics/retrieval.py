"""Retrieval and span-evidence metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable

import ir_measures

from rag_eval.contracts import EvalSample, MetricSummary, Prediction, Span
from rag_eval.metrics import safe_mean


def _prediction_map(predictions: Iterable[Prediction]) -> dict[str, Prediction]:
    result: dict[str, Prediction] = {}
    for prediction in predictions:
        if prediction.sample_id in result:
            raise ValueError(f"duplicate prediction sample_id: {prediction.sample_id}")
        result[prediction.sample_id] = prediction
    return result


def _ordered_documents(prediction: Prediction) -> list[str]:
    documents = sorted(
        prediction.documents,
        key=lambda item: item.rank if item.rank is not None else 10**9,
    )
    ordered = [document.document_id for document in documents]
    if ordered:
        return ordered
    seen: set[str] = set()
    for chunk in _ordered_chunks(prediction):
        if chunk.document_id not in seen:
            seen.add(chunk.document_id)
            ordered.append(chunk.document_id)
    return ordered


def _ordered_chunks(prediction: Prediction):
    return sorted(
        prediction.chunks,
        key=lambda item: item.rank if item.rank is not None else 10**9,
    )


def _gold_document_relevance(sample: EvalSample) -> dict[str, float]:
    result: dict[str, float] = {}
    for document in sample.gold.documents:
        if document.relevance > 0:
            result[document.document_id] = max(
                result.get(document.document_id, 0.0), document.relevance
            )
    for chunk in sample.gold.chunks:
        if chunk.relevance > 0:
            result[chunk.document_id] = max(
                result.get(chunk.document_id, 0.0), chunk.relevance
            )
    return result


def _gold_spans(sample: EvalSample) -> list[tuple[str, Span, float]]:
    spans: dict[tuple[str, int, int], float] = {}
    for document in sample.gold.documents:
        for span in document.spans:
            key = (document.document_id, span.start, span.end)
            spans[key] = max(spans.get(key, 0.0), document.relevance)
    for chunk in sample.gold.chunks:
        for span in chunk.spans:
            key = (chunk.document_id, span.start, span.end)
            spans[key] = max(spans.get(key, 0.0), chunk.relevance)
    return [
        (document_id, Span(start=start, end=end), relevance)
        for (document_id, start, end), relevance in spans.items()
        if relevance > 0
    ]


def _span_match(gold_document: str, gold_span: Span, chunk_document: str, chunk_span: Span) -> bool:
    return gold_document == chunk_document and gold_span.overlaps(chunk_span)


def span_ndcg_at_k(sample: EvalSample, prediction: Prediction | None, k: int) -> float:
    """NDCG over unique gold spans, with duplicate chunks counted once."""

    if prediction is None or prediction.status != "ok" or k <= 0:
        return 0.0
    gold_spans = _gold_spans(sample)
    if not gold_spans:
        gold_documents = _gold_document_relevance(sample)
        if not gold_documents:
            return 0.0
        ideal = sorted(gold_documents.values(), reverse=True)
        seen_documents: set[str] = set()
        dcg = 0.0
        rank = 0
        for document_id in _ordered_documents(prediction):
            if document_id in seen_documents:
                continue
            seen_documents.add(document_id)
            rank += 1
            if rank > k:
                break
            if document_id in gold_documents:
                relevance = gold_documents[document_id]
                dcg += (2**relevance - 1) / math.log2(rank + 1)
        idcg = sum(
            (2**relevance - 1) / math.log2(index + 1)
            for index, relevance in enumerate(ideal[:k], start=1)
        )
        return max(0.0, min(1.0, dcg / idcg)) if idcg else 0.0

    ideal = sorted((relevance for _, _, relevance in gold_spans), reverse=True)
    matched: set[tuple[str, int, int]] = set()
    seen_chunks: set[tuple[str, str]] = set()
    dcg = 0.0
    rank = 0
    for chunk in _ordered_chunks(prediction):
        chunk_key = (chunk.document_id, chunk.chunk_id)
        if chunk_key in seen_chunks or chunk.span is None:
            continue
        seen_chunks.add(chunk_key)
        rank += 1
        if rank > k:
            break
        candidates = [
            (document_id, span, relevance)
            for document_id, span, relevance in gold_spans
            if (document_id, span.start, span.end) not in matched
            and _span_match(document_id, span, chunk.document_id, chunk.span)
        ]
        if candidates:
            document_id, span, relevance = max(candidates, key=lambda item: item[2])
            matched.add((document_id, span.start, span.end))
            dcg += (2**relevance - 1) / math.log2(rank + 1)
    idcg = sum(
        (2**relevance - 1) / math.log2(index + 1)
        for index, relevance in enumerate(ideal[:k], start=1)
    )
    return max(0.0, min(1.0, dcg / idcg)) if idcg else 0.0


def evidence_hit_at_k(sample: EvalSample, prediction: Prediction | None, k: int) -> float:
    if prediction is None or prediction.status != "ok" or k <= 0:
        return 0.0
    gold_spans = _gold_spans(sample)
    chunks = _ordered_chunks(prediction)[:k]
    if gold_spans:
        for chunk in chunks:
            if chunk.span is None:
                continue
            if any(
                _span_match(document_id, span, chunk.document_id, chunk.span)
                for document_id, span, _ in gold_spans
            ):
                return 1.0
        return 0.0
    gold_documents = set(_gold_document_relevance(sample))
    return 1.0 if any(chunk.document_id in gold_documents for chunk in chunks) else 0.0


def all_docs_hit_at_k(sample: EvalSample, prediction: Prediction | None, k: int) -> float:
    relevant = set(_gold_document_relevance(sample))
    if not relevant:
        return 0.0
    if prediction is None or prediction.status != "ok":
        return 0.0
    retrieved = set(_ordered_documents(prediction)[:k])
    return 1.0 if relevant.issubset(retrieved) else 0.0


def _measure(factory, k: int):
    return factory @ k


def _backend_relevance(relevance: float) -> int:
    if relevance <= 0:
        return 0
    return max(1, int(round(relevance)))


def evaluate_retrieval(
    samples: list[EvalSample],
    predictions: Iterable[Prediction],
    k_values: tuple[int, ...] | list[int] = (5, 10),
    qrels: dict[str, dict[str, float]] | None = None,
) -> MetricSummary:
    if not samples:
        return MetricSummary()
    prediction_by_id = _prediction_map(predictions)
    values: dict[str, float] = {}
    denominators: dict[str, int] = {}
    counts: dict[str, int] = {"expected_samples": len(samples)}

    document_qrels: dict[str, dict[str, float]] = {}
    document_run: dict[str, dict[str, float]] = {}
    chunk_qrels: dict[str, dict[str, float]] = {}
    chunk_run: dict[str, dict[str, float]] = {}
    has_chunk_qrels = False
    for sample in samples:
        sample_qrels = (
            qrels.get(sample.id, {})
            if qrels is not None
            else _gold_document_relevance(sample)
        )
        document_qrels[sample.id] = {
            document_id: _backend_relevance(relevance)
            for document_id, relevance in sample_qrels.items()
        }
        prediction = prediction_by_id.get(sample.id)
        ordered = _ordered_documents(prediction) if prediction else []
        document_run[sample.id] = {
            document_id: float(len(ordered) - index)
            for index, document_id in enumerate(ordered)
        }
        chunk_qrels[sample.id] = {
            chunk.chunk_id: _backend_relevance(chunk.relevance)
            for chunk in sample.gold.chunks
            if chunk.relevance > 0
        }
        has_chunk_qrels = has_chunk_qrels or bool(chunk_qrels[sample.id])
        predicted_chunks = _ordered_chunks(prediction) if prediction else []
        chunk_run[sample.id] = {
            chunk.chunk_id: float(len(predicted_chunks) - index)
            for index, chunk in enumerate(predicted_chunks)
        }

    measure_factories = {
        "recall": ir_measures.Recall,
        "precision": ir_measures.P,
        "mrr": ir_measures.RR,
        "ndcg": ir_measures.nDCG,
    }
    for k in sorted(set(k_values)):
        measures = [_measure(factory, k) for factory in measure_factories.values()]
        aggregate = ir_measures.calc_aggregate(
            measures, document_qrels, document_run
        )
        for key, value in aggregate.items():
            measure_name = str(key).split("@", 1)[0].lower()
            name_map = {"r": "recall", "p": "precision", "rr": "mrr", "ndcg": "ndcg"}
            metric_name = name_map.get(measure_name, measure_name)
            values[f"{metric_name}@{k}"] = max(0.0, min(1.0, float(value)))
            denominators[f"{metric_name}@{k}"] = len(samples)

        span_values = [
            span_ndcg_at_k(sample, prediction_by_id.get(sample.id), k) for sample in samples
        ]
        evidence_values = [
            evidence_hit_at_k(sample, prediction_by_id.get(sample.id), k)
            for sample in samples
        ]
        all_doc_values = [
            all_docs_hit_at_k(sample, prediction_by_id.get(sample.id), k)
            for sample in samples
        ]
        values[f"span_ndcg@{k}"] = safe_mean(span_values)
        values[f"evidence_hit@{k}"] = safe_mean(evidence_values)
        values[f"all_docs_hit@{k}"] = safe_mean(all_doc_values)
        denominators[f"span_ndcg@{k}"] = len(samples)
        denominators[f"evidence_hit@{k}"] = len(samples)
        denominators[f"all_docs_hit@{k}"] = len(samples)

        if has_chunk_qrels:
            chunk_measures = [_measure(factory, k) for factory in measure_factories.values()]
            chunk_aggregate = ir_measures.calc_aggregate(
                chunk_measures, chunk_qrels, chunk_run
            )
            for key, value in chunk_aggregate.items():
                measure_name = str(key).split("@", 1)[0].lower()
                name_map = {"r": "recall", "p": "precision", "rr": "mrr", "ndcg": "ndcg"}
                metric_name = name_map.get(measure_name, measure_name)
                values[f"chunk_{metric_name}@{k}"] = max(0.0, min(1.0, float(value)))
                denominators[f"chunk_{metric_name}@{k}"] = len(samples)

    return MetricSummary(values=values, denominators=denominators, counts=counts)
