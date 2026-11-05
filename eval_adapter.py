"""production-rag adapters for the standalone rag-eval platform.

LocalRetrievalAdapter is the P4 smoke path. It reads metadata.json and runs
BM25 only, without constructing RAGService or touching MySQL, Redis, ES, or
Milvus.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from rag_eval.contracts import (
    Citation,
    EvalSample,
    Evidence,
    Prediction,
    RetrievedChunk,
    RetrievedDocument,
)
from rag_eval.registry import RunContext

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data/index/recursive/metadata.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/config.yaml"
DEFAULT_CMTEB_ROOT = PROJECT_ROOT / "data/evaluation/cmteb"
DEFAULT_CMTEB_CACHE = PROJECT_ROOT / "data/evaluation/cmteb_cache"


def _project_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _document_id(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    value = (
        item.get("document_id")
        or metadata.get("document_id")
        or item.get("file_name")
        or ""
    )
    return str(value)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class LocalRetrievalAdapter:
    """Local BM25 retrieval against a production-rag metadata.json file."""

    name = "production-rag-local-bm25"
    capabilities = {"retrieval", "latency"}

    def __init__(
        self,
        metadata_path: str | Path | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
    ):
        configured_path = (
            str(metadata_path)
            if metadata_path is not None
            else os.getenv("PRODUCTION_RAG_EVAL_METADATA")
        )
        self.metadata_path = _project_path(
            configured_path, DEFAULT_METADATA_PATH
        )
        self.top_k = int(
            top_k
            if top_k is not None
            else os.getenv("PRODUCTION_RAG_EVAL_TOP_K", "10")
        )
        self.strategy = strategy or os.getenv(
            "PRODUCTION_RAG_EVAL_STRATEGY", "recursive"
        )
        self._search = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.top_k <= 0:
            errors.append("top_k must be positive")
        if not self.metadata_path.is_file():
            errors.append(f"metadata file not found: {self.metadata_path}")
            return errors
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot load metadata: {exc}")
            return errors
        if not isinstance(payload, dict) or not payload:
            errors.append("metadata must be a non-empty JSON object")
        return errors

    def _ensure_search(self):
        if self._search is not None:
            return self._search
        from app.search.bm25_search import BM25Search
        from app.storage.base import MetadataChunkRepository
        from app.storage.metadata_store import MetadataStore

        metadata = MetadataStore().load(str(self.metadata_path))
        repository = MetadataChunkRepository(metadata)
        self._search = BM25Search(repository)
        self._search.strategy = self.strategy
        return self._search

    def warmup(self, context: RunContext) -> None:
        self._ensure_search()

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        if sample.task != "retrieval":
            raise ValueError(f"local BM25 adapter does not support task {sample.task!r}")
        started = time.perf_counter()
        hits = self._ensure_search().search(
            sample.query, top_k=self.top_k
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        documents: list[RetrievedDocument] = []
        chunks: list[RetrievedChunk] = []
        evidence: list[Evidence] = []
        seen_documents: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            document_id = _document_id(hit)
            chunk_id = str(hit.get("chunk_id") or "")
            start = _int_value(hit.get("start_offset"))
            end = _int_value(hit.get("end_offset"))
            if not document_id or not chunk_id:
                continue
            span = None
            if end > start:
                span = {"start": start, "end": end}
            if document_id not in seen_documents:
                seen_documents.add(document_id)
                documents.append(
                    RetrievedDocument(
                        document_id=document_id,
                        score=float(hit.get("score", 0.0) or 0.0),
                        rank=len(documents) + 1,
                    )
                )
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    score=float(hit.get("score", 0.0) or 0.0),
                    rank=rank,
                    span=span,
                    text=hit.get("content"),
                    metadata=hit.get("metadata") or {},
                )
            )
            evidence.append(
                Evidence(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    span=span,
                    text=hit.get("content"),
                )
            )

        return Prediction(
            sample_id=sample.id,
            status="ok",
            documents=documents,
            chunks=chunks,
            evidence=evidence,
            latency_ms=elapsed_ms,
            usage={"backend": "local_bm25", "strategy": self.strategy},
            metadata={"retrieval_mode": "bm25", "run_id": context.run_id},
        )


class ProductionRAGServiceAdapter:
    """Full RAGService adapter for explicit integration runs."""

    name = "production-rag-service"
    capabilities = {"retrieval", "generation", "citation", "latency"}

    def __init__(
        self,
        config_path: str | Path | None = None,
        strategy: str | None = None,
        mode: str | None = None,
        use_rerank: bool | None = None,
        tenant_id: str | None = None,
    ):
        configured = (
            str(config_path)
            if config_path is not None
            else os.getenv("PRODUCTION_RAG_EVAL_CONFIG")
        )
        self.config_path = _project_path(configured, DEFAULT_CONFIG_PATH)
        self.strategy = strategy or os.getenv(
            "PRODUCTION_RAG_EVAL_STRATEGY", "recursive"
        )
        self.mode = mode or os.getenv("PRODUCTION_RAG_EVAL_MODE", "bm25")
        env_rerank = os.getenv("PRODUCTION_RAG_EVAL_RERANK")
        self.use_rerank = (
            use_rerank
            if use_rerank is not None
            else str(env_rerank or "false").lower() in {"1", "true", "yes", "on"}
        )
        self.tenant_id = tenant_id or os.getenv(
            "PRODUCTION_RAG_EVAL_TENANT", "default"
        )
        self._service = None

    def validate(self) -> list[str]:
        if not self.config_path.is_file():
            return [f"config file not found: {self.config_path}"]
        return []

    def _ensure_service(self):
        if self._service is None:
            from app.core.config import Config
            from app.rag.service import RAGService

            self._service = RAGService(
                config=Config(str(self.config_path)),
                strategy=self.strategy,
                mode=self.mode,
                use_rerank=self.use_rerank,
                tenant_id=self.tenant_id,
            )
        return self._service

    def warmup(self, context: RunContext) -> None:
        self._ensure_service()

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        started = time.perf_counter()
        response = self._ensure_service().query(
            sample.query, history=sample.history or None
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        chunks, documents = _map_service_chunks(response.chunks)
        chunk_lookup = {chunk.chunk_id: chunk.document_id for chunk in chunks}
        citations: list[Citation] = []
        for item in response.citations:
            chunk_id = str(item.get("chunk_id") or "") or None
            document_id = str(
                item.get("document_id")
                or chunk_lookup.get(chunk_id or "", "")
                or item.get("file_name")
                or ""
            )
            if document_id:
                citations.append(
                    Citation(document_id=document_id, chunk_id=chunk_id)
                )
        return Prediction(
            sample_id=sample.id,
            status="ok",
            documents=documents,
            chunks=chunks,
            answer=response.answer,
            citations=citations,
            evidence=[
                Evidence(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    span=chunk.span,
                    text=chunk.text,
                )
                for chunk in chunks
            ],
            latency_ms=elapsed_ms,
            usage={"backend": "rag_service", "mode": self.mode},
            metadata={
                "run_id": context.run_id,
                "strategy": self.strategy,
                "tenant_id": self.tenant_id,
                "stats": response.stats,
            },
        )


def _map_service_chunks(
    items: list[dict[str, Any]],
) -> tuple[list[RetrievedChunk], list[RetrievedDocument]]:
    chunks: list[RetrievedChunk] = []
    documents: list[RetrievedDocument] = []
    seen_documents: set[str] = set()
    for rank, item in enumerate(items, start=1):
        document_id = _document_id(item)
        chunk_id = str(item.get("chunk_id") or "")
        if not document_id or not chunk_id:
            continue
        start = _int_value(item.get("start_offset"))
        end = _int_value(item.get("end_offset"))
        span = {"start": start, "end": end} if end > start else None
        score = float(
            item.get("rerank_score", item.get("score", 0.0)) or 0.0
        )
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                score=score,
                rank=rank,
                span=span,
                text=item.get("content"),
                metadata=item.get("metadata") or {},
            )
        )
        if document_id not in seen_documents:
            seen_documents.add(document_id)
            documents.append(
                RetrievedDocument(
                    document_id=document_id,
                    score=score,
                    rank=len(documents) + 1,
                )
            )
    return chunks, documents


HttpTransport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class HttpEvaluationAdapter:
    """Optional HTTP adapter for a running production-rag API."""

    name = "production-rag-http"
    capabilities = {"retrieval", "generation", "citation", "latency"}

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
        timeout_seconds: float | None = None,
        transport: HttpTransport | None = None,
        strategy: str | None = None,
        mode: str | None = None,
        use_rerank: bool | None = None,
    ):
        base_url = (endpoint or os.getenv("PRODUCTION_RAG_EVAL_URL", "")).rstrip("/")
        if base_url and not base_url.endswith("/api/chat"):
            base_url = base_url + "/api/chat"
        self.endpoint = base_url
        self.token = token or os.getenv("PRODUCTION_RAG_EVAL_TOKEN")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("PRODUCTION_RAG_EVAL_HTTP_TIMEOUT", "60")
        )
        self.transport = transport or _urllib_transport
        self.strategy = strategy or os.getenv(
            "PRODUCTION_RAG_EVAL_STRATEGY", "recursive"
        )
        self.mode = mode or os.getenv("PRODUCTION_RAG_EVAL_MODE", "bm25")
        env_rerank = os.getenv("PRODUCTION_RAG_EVAL_RERANK", "false")
        self.use_rerank = (
            use_rerank
            if use_rerank is not None
            else str(env_rerank).lower() in {"1", "true", "yes", "on"}
        )

    def validate(self) -> list[str]:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ["PRODUCTION_RAG_EVAL_URL must be an absolute http(s) URL"]
        if self.timeout_seconds <= 0:
            return ["HTTP timeout must be positive"]
        return []

    def warmup(self, context: RunContext) -> None:
        return None

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        payload = {
            "query": sample.query,
            "history": sample.history,
            "strategy": self.strategy,
            "mode": self.mode,
            "use_rerank": self.use_rerank,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        started = time.perf_counter()
        response = self.transport(
            self.endpoint, payload, headers, self.timeout_seconds
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_chunks = response.get("chunks") or []
        chunks, documents = _map_service_chunks(raw_chunks)
        lookup = {chunk.chunk_id: chunk.document_id for chunk in chunks}
        citations: list[Citation] = []
        for item in response.get("citations") or []:
            chunk_id = str(item.get("chunk_id") or "") or None
            document_id = str(
                item.get("document_id")
                or lookup.get(chunk_id or "", "")
                or item.get("file_name")
                or ""
            )
            if document_id:
                citations.append(
                    Citation(document_id=document_id, chunk_id=chunk_id)
                )
        return Prediction(
            sample_id=sample.id,
            status="ok",
            documents=documents,
            chunks=chunks,
            answer=response.get("answer"),
            citations=citations,
            evidence=[
                Evidence(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    span=chunk.span,
                    text=chunk.text,
                )
                for chunk in chunks
            ],
            latency_ms=elapsed_ms,
            usage={"backend": "http_api"},
            metadata={"run_id": context.run_id, "stats": response.get("stats", {})},
        )


def _urllib_transport(
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from evaluation endpoint: {detail}"
        ) from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("HTTP evaluation response must be a JSON object")
    return parsed


class CmtebRetrievalAdapter:
    """C-MTEB T2Retrieval adapter that reuses the existing benchmark pipeline."""

    name = "production-rag-cmteb"
    capabilities = {"retrieval", "latency"}

    def __init__(
        self,
        data_root: str | Path | None = None,
        max_corpus: int | None = None,
        max_queries: int | None = None,
        fill_factor: int | None = None,
        seed: int | None = None,
        top_k: int | None = None,
        embedding_cache: str | Path | None = None,
    ):
        self.data_root = _project_path(
            str(data_root)
            if data_root is not None
            else os.getenv("PRODUCTION_RAG_CMTEB_ROOT"),
            DEFAULT_CMTEB_ROOT,
        )
        self.max_corpus = int(
            max_corpus
            if max_corpus is not None
            else os.getenv("PRODUCTION_RAG_CMTEB_MAX_CORPUS", "1000")
        )
        self.max_queries = int(
            max_queries
            if max_queries is not None
            else os.getenv("PRODUCTION_RAG_CMTEB_MAX_QUERIES", "50")
        )
        self.fill_factor = int(
            fill_factor
            if fill_factor is not None
            else os.getenv("PRODUCTION_RAG_CMTEB_FILL_FACTOR", "5")
        )
        self.seed = int(
            seed if seed is not None else os.getenv("PRODUCTION_RAG_CMTEB_SEED", "42")
        )
        self.top_k = int(
            top_k
            if top_k is not None
            else os.getenv("PRODUCTION_RAG_CMTEB_TOP_K", "10")
        )
        self.embedding_cache = _project_path(
            str(embedding_cache)
            if embedding_cache is not None
            else os.getenv("PRODUCTION_RAG_CMTEB_CACHE"),
            DEFAULT_CMTEB_CACHE,
        )
        self._ready = None

    def _benchmark_module(self):
        script_path = PROJECT_ROOT / "scripts/eval_cmteb.py"
        if not script_path.is_file():
            raise FileNotFoundError(f"C-MTEB benchmark script not found: {script_path}")
        scripts_dir = str(script_path.parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import eval_cmteb

        return eval_cmteb

    def validate(self) -> list[str]:
        errors: list[str] = []
        for path in (
            self.data_root / "C-MTEB/T2Retrieval/data",
            self.data_root / "C-MTEB/T2Retrieval-qrels/data",
        ):
            if not path.is_dir():
                errors.append(f"C-MTEB data directory not found: {path}")
        if min(
            self.max_corpus,
            self.max_queries,
            self.fill_factor,
            self.top_k,
        ) <= 0:
            errors.append("C-MTEB limits must be positive")
        return errors

    def _ensure_ready(self):
        if self._ready is not None:
            return self._ready
        benchmark = self._benchmark_module()
        from app.core.config import Config
        from app.embedding.model import EmbeddingModel

        corpus, queries, relevant_map = benchmark.load_data(
            self.max_corpus,
            self.max_queries,
            self.fill_factor,
            seed=self.seed,
        )
        config = Config()
        model = EmbeddingModel(config.embedding_model)
        normalize = config.data.get("embedding", {}).get("normalize", True)
        corpus_ids = corpus["id"].astype(str).tolist()
        corpus_hash = benchmark.hash_corpus(corpus)
        cache = benchmark.EmbeddingCache(str(self.embedding_cache))
        embeddings = cache.load(
            config.embedding_model, normalize, corpus_hash, model.dimension
        )
        if embeddings is None:
            embeddings = model.encode(list(corpus["text"]))
            cache.save(
                config.embedding_model,
                normalize,
                corpus_hash,
                embeddings,
                corpus_ids,
            )
        index = benchmark.build_index(embeddings)
        self._ready = {
            "benchmark": benchmark,
            "corpus": corpus,
            "corpus_ids": corpus_ids,
            "relevant_map": relevant_map,
            "model": model,
            "index": index,
            "embeddings": embeddings,
            "query_ids": set(queries["id"].astype(str)),
        }
        return self._ready

    def warmup(self, context: RunContext) -> None:
        self._ensure_ready()

    def run(self, sample: EvalSample, context: RunContext) -> Prediction:
        import numpy as np

        ready = self._ensure_ready()
        if sample.id not in ready["query_ids"]:
            raise ValueError(
                f"sample {sample.id!r} is outside the configured C-MTEB subset"
            )
        started = time.perf_counter()
        query_vector = ready["model"].encode([sample.query])
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = ready["index"].search(query_vector, self.top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        corpus = ready["corpus"]
        documents: list[RetrievedDocument] = []
        chunks: list[RetrievedChunk] = []
        for rank, (row_index, score) in enumerate(
            zip(indices[0], scores[0]), start=1
        ):
            if row_index < 0:
                continue
            row = corpus.iloc[int(row_index)]
            document_id = str(row["id"])
            text = str(row["text"])
            documents.append(
                RetrievedDocument(
                    document_id=document_id,
                    score=float(score),
                    rank=rank,
                )
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=document_id,
                    document_id=document_id,
                    score=float(score),
                    rank=rank,
                    text=text,
                )
            )
        return Prediction(
            sample_id=sample.id,
            status="ok",
            documents=documents,
            chunks=chunks,
            latency_ms=elapsed_ms,
            usage={
                "backend": "cmteb_faiss",
                "corpus_size": len(ready["corpus_ids"]),
            },
            metadata={"run_id": context.run_id},
        )
