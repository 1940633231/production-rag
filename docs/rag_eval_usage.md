# rag-eval production-rag integration

## Local smoke

Install the package once from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e packages\rag-eval
```

Run the four migrated smoke queries:

```powershell
.\.venv\Scripts\rag-eval.exe validate data\evaluation\rag_eval\production_smoke
.\.venv\Scripts\rag-eval.exe run --config configs\eval.local.yaml
```

`LocalRetrievalAdapter` reads `data/index/recursive/metadata.json` and uses
local BM25. It does not construct `RAGService` and does not access MySQL,
Redis, Elasticsearch, or Milvus.

Override local defaults with:

- `PRODUCTION_RAG_EVAL_METADATA`
- `PRODUCTION_RAG_EVAL_STRATEGY`
- `PRODUCTION_RAG_EVAL_TOP_K`

## C-MTEB subset

Regenerate the deterministic 50-query dataset:

```powershell
.\.venv\Scripts\python.exe scripts\migrate_rag_eval_datasets.py cmteb-50 `
  --max-corpus 1000 --max-queries 50 --fill-factor 5 --seed 42
```

Run or regenerate the baseline:

```powershell
.\.venv\Scripts\rag-eval.exe run `
  --config configs\eval.cmteb.yaml `
  --save-baseline reports\rag-eval\baselines\cmteb_50.json
```

The corpus embedding cache is stored under `data/evaluation/cmteb_cache/`.
The cache is local and ignored by Git.

## Gate

```powershell
.\.venv\Scripts\rag-eval.exe gate `
  --config configs\eval.cmteb.yaml `
  --baseline reports\rag-eval\baselines\cmteb_50.json `
  --report reports\rag-eval\cmteb\<run-id>\report.json
```

The gate checks dataset hash, execution configuration, coverage, and metric
drop thresholds. It does not start external services.

## Full service adapter

`ProductionRAGServiceAdapter` builds the existing `RAGService` lazily:

```yaml
adapter: eval_adapter:ProductionRAGServiceAdapter
```

Configuration:

- `PRODUCTION_RAG_EVAL_CONFIG`
- `PRODUCTION_RAG_EVAL_STRATEGY`
- `PRODUCTION_RAG_EVAL_MODE`
- `PRODUCTION_RAG_EVAL_RERANK`
- `PRODUCTION_RAG_EVAL_TENANT`

This adapter can use vector, BM25, or hybrid mode according to the production
configuration. Start required services separately with `scripts/dev_up.ps1`
only when an integration run actually needs them.

## HTTP adapter

`HttpEvaluationAdapter` targets `POST /api/chat` on a running service:

```yaml
adapter: eval_adapter:HttpEvaluationAdapter
```

- `PRODUCTION_RAG_EVAL_URL`: base URL or full `/api/chat` URL
- `PRODUCTION_RAG_EVAL_TOKEN`: optional bearer token
- `PRODUCTION_RAG_EVAL_HTTP_TIMEOUT`

The adapter is optional and is not used by the local smoke or C-MTEB baseline.
