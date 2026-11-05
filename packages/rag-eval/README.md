# rag-eval

Standalone evaluation contracts and runners for retrieval, generation, and
agent workflows. The package does not import production service code and does
not start Docker, Elasticsearch, or other external services.

Adapters may optionally implement `warmup(context)`. The runner calls it once
before the first cache miss, outside the per-sample timeout, so expensive model
or index initialization is not mistaken for a timed-out sample.
