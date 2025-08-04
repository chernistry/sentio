# LangGraph — Green-field Build Guide

This document is the single source of truth for the new Sentio vNext
implementation. We start from **zero code** and grow the system feature by
feature. Treat every box as a hard gate: do *not* continue until the current
stage is ✅.

---

## 0 · Prepare

- [x] Repository `sentio-vnext` created (private, MIT). URL: https://github.com/chernistry/sentio-vnext
- [x] Initialise Git, Poetry, pre-commit, Ruff, Black (88), Mypy (strict).
- [x] Add CI matrix (Linux/macOS, Python 3.12 & 3.13 alpha).
- [x] Add minimal Dockerfile + Compose (API container only). Qdrant is managed → set `QDRANT_URL` & `QDRANT_API_KEY` env vars.


---

## 1 · Project skeleton

- [x] Project skeleton generated (directory tree, empty __init__.py files, updated README, pyproject, Dockerfile, docker-compose).

```
src/
├── api/                # FastAPI entrypoints
├── core/
│   ├── chunking/       # Sentence/para splitters
│   ├── embeddings/     # Provider adapters
│   ├── ingest/         # Ingestion CLI + tasks
│   ├── llm/            # Chat/Prompt builders
│   ├── retrievers/     # Dense / hybrid search
│   ├── rerankers/      # Cross-encoders etc.
│   ├── graph/          # LangGraph nodes & factories
│   └── plugins/        # Optional hooks
├── utils/              # Settings, logging helpers
├── tests/              # Pytest with pytest-asyncio
└── cli/                # Typer commands
infra/                  # k8s, Terraform
adr/                    # Architecture decision records
docs/                   # MkDocs
```

> Rule of thumb: **everything callable lives under `core/` or `api/`, nothing
> in project root.**

---

## 2 · Chunking & Document model

- [x] `core.chunking.text_splitter` → sentence / token splitter with
      configurable `chunk_size`/`overlap`.
- [x] `core.models.Document` dataclass `{id, text, metadata}` (no pydantic yet).
- [x] Unit tests for splitter edge-cases (unicode, code snippets, huge docs).
- [x] **Extended**: Added additional chunking strategies (SEMANTIC, PARAGRAPH, FIXED) from legacy.
- [x] **Extended**: Added facade TextChunker with multi-strategy support.
- [x] **Extended**: Implemented text preprocessing, validation, and statistics tracking.

---

## 3 · Ingestion CLI

- [x] Typer command `sentio ingest <path>`.
- [x] Reads files, splits via TextChunker, stores raw chunks on disk (`data/`, temporary by default).
- [x] Emits `chunks.parquet` for deterministic tests.
- [x] **Extended**: Added robust error handling and logging.
- [x] **Extended**: Implemented support for multiple file formats (TXT, PDF, DOCX, etc.).
- [x] **Extended**: Added ingestion statistics tracking.

---

## 4 · Embedding service

- [x] `core.embeddings.base.BaseEmbedder` (async + sync wrappers).
- [x] Jina + OpenAI adapters behind factory.
- [x] Memory cache with TTL (LFU).
- [x] Warm-up on app start.
- [x] **Extended**: Added robust error handling with retries.
- [x] **Extended**: Implemented batch processing for efficiency.
- [x] **Extended**: Added usage statistics tracking.

---

## 5 · Vector DB layer (Managed)

- [x] Qdrant **Cloud** client wrapper `core.vector_store.QdrantStore` (reads `QDRANT_URL`).
- [x] Collection bootstrap migration (create via REST if absent).
- [x] Health-check ping to `/v1/collections`.
- [x] **Extended**: Integrated with document ingestion pipeline.
- [x] **Extended**: Implemented efficient batch upsert operations.

---

## 6 · Retrieval ✅

- Dense search via Qdrant (`core.retrievers.dense.DenseRetriever`).
- Hybrid search (`core.retrievers.hybrid.HybridRetriever`) combining BM25 & dense with RRF fusion.
- **Extended**: Pluggable scorers – semantic similarity, keyword matching, and MMR-based diversification.

## 7 · Reranker ✅

- Interface `Reranker.rerank(query, docs, top_k)` in `core.rerankers.base`.
- Jina-API cross-encoder adapter (`core.rerankers.jina_reranker.JinaReranker`).
- **Extended**: Uniform metadata key `metadata["score"]` for downstream metrics.

## 8 · LangGraph pipeline (in progress)

- ✅ Basic graph factory (`core.graph.factory`) with nodes: **retriever → reranker → selector**.
- ✅ `RAGState` draft defined.
- ⏳ LLM generator node & streaming wrapper – **todo**.

---

## 9 · LLM Generation

- [ ] PromptBuilder (system/instruction templates).
- [ ] Chat adapter (OpenAI / Azure / Local).
- [ ] Generation modes `{fast, balanced, quality, creative}`.

---

## 10 · Evaluation (RAGAS)

- [ ] Optional LangGraph branch after generation.
- [ ] Persist metrics in Prometheus.

---

## 11 · API & Observability

- [ ] FastAPI route `/query` (sync + streaming).
- [ ] `/ingest`, `/health`, `/stats` endpoints.
- [ ] Structured logging (OTLP) + OpenTelemetry traces.

---

## 12 · CI/CD & Dev Ex

- [ ] GitHub Actions: lint → test → build → publish Docker.
- [ ] Helm chart `charts/sentio` with values for replicas, resources, env.
- [ ] `make deploy-local` spins up stack via Compose + Traefik.

---

## 13 · Clean-up checklist before `v0.1.0`

- [ ] 90 %+ unit coverage, no `warnings` in test run.
- [ ] `ruff --select I` zero import-sorting issues.
- [ ] Review OWASP A10, fix any high findings.
- [ ] ADR-0002: **Vector store choice**.
- [ ] Tag release & push image to registry.

---

### 🚀 Roadmap after v0.1.0

1. Multi-tenant auth (JWT + per-tenant Qdrant collections).
2. Memory mode (conversational buffer in Redis).
3. Agents & tools (function calling).
4. Web UI (Next.js + shadcn/ui).

---

*End of file — keep this guide updated on every PR.*
