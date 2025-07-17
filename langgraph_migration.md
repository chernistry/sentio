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

- [ ] `core.chunking.text_splitter` → sentence / token splitter with
      configurable `chunk_size`/`overlap`.
- [ ] `core.models.Document` dataclass `{id, text, metadata}` (no pydantic yet).
- [ ] Unit tests for splitter edge-cases (unicode, code snippets, huge docs).

---

## 3 · Ingestion CLI

- [ ] Typer command `sentio ingest <path>`.
- [ ] Reads files, splits via splitter, stores raw chunks on disk (`data/`).
- [ ] Emits `chunks.parquet` for deterministic tests.

---

## 4 · Embedding service

- [ ] `core.embeddings.base.BaseEmbedder` (async + sync wrappers).
- [ ] Jina + OpenAI adapters behind factory.
- [ ] Memory cache with TTL (LFU).
- [ ] Warm-up on app start.

---

## 5 · Vector DB layer (Managed)

- [ ] Qdrant **Cloud** client wrapper `core.vector_store.QdrantStore` (reads `QDRANT_URL`).
- [ ] Collection bootstrap migration (create via REST if absent).
- [ ] Health-check ping to `/v1/collections`.

---

## 6 · Retrieval

- [ ] Dense search via Qdrant.
- [ ] Hybrid search (`SparseBM25 + Dense`) with RRF fusion.
- [ ] Pluggable scorers.

---

## 7 · Reranker

- [ ] Interface `Reranker.rerank(query, docs, top_k)`.
- [ ] Default: mini-cross-encoder (sentence-transformers) loaded lazily.

---

## 8 · LangGraph pipeline

- [ ] Define `RAGState` (pydantic v2 BaseModel).
- [ ] Implement nodes: normaliser → retriever → reranker → generator →
      post-processor.
- [ ] `graph_factory.build_basic_graph(cfg)` returns compiled graph.
- [ ] Provide streaming wrapper (`astream`).

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
