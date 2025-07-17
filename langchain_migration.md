# LangGraph Migration Checklist

This step-by-step plan integrates LangGraph as the backbone of the LLM pipeline and
cleans up the repository structure. Tick the boxes as you progress.

## 0 · Preparation

- [x] **Create migration branch** `feat/langgraph-migration`.
- [x] **Baseline assurance** – run `pytest -q`, save coverage report.
- [x] **Add dependencies** – append `langgraph>=0.0.35` (or latest) and
      `langchain>=0.1` to *requirements*. Build Docker locally to confirm.
- [x] **Freeze versions** with `pip-compile` or similar.

## 1 · Prototype Graph Backbone

- [ ] Create package `src/core/graph/` with `__init__.py`.
- [ ] Implement `graph_factory.py` that converts legacy `Pipeline` steps into
      LangGraph nodes:
  - `InputNormalizer`
  - `Retriever`
  - `Reranker`
  - `Generator`
  - `PostProcessor`
- [ ] Build linear graph mirroring current flow; expose
      `build_basic_graph(settings: Settings) -> Graph`.
- [ ] Add smoke test `tests/test_graph_smoke.py` ensuring identical output to
      legacy pipeline on a fixed prompt.

## 2 · API Switching

- [ ] Inject the graph into FastAPI: adapt `src/api/routes.py` to call
      `graph_factory.build_basic_graph()` instead of `Pipeline`.
- [ ] Toggle via `settings.USE_LANGGRAPH` env-flag; default **on** after parity.

## 3 · Incremental Feature Ports

- [ ] HYDE expansion branch (`plugins.hyde_expander`) → optional LangGraph path.
- [ ] RAGAS evaluation node (`core.llm.ragas`) after answer generation.
- [ ] Streaming support – wrap generator node with async iterator.
- [ ] Plugin Manager hooks – allow nodes to be dynamically registered.

## 4 · Repository Restructure (after graph stable)

```
sense/
├── api/               # FastAPI (routers, deps, schemas)
├── core/
│   ├── graph/         # LangGraph graphs + nodes
│   ├── embeddings/
│   ├── llm/
│   ├── retrievers/
│   ├── rerankers/
│   └── plugins/       # lightweight hooks, moved from top-level
├── utils/
├── cli/
├── tests/
└── infra/             # docker, k8s, azure, etc.
```

- [ ] Move `plugins/*.py` under `core/plugins/`, keep thin wrappers importing old
      paths for backward compatibility.
- [ ] Delete `core/pipeline.py` once all tests use graph.

## 5 · CI + Docker

- [ ] Extend CI matrix to run both legacy + graph tests until cut-over.
- [ ] Update `docker-compose.yml` and Dockerfiles to use slim images & install
      LangGraph.
- [ ] Add `pre-commit` hooks: *ruff*, *black* (88 chars), *mypy* strict.

## 6 · Documentation

- [ ] Replace old *pipeline* diagram with LangGraph DAG in `README.md`.
- [ ] Add ADR (Architecture Decision Record) describing LangGraph adoption.

## 7 · Clean-up & De-risk

- [ ] Remove deprecated code paths and TODO comments.
- [ ] Run `pytest --cov`, aim for ≥ 90 %.
- [ ] Tag release `v0.2.0-LangGraph`.

---

### Acceptance Criteria

- All tests green.
- API returns same or better latency & accuracy.
- New nodes can be added by creating a single file & registering via Plugin
  Manager without touching core graph logic.
