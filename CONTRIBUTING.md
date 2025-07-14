# Contributing to the Sentio RAG Platform

Thank you for your interest in Sentio. Your contributions help enhance our Retrieval-Augmented Generation platform for enterprise-grade document processing and intelligent search.

---

## Contribution Workflow

1. **Fork → Branch → PR**
   - Create a branch `feat/<topic>` or `fix/<topic>` from `main`.
   - For small changes, squash commits before submitting a PR.
2. **Issue First**
   - Describe the problem or proposal before starting work.
   - Attach logs, traces, and a minimal reproducible example.
3. **CI Green**
   - All tests (`pytest -q`) and static analysis (`ruff`, `mypy`) must pass.
4. **Review & Merge**
   - Two maintainer approvals required.
   - Use “Re-request review” after making changes.

---

## Coding Guidelines

| Stack                | Standards                | Notes                                                                 |
|----------------------|-------------------------|-----------------------------------------------------------------------|
| Python               | PEP 8 / PEP 20 / PEP 257| Version 3.11+. Use `typing` (+ `|` pipes), `async` functions, `pydantic` v2 models. |
| TypeScript (Web UI)  | ESLint, Prettier        | React 18, Vite.                                                       |
| Infrastructure       | Bash, Bicep / Terraform | Scripts must be idempotent.                                           |

Additional rules:
* **SOLID + Clean Architecture**.
* Split files >400 lines, avoid code smells (Sonar taxonomy).
* Write docstrings and comments in English.
* Test coverage ≥ 85% for core modules (`pytest --cov`).
* Use feature flags for experimental changes.

---

## Performance & Reliability

* Optimize vector queries (`qdrant` filters, batching).
* Cache external calls (`functools.cache` / `async TTLCache`).
* Do not block the event loop: for heavy CPU tasks use `ProcessPoolExecutor`.
* Monitor latencies in Prometheus (metrics at `/sentio/metrics`).

---

## Security & Compliance

* Follow OWASP Top 10 (2025). Check with `bandit`, `trivy`.
* Secrets must be managed via environment variables or Azure Key Vault.
* Validate input data (`pydantic.ValidationError` → 422).
* For AI pipeline changes, consider GDPR / PII-masking.

---

## Documentation

* Update `README.md`, Mermaid diagrams, and OpenAPI specs when making changes.
* Generate docstrings → `mkdocs` automatically.
* For major features, create an ADR in `docs/adr/`.

---

## License Agreement

By contributing, you agree that your work is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**, as is the rest of the project.

---

Thank you for contributing to Sentio.
