# Contributing to Sentio RAG Platform

Спасибо за интерес к Sentio! Ваш вклад помогает улучшать нашу Retrieval-Augmented Generation систему для корпоративной обработки документов и интеллектуального поиска.

---

## Contribution Workflow

1. **Fork → Branch → PR**
   - Создайте ветку `feat/<topic>` или `fix/<topic>` от `main`.
   - Мелкие правки — объединяйте (squash) перед отправкой PR.
2. **Issue First**
   - Опишите проблему / предложение до начала работы.
   - Приложите логи, трассировки и минимальный пример.
3. **CI Green**
   - Все тесты (`pytest -q`) и статический анализ (`ruff`, `mypy`) должны проходить.
4. **Review & Merge**
   - Два одобрения мейнтейнеров.
   - Используйте “Re-request review” после правок.

---

## Coding Guidelines

| Stack | Standards | Notes |
|-------|-----------|-------|
| Python | PEP 8 / PEP 20 / PEP 257 | Версия 3.11+. Используйте `typing` (+ `|` pipes), `async`-функции, `pydantic` v2 models. |
| TypeScript (Web UI) | ESLint, Prettier | React 18, Vite. |
| Infrastructure | Bash, Bicep / Terraform | Скрипты должны быть идемпотентны. |

Additional rules:
* **SOLID + Clean Architecture**.
* Разбивайте файлы >400 строк, избегайте code-smells (Sonar taxonomy).
* Пишите докстринги English, комментарии English.
* Покрытие тестами ≥ 85 % для core модулей (`pytest --cov`).
* Используйте feature flags для экспериментальных изменений.

---

## Performance & Reliability

* Оптимизируйте векторные запросы (`qdrant` filters, batching).
* Кэшируйте внешние вызовы (`functools.cache` / `async TTLCache`).
* Не блокируйте event-loop: heavy CPU —> `ProcessPoolExecutor`.
* Следите за latencies в Prometheus (metrics в `/sentio/metrics`).

---

## Security & Compliance

* Следуем OWASP Top 10 (2025). Проверяем `bandit`, `trivy`.
* Секреты — только через переменные окружения / Azure Key Vault.
* Валидация входных данных (`pydantic.ValidationError` → 422).
* При изменении AI-пайплайна учитывайте GDPR / PII-masking.

---

## Documentation

* Обновляйте `README.md`, схемы Mermaid и OpenAPI при изменениях.
* Генерируйте docstrings → `mkdocs` автоматически.
* Для крупных фич заведите ADR в `docs/adr/`.

---

## License Agreement

Внося изменения, вы соглашаетесь, что ваш вклад лицензируется под **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**, как и остальной проект.

---

Спасибо, что делаете Sentio лучше! 💚
