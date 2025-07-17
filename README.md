# Sentio vNext

LangGraph-based implementation of the Sentio RAG system.

## Project Structure

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
```

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)

## Installation

```bash
# Install dependencies
poetry install

# Set up pre-commit hooks
pre-commit install
```

## Configuration

Provide Qdrant credentials in environment variables:
```bash
export QDRANT_URL=https://your-qdrant-instance.cloud
export QDRANT_API_KEY=your-api-key
```

## Usage

```bash
# Run the API server
poetry run uvicorn src.api.main:app --reload

# Ingest data
poetry run sentio ingest path/to/documents
``` 