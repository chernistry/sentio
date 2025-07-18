# Sentio vNext

## Overview

Sentio vNext is a Retrieval-Augmented Generation (RAG) pipeline implementation built on LangGraph, designed to enable pipeline visualization and enhanced management capabilities. This project represents a ground-up rewrite with a focus on modularity, observability, and production-ready deployment. The system is currently in early development phase with foundational project structure established.

## Project Structure

```
src/
├── api/                # FastAPI entrypoints (planned)
├── core/
│   ├── chunking/       # Text processing and document splitting (planned)
│   ├── embeddings/     # Embedding provider adapters (planned)
│   ├── ingest/         # Document ingestion pipeline (planned)
│   ├── llm/            # Language model interfaces (planned)
│   ├── retrievers/     # Vector search implementations (planned)
│   ├── rerankers/      # Document reranking logic (planned)
│   ├── graph/          # LangGraph node definitions (planned)
│   └── plugins/        # Extension hooks (planned)
├── utils/              # Configuration and logging utilities (planned)
├── tests/              # Test suite with pytest-asyncio (planned)
└── cli/                # Command-line interface (planned)
```

## Prerequisites

- Python 3.12 or 3.13
- Poetry 1.7.1 or higher
- Docker and Docker Compose (for containerized deployment)
- Access to a Qdrant Cloud instance (managed vector database)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd sentio-vnext
```

2. Install Poetry if not already installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install project dependencies:
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

5. Install pre-commit hooks:
```bash
pre-commit install
```

### Docker Deployment

1. Build the container:
```bash
docker build -t sentio-vnext .
```

2. Run with Docker Compose:
```bash
docker-compose up -d
```

## Configuration

### Environment Variables

The following environment variables are required for production deployment:

- `QDRANT_URL`: URL to your Qdrant Cloud instance
- `QDRANT_API_KEY`: API key for Qdrant authentication
- `PYTHONPATH`: Set to `/app` (automatically configured in Docker)

### Development Configuration

Create a `.env` file in the project root with your configuration:

```bash
QDRANT_URL=https://your-qdrant-instance.cloud
QDRANT_API_KEY=your-api-key
```

## Usage

Currently, the project contains only the foundational structure. No functional modules are implemented yet.

### CLI Interface (Planned)

The CLI will be available through Poetry scripts:

```bash
sentio --help
```

### API Interface (Planned)

The FastAPI server will run on port 8000:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Current Status / Work in Progress

This project is in early development. The following components are planned but not yet implemented:

### Phase 1 - Foundation (In Progress)
- [ ] Text chunking and splitting logic
- [ ] Document model definition
- [ ] Unit test framework setup

### Phase 2 - Core Services (Planned)
- [ ] Document ingestion CLI
- [ ] Embedding service with provider adapters
- [ ] Vector database integration
- [ ] Retrieval and reranking systems

### Phase 3 - LangGraph Integration (Planned)
- [ ] RAG pipeline state management
- [ ] Graph node implementations
- [ ] Streaming capabilities

### Phase 4 - Production Features (Planned)
- [ ] FastAPI endpoints
- [ ] Observability and logging
- [ ] Evaluation metrics
- [ ] CI/CD pipeline

Refer to `langgraph_migration.md` for detailed implementation roadmap and progress tracking.

## Troubleshooting

### Common Issues

**Poetry installation fails**
- Ensure Python 3.12+ is installed and accessible
- Try using `pip install poetry` if the curl method fails

**Docker build errors**
- Verify Docker daemon is running
- Check that all required files are present and not gitignored

**Missing environment variables**
- Ensure `.env` file is created with required Qdrant credentials
- Verify environment variables are loaded in your shell

## Contribution

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the coding standards
4. Run the test suite: `pytest`
5. Run code formatting: `black . && ruff check .`
6. Run type checking: `mypy src/`
7. Commit your changes with a descriptive message
8. Push to your fork and submit a pull request

### Code Standards

- Follow PEP 8 styling with 88-character line limit
- Use type hints for all function definitions
- Maintain test coverage above 90%
- Document all public APIs with docstrings

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Built with LangGraph for pipeline orchestration
- Uses Qdrant for vector storage and retrieval
- Powered by FastAPI for REST API implementation
- Developed with Poetry for dependency management

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Qdrant Vector Database](https://qdrant.tech/)
- [FastAPI Framework](https://fastapi.tiangolo.com/)
- [Poetry Dependency Manager](https://python-poetry.org/) 