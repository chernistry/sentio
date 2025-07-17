# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-LangGraph] - 2025-07-17

### Added
- Integration with LangGraph for visualizable, maintainable RAG pipelines
- LangGraph Server for visual debugging and monitoring of the RAG graph
- Streaming support for token-by-token generation
- Plugin system for dynamically registering graph nodes
- HyDE expansion as an optional node in the graph
- RAGAS evaluation node for automatic quality assessment
- Comprehensive test suite for LangGraph components
- Docker support for LangGraph Server

### Changed
- Refactored pipeline steps into LangGraph nodes
- Updated Docker configuration for slim images and better caching
- Improved error handling and async initialization
- Enhanced documentation with LangGraph usage examples
- Optimized streaming implementation for better performance

### Fixed
- Blocking calls in async context
- Pipeline initialization issues
- Error handling in retrieval and generation steps

## [0.1.0] - 2025-06-01

### Added
- Initial release of Sentio RAG Pipeline
- Basic retrieval and generation functionality
- Vector store integration with Qdrant
- Embedding support for multiple providers
- Simple API for querying the RAG pipeline 