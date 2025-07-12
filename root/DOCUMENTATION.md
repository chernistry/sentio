# Sentio RAG System - Technical Documentation

**Version:** 1.0.0  
**Author:** Sentio RAG Development Team  
**License:** MIT

---

## Table of Contents

1. [API Reference](#api-reference)
2. [System Architecture](#system-architecture)
3. [Deployment Guide](#deployment-guide)
4. [Operations Manual](#operations-manual)
5. [Security Guide](#security-guide)

---

# API Reference

## Overview

The Sentio RAG API provides a RESTful interface for intelligent document retrieval and question answering. The API is built on FastAPI with automatic OpenAPI specification generation and comprehensive error handling.

## Base URL
```
Production: https://your-domain.com/
Development: http://localhost:8910/
```

## Authentication

The API currently operates without authentication for internal deployments. For production environments, implement API key authentication or OAuth 2.0 integration at the gateway level.

## Content Types

All API endpoints accept and return `application/json` unless otherwise specified.

## Rate Limiting

Default rate limits apply to prevent resource exhaustion:
- **Standard endpoints**: 100 requests per minute
- **Chat endpoints**: 20 requests per minute
- **Bulk operations**: 5 requests per minute

## Error Handling

### Standard HTTP Status Codes

| Code | Description | Usage |
|------|-------------|--------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid request parameters or body |
| 422 | Validation Error | Request body validation failed |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Dependent service unavailable |

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "request_id": "uuid-correlation-id"
}
```

## Data Models

### ChatRequest

Request model for chat interactions.

```json
{
  "question": "string",
  "history": [
    {
      "role": "user|assistant", 
      "content": "string"
    }
  ]
}
```

**Fields:**
- `question` (required): User query string, max 1000 characters
- `history` (optional): Conversation context, max 10 exchanges

### ChatResponse

Response model for chat interactions.

```json
{
  "answer": "string",
  "sources": [
    {
      "text": "string",
      "source": "string", 
      "score": "number"
    }
  ]
}
```

**Fields:**
- `answer`: Generated response text
- `sources`: Array of source documents with relevance scores

### Source

Document source with relevance information.

```json
{
  "text": "string",
  "source": "string",
  "score": "number"
}
```

**Fields:**
- `text`: Relevant document excerpt
- `source`: Document identifier or filename
- `score`: Relevance score (0.0-1.0)

## Endpoints

### Health Check

Monitor system health and service availability.

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "services": {
    "qdrant": "healthy",
    "jina_ai": "healthy"
  }
}
```

### Chat Interface

Process user questions and return contextual answers.

```http
POST /chat
```

**Request Body:**
```json
{
  "question": "What are the security policies for remote access?",
  "history": [
    {
      "role": "user",
      "content": "Tell me about VPN requirements"
    },
    {
      "role": "assistant", 
      "content": "VPN access requires multi-factor authentication..."
    }
  ]
}
```

**Response:**
```json
{
  "answer": "Based on the security documentation, remote access requires the following protocols...",
  "sources": [
    {
      "text": "Remote access security policy requires VPN connection with MFA...",
      "source": "security-policy.pdf",
      "score": 0.89
    }
  ]
}
```

## Integration Examples

### Python Client

```python
import httpx
import asyncio

class SentioClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def ask_question(self, question: str, history: list = None):
        """Ask a question to the Sentio RAG system."""
        payload = {
            "question": question,
            "history": history or []
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json()
```

### cURL Examples

**Basic question:**
```bash
curl -X POST http://localhost:8910/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the incident response procedure?",
    "history": []
  }'
```

---

# System Architecture

## Executive Summary

Sentio is a distributed Retrieval-Augmented Generation (RAG) system designed for enterprise-scale document processing and intelligent query response. The architecture emphasizes scalability, reliability, and maintainability through microservices decomposition, event-driven processing, and modern cloud-native patterns.

## Architectural Principles

### Design Philosophy
- **Separation of Concerns**: Clear boundaries between data ingestion, retrieval, and generation
- **Scalability**: Horizontal scaling capabilities for all critical components
- **Fault Tolerance**: Graceful degradation and error recovery mechanisms
- **Observability**: Comprehensive monitoring, logging, and tracing
- **Security**: Defense-in-depth security model with encryption and access controls

### Technical Principles
- **API-First Design**: REST-based interfaces with OpenAPI specifications
- **Stateless Services**: Horizontally scalable stateless components
- **Data Consistency**: Eventually consistent model with conflict resolution
- **Performance Optimization**: Sub-second response times for user queries
- **Resource Efficiency**: Optimal utilization of compute and storage resources

## System Overview

### High-Level Architecture

```mermaid
┌─────────────────────────────────────────────────────────────────┐
│                           Client Layer                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Web Apps      │   Mobile Apps   │      API Clients            │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                              │
│              (Authentication, Rate Limiting, CORS)             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Sentio Core Platform                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Ingestion      │   Query         │     Management              │
│  Service        │   Service       │     Service                 │
└─────────────────┴─────────────────┴─────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      External Services                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Jina AI       │   Vector DB     │      LLM Service            │
│  (Embed/Rank)   │   (Qdrant)      │     (OpenRouter)                │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Core Components

### 1. API Gateway Layer

**Purpose**: Entry point for all external requests with cross-cutting concerns.

**Responsibilities**:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- CORS policy enforcement
- SSL termination

**Technology Stack**:
- **Framework**: FastAPI with Uvicorn ASGI server
- **Middleware**: Custom middleware for CORS, authentication
- **Monitoring**: Prometheus metrics integration

### 2. Document Ingestion Service

**Purpose**: Process and index documents into the vector database.

**Processing Pipeline**:
1. **Document Loading**: Multi-format support (PDF, Markdown, TXT)
2. **Content Extraction**: Text extraction with structure preservation
3. **Segmentation**: Intelligent chunking via Jina AI Segmenter
4. **Embedding Generation**: Vector embedding via Jina AI Embeddings v3
5. **Metadata Enrichment**: Document metadata and indexing information
6. **Vector Storage**: Batch insertion into Qdrant vector database

### 3. Query Processing Service

**Purpose**: Handle user queries through retrieval and generation pipeline.

**Processing Stages**:
1. **Query Validation**: Input sanitization and validation
2. **Embedding Generation**: Query vectorization using Jina AI
3. **Similarity Search**: K-nearest neighbor search in Qdrant
4. **Document Reranking**: Relevance reranking via Jina AI Reranker
5. **Context Assembly**: Document context preparation
6. **Answer Generation**: LLM-based response generation via Ollama
7. **Response Formatting**: Structured response with sources

### 4. Vector Database (Qdrant)

**Purpose**: High-performance vector storage and similarity search.

**Configuration**:
- **Vector Dimensions**: 1024 (Jina Embeddings v3)
- **Distance Metric**: Cosine similarity
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Quantization**: Scalar quantization for memory efficiency

## Technology Stack Summary

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| API Framework | FastAPI | 0.111.0 | REST API development |
| ASGI Server | Uvicorn | 0.29.0 | High-performance HTTP server |
| Vector Database | Qdrant | Latest | Vector storage and search |
| LLM Service | **OpenRouter API (cloud, default)** | N/A | Language model generation |
| Embedding Service | **Jina AI Cloud** | API | Text embedding generation |
| Reranking Service | Jina AI Reranker | API | Result reranking |
| Container Runtime | Docker | Latest | Application containerization |

> Optional local components (Ollama embeddings/LLM, transformers-based rerankers, web search, HyDE, RAGAS, etc.) are available in the `plugins/` directory and can be enabled for advanced use-cases beyond the default PoC deployment.

---

# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Sentio RAG system in production environments. The deployment strategy emphasizes reliability, scalability, and operational excellence through infrastructure as code, automated deployment pipelines, and comprehensive monitoring.

## Infrastructure Requirements

### Minimum Production Requirements

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| API Service | 2 cores | 4GB | 20GB | 1Gbps |
| Qdrant | 4 cores | 16GB | 200GB SSD | 1Gbps |
| Load Balancer | 2 cores | 4GB | 50GB | 10Gbps |
| **Total** | **8 cores** | **24GB** | **270GB** | **12Gbps** |

### Recommended Production Requirements

| Component | CPU | Memory | Storage | Network | Replicas |
|-----------|-----|--------|---------|---------|----------|
| API Service | 4 cores | 8GB | 50GB | 1Gbps | 3 |
| Qdrant Cluster | 8 cores | 32GB | 1TB NVMe | 10Gbps | 3 |
| Redis Cache | 4 cores | 16GB | 100GB | 1Gbps | 2 |
| **Total** | **40 cores** | **144GB** | **1.3TB** | **32Gbps** | **8** |

## Container Deployment

### Docker Production Configuration

#### Multi-Stage Dockerfile

```dockerfile
# Production Dockerfile for Sentio API
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim as production

# Security: Create non-root user
RUN useradd --create-home --shell /bin/bash Sentio
USER Sentio
WORKDIR /home/Sentio/app

# Copy application code and dependencies
COPY --from=builder /root/.local /home/Sentio/.local
COPY --chown=Sentio:Sentio . .

# Environment configuration
ENV PATH=/home/Sentio/.local/bin:$PATH
ENV PYTHONPATH=/home/Sentio/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8910/health || exit 1

# Expose application port
EXPOSE 8910

# Start application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8910", "--workers", "4"]
```

## Kubernetes Deployment

### Namespace Configuration

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: Sentio-rag
  labels:
    name: Sentio-rag
    environment: production
```

### ConfigMap for Application Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: Sentio-config
  namespace: Sentio-rag
data:
  QDRANT_URL: "http://qdrant-service:6333"
  LOG_LEVEL: "INFO"
  COLLECTION_NAME: "Sentio_docs"
  TOP_K_RETRIEVAL: "10"
  TOP_K_RERANK: "3"
```

### API Service Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: Sentio-api
  namespace: Sentio-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: Sentio-api
  template:
    metadata:
      labels:
        app: Sentio-api
    spec:
      containers:
      - name: api
        image: Sentio-rag:latest
        ports:
        - containerPort: 8910
        envFrom:
        - configMapRef:
            name: Sentio-config
        - secretRef:
            name: Sentio-secrets
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8910
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8910
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Environment Configuration

### Production Environment Variables

```bash
# Sentio RAG Production Environment Configuration

# Core Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
DEBUG=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8910
API_WORKERS=4
API_TIMEOUT=300

# External Service URLs
QDRANT_URL=http://qdrant-cluster:6333
REDIS_URL=redis://redis-cluster:6379

# Model Configuration
EMBEDDING_MODEL=jina-embeddings-v3
RERANKER_MODEL=jina-reranker-v2-base-multilingual

# Processing Parameters
COLLECTION_NAME=Sentio_docs
TOP_K_RETRIEVAL=10
TOP_K_RERANK=3
VECTOR_SIZE=1024

# Security Configuration
JINA_API_KEY=${JINA_API_KEY}
CORS_ORIGINS=["https://app.Sentio.company.com"]
RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Performance Tuning
CONNECTION_POOL_SIZE=20
MAX_CONCURRENT_REQUESTS=50
CACHE_TTL_SECONDS=3600

# Monitoring Configuration
PROMETHEUS_METRICS_ENABLED=true
TRACING_ENABLED=true
JAEGER_ENDPOINT=http://jaeger:14268
```

---

# Operations Manual

## Overview

This operations manual provides comprehensive guidance for monitoring, maintaining, and troubleshooting the Sentio RAG system in production environments. The document follows industry best practices for operational excellence and incident management.

## System Health Monitoring

### Key Performance Indicators (KPIs)

#### Application Layer Metrics

| Metric | Target | Critical Threshold | Alert Condition |
|--------|--------|-------------------|-----------------|
| API Response Time (P95) | < 2s | > 5s | Sustained 2+ minutes |
| API Error Rate | < 1% | > 5% | Sustained 1+ minute |
| API Availability | > 99.9% | < 99% | Any downtime |
| Request Throughput | Baseline ±20% | > 50% deviation | Sustained 5+ minutes |
| Memory Usage | < 80% | > 90% | Current usage |
| CPU Usage | < 70% | > 85% | Sustained 5+ minutes |

#### Vector Database Metrics

| Metric | Target | Critical Threshold | Alert Condition |
|--------|--------|-------------------|-----------------|
| Search Latency (P95) | < 100ms | > 500ms | Sustained 2+ minutes |
| Index Memory Usage | < 80% | > 90% | Current usage |
| Disk Usage | < 70% | > 85% | Current usage |
| Connection Pool Usage | < 80% | > 95% | Current usage |
| Query Success Rate | > 99.5% | < 99% | Sustained 1+ minute |

### Monitoring Infrastructure

#### Prometheus Metrics Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'Sentio-production'
    environment: 'production'

scrape_configs:
  - job_name: 'Sentio-api'
    static_configs:
      - targets: ['Sentio-api:8910']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'
```

## Incident Response Procedures

### Incident Classification

#### Severity Levels

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| P0 - Critical | Complete service outage | 15 minutes | Total system down, data loss |
| P1 - High | Major service degradation | 30 minutes | High error rates, slow responses |
| P2 - Medium | Partial service impact | 2 hours | Non-critical feature failure |
| P3 - Low | Minor issues | 24 hours | Performance degradation |

### Runbooks

#### API Service Failure

**Symptoms:**
- API health check failures
- High error rates (5xx responses)
- Complete API unresponsiveness

**Investigation Steps:**
1. Check application logs:
   ```bash
   kubectl logs -f deployment/Sentio-api -n Sentio-rag --tail=100
   ```

2. Verify pod status:
   ```bash
   kubectl get pods -n Sentio-rag -l app=Sentio-api
   ```

3. Check resource utilization:
   ```bash
   kubectl top pods -n Sentio-rag
   ```

**Resolution Actions:**
1. **Rolling Restart**: 
   ```bash
   kubectl rollout restart deployment/Sentio-api -n Sentio-rag
   ```

2. **Scale Up**:
   ```bash
   kubectl scale deployment/Sentio-api --replicas=5 -n Sentio-rag
   ```

## Maintenance Procedures

### Scheduled Maintenance

#### Monthly Maintenance Tasks

1. **Security Updates**
   - Update base Docker images
   - Apply security patches to dependencies
   - Review and update SSL certificates
   - Audit access controls and permissions

2. **Performance Optimization**
   - Analyze query performance metrics
   - Optimize database indices
   - Review and tune cache configurations
   - Clean up old logs and temporary files

3. **Backup Verification**
   - Test backup restoration procedures
   - Verify backup integrity and completeness
   - Update disaster recovery documentation
   - Test failover mechanisms

### Database Maintenance

#### Vector Database Optimization

```bash
#!/bin/bash
# qdrant_optimization.sh - Qdrant maintenance script

NAMESPACE="Sentio-rag"
COLLECTION="Sentio_docs"
QDRANT_URL="http://qdrant-service:6333"

# Optimize collection indices
optimize_indices() {
    echo "Optimizing collection indices..."
    curl -X POST "$QDRANT_URL/collections/$COLLECTION/index" \
        -H "Content-Type: application/json" \
        -d '{
            "operation": "optimize",
            "optimize_config": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000
            }
        }'
}
```

## Performance Optimization

### Query Performance Tuning

#### Search Optimization Guidelines

1. **Vector Search Parameters**
   ```python
   # Optimal search configuration
   search_params = {
       "limit": 10,  # Initial retrieval count
       "with_payload": True,
       "with_vectors": False,  # Reduce bandwidth
       "score_threshold": 0.7,  # Filter low-relevance results
   }
   ```

2. **Connection Pool Optimization**
   ```python
   # Configure optimal connection pools
   qdrant_client = QdrantClient(
       url=QDRANT_URL,
       timeout=60,
       prefer_grpc=True,
       grpc_options={
           'grpc.keepalive_time_ms': 30000,
           'grpc.keepalive_timeout_ms': 5000,
           'grpc.keepalive_permit_without_calls': True,
       }
   )
   ```

## Troubleshooting Guide

### Common Issues and Solutions

#### High API Latency

**Symptoms:**
- API response times > 5 seconds
- Timeouts in client applications
- Queue buildup in load balancer

**Solutions:**
1. Scale API service horizontally
2. Optimize database queries
3. Implement request caching
4. Tune connection pools

#### Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors
- Pod restarts due to memory limits

**Solutions:**
1. Implement proper connection cleanup
2. Configure garbage collection tuning
3. Add memory limits and monitoring
4. Regular service restarts

---

# Security Guide

## Overview

This document outlines the security architecture, best practices, and compliance considerations for the Sentio RAG system. The security model follows defense-in-depth principles with multiple layers of protection across network, application, data, and infrastructure components.

## Security Architecture

### Defense-in-Depth Model

```
┌─────────────────────────────────────────────────────────────────┐
│                       Network Security Layer                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Firewall      │  │   Load Balancer │  │   VPN Gateway   │ │
│  │   Rules         │  │   with WAF      │  │   (Optional)    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Application Security Layer                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ API Gateway     │  │ Authentication  │  │ Authorization   │ │
│  │ Rate Limiting   │  │ & JWT Tokens    │  │ & RBAC          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                      Data Security Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Encryption      │  │ Data            │  │ Secure          │ │
│  │ at Rest/Transit │  │ Classification  │  │ Communications  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                  Infrastructure Security Layer                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Container       │  │ Secrets         │  │ Monitoring      │ │
│  │ Security        │  │ Management      │  │ & Alerting      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Network Security

### Firewall Configuration

#### Ingress Rules

| Port | Protocol | Source | Purpose | Security Level |
|------|----------|--------|---------|---------------|
| 443 | HTTPS | Internet | API Access | High |
| 80 | HTTP | Internet | Redirect to HTTPS | Medium |
| 8910 | HTTP | Internal | Direct API (Development) | Low |
| 6333 | TCP | Internal | Qdrant Database | High |

### TLS Configuration

#### Certificate Management

```yaml
# TLS Configuration
tls:
  version: "1.3"
  cipher_suites:
    - "TLS_AES_256_GCM_SHA384"
    - "TLS_CHACHA20_POLY1305_SHA256"
    - "TLS_AES_128_GCM_SHA256"
  
certificate_rotation:
  automatic: true
  validity_period: "90 days"
  renewal_threshold: "30 days"
  
hsts:
  enabled: true
  max_age: "31536000"
  include_subdomains: true
  preload: true
```

## Application Security

### Authentication and Authorization

#### API Security Model

```python
# JWT Token Configuration
JWT_CONFIG = {
    "algorithm": "RS256",
    "token_expiration": 3600,  # 1 hour
    "refresh_token_expiration": 86400,  # 24 hours
    "issuer": "Sentio-rag-system",
    "audience": "Sentio-api-clients"
}

# Role-Based Access Control
RBAC_PERMISSIONS = {
    "admin": ["read", "write", "manage", "monitor"],
    "operator": ["read", "write", "monitor"],
    "viewer": ["read", "monitor"],
    "service": ["read", "write"]
}
```

### Input Validation and Sanitization

#### Request Validation

```python
# Input Validation Schema
REQUEST_VALIDATION = {
    "question": {
        "type": "string",
        "min_length": 1,
        "max_length": 1000,
        "pattern": r"^[a-zA-Z0-9\s\-\.\,\?\!\(\)]+$",
        "sanitization": "strict"
    },
    "history": {
        "type": "array",
        "max_items": 10,
        "items": {
            "type": "object",
            "required": ["role", "content"],
            "properties": {
                "role": {"enum": ["user", "assistant"]},
                "content": {"type": "string", "max_length": 2000}
            }
        }
    }
}
```

### Rate Limiting and DDoS Protection

#### Rate Limiting Configuration

```yaml
rate_limiting:
  global:
    requests_per_second: 100
    burst_size: 200
    
  per_endpoint:
    "/chat":
      requests_per_minute: 20
      burst_size: 5
    "/health":
      requests_per_second: 10
      burst_size: 20
      
  per_client:
    requests_per_hour: 1000
    concurrent_requests: 5
```

## Data Security

### Data Classification

#### Classification Levels

| Level | Description | Examples | Security Controls |
|-------|-------------|----------|-------------------|
| Public | Information intended for public consumption | Marketing materials | Standard encryption |
| Internal | Information for internal use only | Process documentation | Access controls + encryption |
| Confidential | Sensitive business information | Customer data, analytics | Strong encryption + audit logging |
| Restricted | Highly sensitive information | Security keys, PII | Maximum security controls |

### Encryption

#### Encryption at Rest

```yaml
# Database Encryption
qdrant:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_rotation: "quarterly"
    backup_encryption: true

# File System Encryption
storage:
  encryption:
    type: "LUKS"
    cipher: "aes-xts-plain64"
    key_size: 256
    hash: "sha256"
```

#### Encryption in Transit

```yaml
# Service-to-Service Communication
internal_tls:
  enabled: true
  mutual_auth: true
  certificate_authority: "internal-ca"
  cipher_suites:
    - "ECDHE-ECDSA-AES256-GCM-SHA384"
    - "ECDHE-RSA-AES256-GCM-SHA384"

# External API Communication
external_apis:
  jina_ai:
    tls_version: "1.3"
    certificate_pinning: true
    timeout: 30
```

## Infrastructure Security

### Container Security

#### Container Image Security

```dockerfile
# Security-hardened Dockerfile
FROM python:3.11-slim as base

# Create non-root user
RUN groupadd --gid 1001 Sentio && \
    useradd --uid 1001 --gid Sentio --shell /bin/bash Sentio

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy application with proper ownership
COPY --chown=Sentio:Sentio . /app
WORKDIR /app

# Switch to non-root user
USER Sentio
```

### Secrets Management

#### Kubernetes Secrets

```yaml
# External Secrets Operator Configuration
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: Sentio-secret-store
spec:
  provider:
    vault:
      server: "https://vault.internal.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "Sentio-rag"
```

## Security Monitoring and Incident Response

### Security Event Logging

#### Log Categories

```yaml
security_logging:
  authentication:
    - login_attempts
    - failed_authentication
    - token_generation
    - token_validation
    
  authorization:
    - permission_checks
    - access_denials
    - privilege_escalation_attempts
    
  data_access:
    - document_queries
    - sensitive_data_access
    - bulk_operations
    
  system_events:
    - configuration_changes
    - service_starts_stops
    - error_conditions
```

### Vulnerability Management

#### Security Scanning Pipeline

```yaml
# Security scanning in CI/CD
security_scans:
  static_analysis:
    tools:
      - bandit      # Python security linter
      - safety      # Dependency vulnerability check
      - semgrep     # Code pattern analysis
    
  dependency_scanning:
    tools:
      - snyk        # Dependency vulnerability scanning
      - trivy       # Container image scanning
      - grype       # Binary vulnerability scanning
    
  dynamic_analysis:
    tools:
      - zap         # Web application security testing
      - nikto       # Web server scanner
```

## Compliance and Governance

### Regulatory Compliance

#### SOC 2 Type II Controls

| Control | Description | Implementation | Monitoring |
|---------|-------------|----------------|------------|
| CC6.1 | Logical Access Controls | RBAC, MFA | Access logs, reviews |
| CC6.2 | System Access Monitoring | SIEM, alerting | Real-time monitoring |
| CC6.3 | Access Revocation | Automated process | Weekly audit |
| CC7.1 | Data Security | Encryption, DLP | Continuous scanning |
| CC7.2 | Data Integrity | Checksums, versioning | Automated validation |

#### ISO 27001 Compliance

```yaml
iso27001_controls:
  A.9.1.1:  # Access control policy
    implementation: "RBAC with documented procedures"
    review_frequency: "annual"
    
  A.9.2.3:  # Management of privileged access rights
    implementation: "Just-in-time access with approval workflow"
    review_frequency: "quarterly"
    
  A.10.1.1: # Cryptographic controls policy
    implementation: "AES-256 encryption for data at rest and in transit"
    review_frequency: "annual"
```

## Security Checklist

### Pre-Deployment Security Checklist

- [ ] Security assessment completed
- [ ] Vulnerability scanning passed
- [ ] Penetration testing completed
- [ ] Security configurations reviewed
- [ ] Access controls validated
- [ ] Encryption verified
- [ ] Monitoring configured
- [ ] Incident response plan tested
- [ ] Compliance requirements met
- [ ] Security documentation updated

### Ongoing Security Operations

#### Daily Tasks
- [ ] Review security alerts
- [ ] Monitor authentication logs
- [ ] Check system health
- [ ] Validate backup integrity

#### Weekly Tasks
- [ ] Security scan results review
- [ ] Access rights audit
- [ ] Incident response drill
- [ ] Policy compliance check

#### Monthly Tasks
- [ ] Vulnerability assessment
- [ ] Security metrics review
- [ ] Training completion tracking
- [ ] Risk assessment update

#### Quarterly Tasks
- [ ] Penetration testing
- [ ] Business continuity testing
- [ ] Security awareness assessment
- [ ] Compliance audit preparation

---

*This technical documentation provides comprehensive coverage of the Sentio RAG system for enterprise deployment and operations. The documentation follows industry standards for accuracy, completeness, and operational readiness.*

## Choosing an Embedding Provider

Sentio RAG now supports only one embedding provider as part of the simplified core for PoC.

### Supported Provider

1. **Jina AI (cloud, default)**
   - High-quality multilingual embeddings
   - Requires an API key and Internet connection
   - Excellent Russian language support

### Configuring the Embedding Provider

Select the provider via environment variables in the `.env` file:

```bash
# Choose the embedding provider
EMBEDDING_PROVIDER=jina

# Settings for Jina AI
JINA_API_KEY=your_jina_api_key_here
```

### Advantages of the Provider

| Factor | Jina AI |
|--------|---------|
| Embedding Quality | Very high |
| Speed | High (network dependent) |
| Privacy | Data sent to the cloud |
| Resource Requirements | Minimal (cloud processing) |
| Language Support | 100+ languages |
