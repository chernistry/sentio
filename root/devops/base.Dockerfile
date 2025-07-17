# syntax=docker/dockerfile:1.4

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies with optimized caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev && \
    apt-get clean && \
    apt-get purge -y gcc python3-dev && \
    apt-get autoremove -y

# Install pip tools
RUN pip install --no-cache-dir pip-tools wheel

# Copy requirements and install dependencies with cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip-compile requirements.txt --output-file requirements.compiled.txt && \
    pip install --no-cache-dir --user -r requirements.compiled.txt

# Add common Python packages including LangGraph
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --user \
    click \
    python-dotenv \
    uvicorn \
    langgraph>=0.5.3 \
    langgraph-api>=0.2.86 \
    aiofiles>=23.2.1

# Final base image with just the installed packages
FROM python:3.11-slim AS base

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH 