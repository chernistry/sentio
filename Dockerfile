FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    QDRANT_URL="" \
    QDRANT_API_KEY=""

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1 \
    && poetry config virtualenvs.create false

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-dev

# Expose the API port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 