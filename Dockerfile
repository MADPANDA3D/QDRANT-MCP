FROM python:3.11-slim

WORKDIR /app

# System deps for document extraction (PDF OCR + .doc parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    antiword \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install this package from source
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
RUN uv pip install --system --no-cache-dir .

# Expose the default port for SSE transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV QDRANT_VECTOR_NAME=""
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Run the server with SSE transport
CMD uvx mcp-server-qdrant --transport sse
