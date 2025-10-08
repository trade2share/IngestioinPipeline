# syntax=docker/dockerfile:1.7

# Fresh, minimal runtime pinned to python:3.11-slim
FROM --platform=$TARGETPLATFORM python:3.11-slim AS base

ARG TARGETPLATFORM
ARG BUILDPLATFORM

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# Only the runtime libs actually needed by current code (PyPDFium2 works without poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Leverage caching by installing requirements before copying full source
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code including Chainlit public assets
COPY . .

# Healthcheck for quick feedback in container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', int(__import__('os').environ.get('PORT','8000')))); s.close()" || exit 1

# Expose Chainlit default port
EXPOSE 8000

# Ensure Chainlit serves static assets from /app/public (default in 1.3)
ENV CHAINLIT_PUBLIC_DIR="/app/public"

# Start Chainlit. It will automatically pick up branding from /app/public/branding.css
CMD ["sh", "-c", "chainlit run chainlit_app.py --host 0.0.0.0 --port ${PORT:-8000}"]


