# ============================================================
# Stage 1: Build frontend
# ============================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

COPY frontend/ ./

# Empty VITE_API_URL so API calls use same-origin
ENV VITE_API_URL=""
RUN npm run build


# ============================================================
# Stage 2: Python backend + serve built frontend
# ============================================================
FROM python:3.11-slim

# System dependencies for document processing and building C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools (needed for chromadb, weasyprint, etc.)
    build-essential \
    gcc \
    g++ \
    # PDF processing
    poppler-utils \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # WeasyPrint dependencies
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    libcairo2 \
    libcairo2-dev \
    libglib2.0-0 \
    libglib2.0-dev \
    shared-mime-info \
    # General utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the local embedding model so first request has no cold start
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" 2>/dev/null || true

# Copy application code
COPY app/ ./app/
COPY tools/ ./tools/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Create data directories
RUN mkdir -p /data/data_rooms /data/chroma_db /data/logs

COPY start.sh .
RUN chmod +x start.sh

ENV PORT=8000
ENV HOST=0.0.0.0
ENV SERVE_FRONTEND=true
# DATABASE_URL is passed at runtime via docker run -e (e.g. postgresql://user:pass@host:5432/dbname)
ENV CHROMA_DB_PATH=/data/chroma_db
ENV DATA_ROOMS_PATH=/data/data_rooms

EXPOSE ${PORT}

CMD ["./start.sh"]
