#!/bin/bash
set -e

echo "=== Noesis AI - Starting ==="

mkdir -p /data/data_rooms /data/chroma_db /data/logs

echo "Initializing database..."
python tools/init_database.py

echo "Starting uvicorn on ${HOST:-0.0.0.0}:${PORT:-8000}..."
exec uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --log-level info \
    --workers 1
