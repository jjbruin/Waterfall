# Multi-stage build: Vue frontend + Flask API

# ── Stage 1: Build Vue frontend ──────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /build
COPY vue_app/package.json vue_app/package-lock.json* ./
RUN npm ci
COPY vue_app/ ./
RUN npm run build

# ── Stage 2: Python API + serve static files ─────────────────────────
FROM python:3.12-slim
WORKDIR /app

# System deps for psycopg2 and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY flask_app/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
# Root-level modules (waterfall engine, compute, database, etc.)
COPY *.py ./
COPY *.txt ./

# Flask app package
COPY flask_app/ flask_app/

# Vue built assets
COPY --from=frontend-build /build/dist/ static/

# Production config
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Gunicorn config
ENV GUNICORN_WORKERS=4
ENV GUNICORN_TIMEOUT=300
ENV PORT=8000

EXPOSE 8000

CMD gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers ${GUNICORN_WORKERS} \
    --timeout ${GUNICORN_TIMEOUT} \
    --access-logfile - \
    --error-logfile - \
    "flask_app:create_app()"
