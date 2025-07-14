FROM python:3.11.5 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Create virtual environment and install dependencies
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

# Production stage
FROM python:3.11.5-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv .venv/

# Copy application code
COPY . .

RUN test -f backend/__init__.py || echo "" > backend/__init__.py

# Run the application
# shell form so $PORT is picked up
CMD .venv/bin/uvicorn backend.api_server:app \
     --host 0.0.0.0 \
     --port $PORT