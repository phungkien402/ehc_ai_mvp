#!/bin/bash
# Bootstrap script to start the EHC AI MVP stack

set -e

echo "=== Starting EHC AI MVP Stack ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    if [ ! -f .env.example ]; then
        echo "Error: .env.example not found in $PROJECT_ROOT"
        exit 1
    fi
    cp .env.example .env
fi

set -a
. ./.env
set +a

# Support both Compose V1 (`docker-compose`) and V2 (`docker compose`).
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
else
    echo "Error: neither 'docker-compose' nor 'docker compose' is available"
    exit 1
fi

# Start Docker services
echo "Starting Docker services..."
"${COMPOSE_CMD[@]}" -f docker/docker-compose.yml up -d

# Wait for services to be healthy
echo "Waiting for services to be ready (30 seconds)..."
sleep 30

# Health check
echo ""
echo "=== Health Check ==="
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "✓ Qdrant is ready"
else
    echo "✗ Qdrant is not ready - check Docker logs"
fi

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API Gateway is ready"
else
    echo "✗ API Gateway is not ready - check Docker logs"
fi

echo ""
echo "=== Stack is ready ==="
echo "API: http://localhost:8000"
echo "Qdrant: http://localhost:6333"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "To run ingestion:"
echo "  python pipelines/ingestion/main.py --project FAQ"
