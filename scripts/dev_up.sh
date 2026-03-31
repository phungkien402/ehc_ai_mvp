#!/bin/bash
# Bootstrap script to start the EHC AI MVP stack

set -e

echo "=== Starting EHC AI MVP Stack ==="
cd "$(dirname "$0")"

# Load .env
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

export $(cat .env | grep -v '^#' | xargs)

# Start Docker services
echo "Starting Docker services..."
docker-compose -f docker/docker-compose.yml up -d

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
