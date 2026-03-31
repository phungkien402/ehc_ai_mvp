#!/bin/bash
# Run the FAQ ingestion pipeline

set -e

# Load .env
[ -f .env ] || cp .env.example .env
export $(cat .env | grep -v '^#' | xargs)

cd "$(dirname "$0")/.."

# Check if Qdrant is running
echo "Checking Qdrant connection..."
if ! curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "Error: Qdrant is not running. Start the stack first with: ./scripts/dev_up.sh"
    exit 1
fi

# Check if Ollama is accessible
echo "Checking Ollama connection..."
if ! curl -f "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo "Error: Ollama is not accessible at ${OLLAMA_BASE_URL}"
    exit 1
fi

# Run ingestion
cd pipelines/ingestion
python main.py "$@"
