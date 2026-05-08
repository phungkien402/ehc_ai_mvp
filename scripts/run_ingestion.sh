#!/bin/bash
# Run the FAQ ingestion pipeline

set -e

# Load .env
[ -f .env ] || cp .env.example .env
export $(cat .env | grep -v '^#' | xargs)

cd "$(dirname "$0")/.."

# Check if Qdrant is running
echo "Checking Qdrant connection..."
if ! curl -f http://localhost:6333/collections > /dev/null 2>&1; then
    echo "Error: Qdrant is not running. Start the stack first with: ./scripts/dev_up.sh"
    exit 1
fi

# Check if Ollama is accessible
if [[ "${MODEL_PROVIDER:-ollama}" == "vllm" ]]; then
    echo "Checking vLLM embedding endpoint..."
    if ! curl -f "${VLLM_EMBEDDING_URL:-http://localhost:8001}/v1/models" > /dev/null 2>&1; then
        echo "Error: vLLM embedding endpoint is not accessible at ${VLLM_EMBEDDING_URL:-http://localhost:8001}"
        exit 1
    fi
else
    echo "Checking Ollama connection..."
    if ! curl -f "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        echo "Error: Ollama is not accessible at ${OLLAMA_BASE_URL}"
        exit 1
    fi
fi

# Run ingestion
cd pipelines/ingestion
if [[ -x "../../venv/bin/python3" ]]; then
    ../../venv/bin/python3 main.py "$@"
else
    python3 main.py "$@"
fi
