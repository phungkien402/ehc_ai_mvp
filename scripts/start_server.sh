#!/bin/bash
# EHC AI MVP — GPU Server Startup Script
# Usage: bash scripts/start_server.sh [--skip-deps] [--skip-qdrant]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SKIP_DEPS=false
SKIP_QDRANT=false
for arg in "$@"; do
  case $arg in
    --skip-deps) SKIP_DEPS=true ;;
    --skip-qdrant) SKIP_QDRANT=true ;;
  esac
done

echo "================================================"
echo "  EHC AI MVP — Server Startup"
echo "  Root: $PROJECT_ROOT"
echo "================================================"

# ── 1. Create .env if missing ────────────────────────────────────────────────
if [ ! -f .env ]; then
  echo "[1/5] Creating .env from template..."
  cat > .env << 'EOF'
# Redmine (fill in when available)
REDMINE_URL=http://localhost:3000
REDMINE_API_KEY=

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_DOCS_COLLECTION=ehc_module_docs

# Ollama — running on remote GPU server
OLLAMA_BASE_URL=http://n3.ckey.vn:11434
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_GRADER_MODEL=qwen3.5:9b
OLLAMA_REWRITE_MODEL=qwen3.5:9b
OLLAMA_VISION_MODEL=qwen3.5:9b

# Model provider
MODEL_PROVIDER=ollama
ROLLOUT_ENABLED=false
ROLLOUT_PERCENT_VLLM=0
ROLLOUT_STICKY_KEY=session_id

# Multi-source retrieval
DOCS_RETRIEVAL_ENABLED=true
DOCS_RETRIEVAL_TOP_K=4

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ingestion
CHUNK_SIZE=500
CHUNK_OVERLAP=100
BATCH_SIZE=10
DOCX_INPUT_DIR=data/module_docs_raw
DOCX_OCR_ENABLED=true
DOCX_OCR_MAX_IMAGES=6
DOCX_OCR_BACKEND=auto
EOF
  echo "  ✓ .env created"
else
  echo "[1/5] .env exists — skipping"
fi

# ── 2. Required directories ──────────────────────────────────────────────────
echo "[2/5] Creating required directories..."
mkdir -p logs data/module_docs_raw data/docx_images
echo "  ✓ logs/, data/module_docs_raw/, data/docx_images/"

# ── 3. Start Qdrant via Docker ───────────────────────────────────────────────
if [ "$SKIP_QDRANT" = false ]; then
  echo "[3/5] Starting Qdrant..."
  if ! command -v docker &>/dev/null; then
    echo "  ✗ Docker not found — install Docker first (curl -fsSL https://get.docker.com | sh)"
    exit 1
  fi

  if docker ps --format '{{.Names}}' | grep -q '^ehc_qdrant$'; then
    echo "  ✓ Qdrant already running"
  else
    docker run -d \
      --name ehc_qdrant \
      --restart unless-stopped \
      -p 6333:6333 \
      -v ehc_qdrant_data:/qdrant/storage \
      qdrant/qdrant:latest
    echo "  ✓ Qdrant started"
    echo "  Waiting 10s for Qdrant to be ready..."
    sleep 10
  fi

  if curl -sf http://localhost:6333/health > /dev/null; then
    echo "  ✓ Qdrant healthy at http://localhost:6333"
  else
    echo "  ✗ Qdrant not responding — check: docker logs ehc_qdrant"
    exit 1
  fi
else
  echo "[3/5] Skipping Qdrant (--skip-qdrant)"
fi

# ── 4. Python venv + dependencies ────────────────────────────────────────────
if [ "$SKIP_DEPS" = false ]; then
  echo "[4/5] Setting up Python venv..."

  if [ ! -d venv ]; then
    python3 -m venv venv
    echo "  ✓ venv created"
  else
    echo "  ✓ venv exists"
  fi

  source venv/bin/activate

  pip install --quiet --upgrade pip

  echo "  Installing shared dependencies..."
  pip install --quiet -r shared/py/requirements.txt

  echo "  Installing API gateway dependencies..."
  pip install --quiet fastapi==0.104.1 "uvicorn[standard]==0.24.0" pydantic-settings python-multipart

  echo "  Installing agent runtime dependencies..."
  pip install --quiet \
    "langgraph>=0.1.0" \
    "langchain>=0.2.0" \
    "langchain-core>=0.2.0" \
    "langchain-ollama>=0.1.0" \
    "langchain-openai>=0.1.0" \
    "langchain-text-splitters>=0.2.0"

  echo "  Installing ingestion dependencies..."
  pip install --quiet \
    python-docx \
    Pillow \
    pytesseract

  echo "  ✓ All dependencies installed"
else
  echo "[4/5] Skipping deps (--skip-deps)"
  source venv/bin/activate
fi

# ── 5. Verify Ollama models ──────────────────────────────────────────────────
echo "[5/5] Checking Ollama models..."
source .env 2>/dev/null || true

OLLAMA_OK=true
for model in "${OLLAMA_LLM_MODEL:-qwen3.5:latest}" "${OLLAMA_EMBEDDING_MODEL:-bge-m3:latest}"; do
  if curl -sf http://localhost:11434/api/tags | grep -q "\"${model}\""; then
    echo "  ✓ $model"
  else
    echo "  ⚠ $model not found in Ollama — pull it: ollama pull $model"
    OLLAMA_OK=false
  fi
done

# ── Start API ────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Starting API Gateway on port 8000"
echo "================================================"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/apps/api_gateway:$PROJECT_ROOT/apps/agent_runtime"

cd "$PROJECT_ROOT/apps/api_gateway"

# Kill any previous instance on port 8000
if command -v fuser &>/dev/null; then
  fuser -k 8000/tcp 2>/dev/null || true
elif command -v lsof &>/dev/null; then
  lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi

echo "  Logs → $PROJECT_ROOT/logs/api.log"
echo "  API  → http://0.0.0.0:8000"
echo "  Docs → http://0.0.0.0:8000/docs"
echo ""

exec uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info
