#!/usr/bin/env bash
set -euo pipefail

# Start API gateway with deterministic canary routing to vLLM.
# Intended for native process mode (not docker compose).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

API_CMD="${API_CMD:-venv/bin/python3 apps/api_gateway/main.py}"
API_PID_FILE="${API_PID_FILE:-tmp/api_gateway.pid}"
API_LOG_FILE="${API_LOG_FILE:-logs/api-rollout.log}"

export MODEL_PROVIDER="${MODEL_PROVIDER:-ollama}"
export ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-true}"
export ROLLOUT_PERCENT_VLLM="${ROLLOUT_PERCENT_VLLM:-10}"
export ROLLOUT_STICKY_KEY="${ROLLOUT_STICKY_KEY:-session_id}"

if [[ -z "${OLLAMA_LLM_MODEL:-}" ]]; then
  if [[ "$MODEL_PROVIDER" == "vllm" ]]; then
    export OLLAMA_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
  else
    export OLLAMA_LLM_MODEL="qwen2.5:14b"
  fi
fi

if [[ -z "${OLLAMA_GRADER_MODEL:-}" ]]; then
  export OLLAMA_GRADER_MODEL="$OLLAMA_LLM_MODEL"
fi

if [[ -z "${OLLAMA_REWRITE_MODEL:-}" ]]; then
  export OLLAMA_REWRITE_MODEL="$OLLAMA_LLM_MODEL"
fi

if [[ -z "${OLLAMA_EMBEDDING_MODEL:-}" ]]; then
  if [[ "$MODEL_PROVIDER" == "vllm" ]]; then
    export OLLAMA_EMBEDDING_MODEL="BAAI/bge-m3"
  else
    export OLLAMA_EMBEDDING_MODEL="bge-m3:latest"
  fi
fi

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:18000}"
export VLLM_LLM_URL="${VLLM_LLM_URL:-http://127.0.0.1:18000}"
export VLLM_EMBEDDING_URL="${VLLM_EMBEDDING_URL:-http://127.0.0.1:18001}"
export VLLM_VISION_URL="${VLLM_VISION_URL:-http://127.0.0.1:18001}"

mkdir -p logs tmp

if pgrep -f "apps/api_gateway/main.py" >/dev/null 2>&1; then
  echo "Stopping existing API gateway process..."
  pkill -f "apps/api_gateway/main.py"
  sleep 1
fi

echo "Preflight: checking vLLM endpoints..."
curl -fsS --max-time 8 "${VLLM_LLM_URL}/v1/models" >/dev/null
echo "  - LLM endpoint OK: ${VLLM_LLM_URL}"
curl -fsS --max-time 8 "${VLLM_EMBEDDING_URL}/v1/models" >/dev/null
echo "  - Embedding endpoint OK: ${VLLM_EMBEDDING_URL}"

echo "Starting API gateway with canary rollout=${ROLLOUT_PERCENT_VLLM}% ..."
nohup bash -lc "$API_CMD" >"$API_LOG_FILE" 2>&1 < /dev/null &
API_PID=$!
echo "$API_PID" > "$API_PID_FILE"

for i in $(seq 1 20); do
  if curl -fsS --max-time 5 "http://127.0.0.1:8000/api/v1/health" >/dev/null 2>&1; then
    echo "API is ready. pid=${API_PID}"
    echo "Log file: ${API_LOG_FILE}"
    echo "Next: python3 scripts/verify_canary_routing.py --runs 10"
    exit 0
  fi
  sleep 1
done

echo "API did not become ready in time. Check: ${API_LOG_FILE}"
exit 1
