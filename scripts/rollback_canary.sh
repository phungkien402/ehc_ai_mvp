#!/usr/bin/env bash
set -euo pipefail

# Rollback canary traffic to 0% and restart API gateway in native process mode.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

API_CMD="${API_CMD:-venv/bin/python3 apps/api_gateway/main.py}"
API_LOG_FILE="${API_LOG_FILE:-logs/api-rollout.log}"

export MODEL_PROVIDER="${MODEL_PROVIDER:-ollama}"
export ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-false}"
export ROLLOUT_PERCENT_VLLM="${ROLLOUT_PERCENT_VLLM:-0}"
export ROLLOUT_STICKY_KEY="${ROLLOUT_STICKY_KEY:-session_id}"

export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11435}"

mkdir -p logs tmp

if pgrep -f "apps/api_gateway/main.py" >/dev/null 2>&1; then
  echo "Stopping existing API gateway process..."
  pkill -f "apps/api_gateway/main.py"
  sleep 1
fi

echo "Starting API gateway with canary disabled..."
nohup bash -lc "$API_CMD" >"$API_LOG_FILE" 2>&1 < /dev/null &
API_PID=$!
echo "$API_PID" > tmp/api_gateway.pid

for i in $(seq 1 20); do
  if curl -fsS --max-time 5 "http://127.0.0.1:8000/api/v1/health" >/dev/null 2>&1; then
    echo "Rollback complete. pid=${API_PID}"
    exit 0
  fi
  sleep 1
done

echo "API did not become ready after rollback. Check: ${API_LOG_FILE}"
exit 1
