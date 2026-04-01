#!/usr/bin/env bash
set -euo pipefail

# Decommission Ollama path after full vLLM cutover:
# 1) archive logs
# 2) stop local Ollama SSH tunnel
# 3) ensure API runs in MODEL_PROVIDER=vllm mode

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="logs/archive"
ARCHIVE_FILE="${ARCHIVE_DIR}/ollama-decommission-${TS}.tar.gz"

LOCAL_OLLAMA_PORT="${EHC_LOCAL_OLLAMA_PORT:-11435}"
VLLM_LLM_URL="${VLLM_LLM_URL:-http://127.0.0.1:18000}"
VLLM_EMBEDDING_URL="${VLLM_EMBEDDING_URL:-http://127.0.0.1:18001}"

mkdir -p "$ARCHIVE_DIR"

echo "Archiving operational logs -> ${ARCHIVE_FILE}"
# Archive available logs without failing if some files are missing.
tar -czf "$ARCHIVE_FILE" \
  --ignore-failed-read \
  logs/api-rollout.log logs/api_gateway.out logs/api.log logs/api.log.1 logs/api.log.2 logs/api.log.3 \
  logs/agent.log logs/agent.log.1 logs/agent.log.2 logs/agent.log.3 \
  logs/bot.log logs/telegram_bot.out

echo "Stopping local Ollama tunnel on port ${LOCAL_OLLAMA_PORT} (if running)..."
# Kill common ssh tunnel patterns forwarding local 11435 or 11434 to remote ollama port.
pkill -f "ssh.*-L ${LOCAL_OLLAMA_PORT}:localhost:11434" || true
pkill -f "ssh.*-L ${LOCAL_OLLAMA_PORT}:127.0.0.1:11434" || true
pkill -f "ssh.*-L 11434:localhost:11434" || true
pkill -f "ssh.*-L 11435:localhost:11434" || true
sleep 1

echo "Starting API in full vLLM mode..."
MODEL_PROVIDER=vllm ROLLOUT_ENABLED=false ROLLOUT_PERCENT_VLLM=0 \
  VLLM_LLM_URL="$VLLM_LLM_URL" VLLM_EMBEDDING_URL="$VLLM_EMBEDDING_URL" \
  ./scripts/start_canary_rollout.sh

echo "Verifying vLLM endpoints + API health..."
curl -fsS --max-time 8 "${VLLM_LLM_URL}/v1/models" >/dev/null
curl -fsS --max-time 8 "${VLLM_EMBEDDING_URL}/v1/models" >/dev/null
curl -fsS --max-time 10 "http://127.0.0.1:8000/api/v1/health"

echo "Decommission complete."
echo "Archive: ${ARCHIVE_FILE}"
echo "Rollback command: ./scripts/rollback_to_ollama.sh"
