#!/usr/bin/env bash
set -euo pipefail

# One-command rollback from full vLLM mode back to Ollama path.
# Steps:
# 1) start/recover Ollama tunnel
# 2) restart API in ollama mode with canary disabled
# 3) verify API health and provider routing

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Defaults for current infra; override via env when needed.
export EHC_GPU_HOST="${EHC_GPU_HOST:-n2.ckey.vn}"
export EHC_GPU_SSH_PORT="${EHC_GPU_SSH_PORT:-1234}"
export EHC_GPU_USER="${EHC_GPU_USER:-root}"
export EHC_GPU_OLLAMA_PORT="${EHC_GPU_OLLAMA_PORT:-11434}"
export EHC_LOCAL_OLLAMA_PORT="${EHC_LOCAL_OLLAMA_PORT:-11435}"

echo "Ensuring Ollama tunnel is up..."
./scripts/start_ollama_tunnel.sh

echo "Restarting API in ollama mode (canary disabled)..."
MODEL_PROVIDER=ollama \
ROLLOUT_ENABLED=false \
ROLLOUT_PERCENT_VLLM=0 \
OLLAMA_BASE_URL="http://127.0.0.1:${EHC_LOCAL_OLLAMA_PORT}" \
./scripts/rollback_canary.sh

echo "Verifying API and routing..."
python3 scripts/verify_canary_routing.py --runs 4 --rollout-percent 0 --base-provider ollama
curl -fsS --max-time 10 "http://127.0.0.1:8000/api/v1/health"

echo "Rollback to Ollama complete."
