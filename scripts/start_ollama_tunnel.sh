#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${EHC_GPU_HOST:-n2.ckey.vn}"
SSH_PORT="${EHC_GPU_SSH_PORT:-1848}"
SSH_USER="${EHC_GPU_USER:-root}"
REMOTE_PORT="${EHC_GPU_OLLAMA_PORT:-11434}"
LOCAL_PORT="${EHC_LOCAL_OLLAMA_PORT:-11434}"

if curl -fsS "http://127.0.0.1:${LOCAL_PORT}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is already reachable at http://127.0.0.1:${LOCAL_PORT}"
    exit 0
fi

ssh \
    -fN \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    -p "${SSH_PORT}" \
    "${SSH_USER}@${SSH_HOST}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3

curl -fsS "http://127.0.0.1:${LOCAL_PORT}/api/tags" >/dev/null
echo "Ollama tunnel ready: http://127.0.0.1:${LOCAL_PORT} -> ${SSH_USER}@${SSH_HOST}:${REMOTE_PORT}"
