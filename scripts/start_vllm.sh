#!/bin/bash
# Start vLLM servers for 2x Tesla V100 16GB
#
# GPU 0+1 → LLM: Qwen2.5-14B-Instruct (tensor_parallel=2, ~28GB)
# GPU 1   → Embeddings: BAAI/bge-m3 (~560MB, runs after LLM claims both GPUs)
#
# NOTE: V100 (sm_70) does not support FlashAttention v2 — vLLM auto-falls back.
# If OOM: switch to Qwen2.5-7B-Instruct with tensor_parallel=1 on GPU_LLM=0
#
# Usage:
#   bash scripts/start_vllm.sh              # LLM (14B) + embeddings
#   bash scripts/start_vllm.sh --llm-only
#   bash scripts/start_vllm.sh --7b         # 7B model on 1 GPU
#   bash scripts/start_vllm.sh --qwen3 4b   # Qwen3-4B with reasoning-parser (1 GPU)
#   bash scripts/start_vllm.sh --qwen3 8b   # Qwen3-8B with reasoning-parser (1 GPU)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLM_PORT=8080
EMBED_PORT=8081
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

USE_7B=false
LLM_ONLY=false
USE_QWEN3=""
for arg in "$@"; do
  case $arg in
    --7b)       USE_7B=true ;;
    --llm-only) LLM_ONLY=true ;;
    --qwen3)    USE_QWEN3="next" ;;
    *)
      if [ "$USE_QWEN3" = "next" ]; then
        USE_QWEN3="$arg"
      fi
      ;;
  esac
done

# ── Check vLLM installed ──────────────────────────────────────────────────────
if ! python3 -c "import vllm" 2>/dev/null; then
  echo "[vllm] vLLM not installed. Installing..."
  pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
fi

# ── Kill any existing vLLM on those ports ─────────────────────────────────────
for port in $LLM_PORT $EMBED_PORT; do
  pid=$(lsof -ti:$port 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "[vllm] Killing existing process on port $port (pid $pid)"
    kill -9 $pid 2>/dev/null || true
    sleep 1
  fi
done

# ── Start LLM server ─────────────────────────────────────────────────────────
REASONING_FLAGS=""
if [ -n "$USE_QWEN3" ] && [ "$USE_QWEN3" != "next" ]; then
  SIZE="${USE_QWEN3:-4b}"
  LLM_MODEL="Qwen/Qwen3-${SIZE^^}"
  TENSOR_PARALLEL=1
  GPU_MEM_UTIL=0.88
  GPU_IDS="0"
  # --reasoning-parser separates <think> blocks into reasoning_content field
  REASONING_FLAGS="--reasoning-parser deepseek_r1"
  echo "[vllm] Qwen3 mode: thinking will be separated into reasoning_content"
elif [ "$USE_7B" = true ]; then
  LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
  TENSOR_PARALLEL=1
  GPU_MEM_UTIL=0.88
  GPU_IDS="0"
else
  LLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
  TENSOR_PARALLEL=2
  GPU_MEM_UTIL=0.90
  GPU_IDS="0,1"
fi

echo "[vllm] Starting LLM: $LLM_MODEL on GPU(s) $GPU_IDS → port $LLM_PORT"
CUDA_VISIBLE_DEVICES=$GPU_IDS nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$LLM_MODEL" \
  --port $LLM_PORT \
  --host 0.0.0.0 \
  --tensor-parallel-size $TENSOR_PARALLEL \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --dtype float16 \
  --max-model-len 8192 \
  --enforce-eager \
  --disable-log-requests \
  $REASONING_FLAGS \
  > "$LOG_DIR/vllm_llm.log" 2>&1 &

LLM_PID=$!
echo "  PID $LLM_PID → logs: $LOG_DIR/vllm_llm.log"

# ── Wait for LLM to be ready ─────────────────────────────────────────────────
echo "[vllm] Waiting for LLM server on port $LLM_PORT..."
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$LLM_PORT/health" > /dev/null 2>&1; then
    echo "  ✓ LLM ready"
    break
  fi
  if ! kill -0 $LLM_PID 2>/dev/null; then
    echo "  ✗ LLM process died. Check: tail -50 $LOG_DIR/vllm_llm.log"
    exit 1
  fi
  echo "  ... ($i/60)"
  sleep 5
done

if [ "$LLM_ONLY" = true ]; then
  echo "[vllm] --llm-only: skipping embedding server"
  echo ""
  echo "✓ vLLM LLM server running on http://localhost:$LLM_PORT/v1"
  exit 0
fi

# ── Start embedding server (GPU1 if 14B, GPU0 if 7B) ─────────────────────────
if [ "$USE_7B" = true ]; then
  EMBED_GPU="1"
else
  # 14B uses both GPUs; embeddings must share GPU1 (tiny model)
  EMBED_GPU="1"
fi

echo "[vllm] Starting embeddings: BAAI/bge-m3 on GPU $EMBED_GPU → port $EMBED_PORT"
CUDA_VISIBLE_DEVICES=$EMBED_GPU nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "BAAI/bge-m3" \
  --port $EMBED_PORT \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.20 \
  --dtype float16 \
  --task embed \
  --enforce-eager \
  --disable-log-requests \
  > "$LOG_DIR/vllm_embed.log" 2>&1 &

EMBED_PID=$!
echo "  PID $EMBED_PID → logs: $LOG_DIR/vllm_embed.log"

# ── Wait for embedding server ─────────────────────────────────────────────────
echo "[vllm] Waiting for embedding server on port $EMBED_PORT..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:$EMBED_PORT/health" > /dev/null 2>&1; then
    echo "  ✓ Embeddings ready"
    break
  fi
  if ! kill -0 $EMBED_PID 2>/dev/null; then
    echo "  ✗ Embedding process died. Check: tail -50 $LOG_DIR/vllm_embed.log"
    exit 1
  fi
  echo "  ... ($i/30)"
  sleep 3
done

echo ""
echo "================================================"
echo "  ✓ vLLM running"
echo "  LLM:    http://localhost:$LLM_PORT/v1  (model: $LLM_MODEL)"
echo "  Embed:  http://localhost:$EMBED_PORT/v1  (model: BAAI/bge-m3)"
echo "  Logs:   $LOG_DIR/vllm_llm.log"
echo "          $LOG_DIR/vllm_embed.log"
echo "================================================"
