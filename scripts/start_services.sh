#!/usr/bin/env bash
set -euo pipefail

# Unified runtime starter for API + Telegram bot.
# Modes:
# - vllm   : full vLLM (no rollout)
# - canary : ollama base + vLLM canary rollout
# - ollama : full ollama (no rollout)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-${MODE:-vllm}}"
BOT_CMD="${BOT_CMD:-venv/bin/python3 apps/telegram_bot/bot.py}"
BOT_LOG_FILE="${BOT_LOG_FILE:-logs/telegram_bot.out}"

mkdir -p logs tmp

case "$MODE" in
  vllm)
    echo "[start] API mode: vLLM full"
    MODEL_PROVIDER=vllm ROLLOUT_ENABLED=false ROLLOUT_PERCENT_VLLM=0 \
      ./scripts/start_canary_rollout.sh
    ;;
  canary)
    echo "[start] API mode: canary"
    ./scripts/start_canary_rollout.sh
    ;;
  ollama)
    echo "[start] API mode: Ollama full"
    MODEL_PROVIDER=ollama ROLLOUT_ENABLED=false ROLLOUT_PERCENT_VLLM=0 \
      ./scripts/rollback_canary.sh
    ;;
  *)
    echo "Usage: $0 [vllm|canary|ollama]"
    exit 2
    ;;
esac

if pgrep -f "apps/telegram_bot/bot.py" >/dev/null 2>&1; then
  echo "[start] Stopping existing Telegram bot..."
  pkill -f "apps/telegram_bot/bot.py"
  sleep 1
fi

echo "[start] Starting Telegram bot..."
nohup bash -lc "$BOT_CMD" >"$BOT_LOG_FILE" 2>&1 < /dev/null &
BOT_PID=$!
echo "$BOT_PID" > tmp/telegram_bot.pid

echo "[ok] API + bot started"
echo "[ok] Bot PID: $BOT_PID"
echo "[ok] Health: http://127.0.0.1:8000/api/v1/health"
