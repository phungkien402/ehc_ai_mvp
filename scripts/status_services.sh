#!/usr/bin/env bash
set -euo pipefail

# Quick status check for runtime services.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Process Status ==="
ps -ef | rg "apps/api_gateway/main.py|apps/telegram_bot/bot.py" | rg -v "rg " || true

echo
echo "=== API Health ==="
if curl -fsS --max-time 8 "http://127.0.0.1:8000/api/v1/health"; then
  echo
  echo "[ok] API health check passed"
else
  echo
  echo "[warn] API health check failed"
fi

echo
echo "=== Recent Errors (tail) ==="
if [[ -f logs/bot.log ]]; then
  tail -n 40 logs/bot.log | rg -i "error|critical|traceback" || true
else
  echo "logs/bot.log not found"
fi
