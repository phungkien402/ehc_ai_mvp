#!/usr/bin/env bash
set -euo pipefail

# Unified runtime stopper for API + Telegram bot.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[stop] Stopping Telegram bot..."
pkill -f "apps/telegram_bot/bot.py" || true

echo "[stop] Stopping API gateway..."
pkill -f "apps/api_gateway/main.py" || true

sleep 1

echo "[ok] Services stopped"
