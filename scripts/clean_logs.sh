#!/usr/bin/env bash
set -euo pipefail

# Archive and clean operational logs.
# Default retention keeps latest 7 archives.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ARCHIVE_DIR="logs/archive"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_FILE="${ARCHIVE_DIR}/ops-logs-${TS}.tar.gz"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-7}"

mkdir -p "$ARCHIVE_DIR" logs

# Collect logs that usually exist; ignore missing files.
tar -czf "$ARCHIVE_FILE" \
  --ignore-failed-read \
  logs/api-rollout.log logs/api_gateway.out logs/api.log logs/api.log.1 logs/api.log.2 logs/api.log.3 \
  logs/agent.log logs/agent.log.1 logs/agent.log.2 logs/agent.log.3 \
  logs/bot.log logs/telegram_bot.out

# Truncate active logs in place so running processes keep file handles.
for f in logs/api-rollout.log logs/api_gateway.out logs/api.log logs/agent.log logs/bot.log logs/telegram_bot.out; do
  if [[ -f "$f" ]]; then
    : > "$f"
  fi
done

# Keep only latest N archives.
ls -1t "$ARCHIVE_DIR"/ops-logs-*.tar.gz 2>/dev/null | tail -n +$((KEEP_ARCHIVES + 1)) | xargs -r rm -f

echo "[ok] Archived logs: $ARCHIVE_FILE"
echo "[ok] Active logs truncated"
