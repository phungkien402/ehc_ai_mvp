# Daily Ops Runbook (EHC AI Helpdesk)

This checklist is designed for day-to-day operation with the current vLLM-first setup.

## 1) Start of Day

1. Go to project root.
2. Ensure tunnels to vLLM are up (`18000` for LLM, `18001` for embedding/vision).
3. Start services in full vLLM mode.
4. Verify health and process status.

```bash
cd /home/phungkien/ehc_ai_mvp
./scripts/start_services.sh vllm
./scripts/status_services.sh
```

Expected:
- API health returns `status=ok`
- One API process (`apps/api_gateway/main.py`)
- One Telegram bot process (`apps/telegram_bot/bot.py`)

## 2) Health Check (Any Time)

1. Check API health endpoint.
2. Check recent errors in bot log.
3. Spot-check one real query in Telegram.

```bash
curl -sS http://127.0.0.1:8000/api/v1/health
rg -n "ERROR|CRITICAL|Traceback" logs/bot.log | tail -n 20
```

## 3) Canary Operation (If Needed)

1. Start canary mode (default 10%).
2. Verify deterministic sticky routing.
3. Increase rollout gradually only when metrics are stable.

```bash
./scripts/start_services.sh canary
python3 scripts/verify_canary_routing.py --runs 10 --rollout-percent 10
```

Common progression:
- `10 -> 25 -> 50 -> 75 -> 100`

## 4) Rollback

### A. Rollback canary to 0% (still Ollama base)

```bash
./scripts/rollback_canary.sh
python3 scripts/verify_canary_routing.py --runs 6 --rollout-percent 0 --base-provider ollama
```

### B. Full rollback from vLLM to Ollama path

```bash
./scripts/rollback_to_ollama.sh
```

## 5) Log Maintenance

1. Archive + truncate active logs.
2. Keep latest 7 archives by default.

```bash
./scripts/clean_logs.sh
```

Archives are stored in `logs/archive/`.

## 6) End of Day

1. Re-check service status.
2. If maintenance window: stop services cleanly.

```bash
./scripts/status_services.sh
./scripts/stop_services.sh
```

## 7) Quick Incident Playbook

1. API unhealthy (`/api/v1/health` degraded):
- Check `logs/api-rollout.log`
- Verify vLLM endpoints (`/v1/models` on `18000`, `18001`)
- Restart with `./scripts/start_services.sh vllm`

2. Telegram bot conflict (`terminated by other getUpdates request`):
- Run `./scripts/stop_services.sh`
- Start again with `./scripts/start_services.sh vllm`

3. Image attachment slow/fail from Redmine:
- Confirm `REDMINE_API_KEY` and `REDMINE_URL` in `.env`
- Check `logs/bot.log` for attachment errors

## 8) Standard Commands Summary

```bash
# Start (full vLLM)
./scripts/start_services.sh vllm

# Start (canary)
./scripts/start_services.sh canary

# Start (full Ollama)
./scripts/start_services.sh ollama

# Status
./scripts/status_services.sh

# Stop
./scripts/stop_services.sh

# Clean logs
./scripts/clean_logs.sh
```
