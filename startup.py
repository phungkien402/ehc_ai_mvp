#!/usr/bin/env python3
"""
EHC AI MVP — One-command startup script.
Starts: SSH Ollama tunnel → Qdrant → API Gateway → Telegram Bot

Usage:
    python3 startup.py
"""

import os
import sys
import time
import hashlib
import signal
import socket
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ── Config ────────────────────────────────────────────────────────────────────
# Using local Ollama (GPU is now local)
OLLAMA_LOCAL = int(os.getenv("EHC_LOCAL_OLLAMA_PORT", "11434"))

PROCS: list[subprocess.Popen] = []

def _die(msg: str):
    print(f"\n✗ {msg}", file=sys.stderr)
    _shutdown(None, None)
    sys.exit(1)

def _shutdown(sig, frame):
    print("\n[shutdown] Stopping all processes...")
    for p in reversed(PROCS):
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try: p.kill()
            except Exception: pass
    print("[shutdown] Done.")
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _http_ok(url: str, timeout: int = 5) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False

def _wait_for(url: str, label: str, retries: int = 30, interval: float = 2.0):
    for i in range(retries):
        if _http_ok(url):
            print(f"  ✓ {label}")
            return
        print(f"  … waiting for {label} ({i+1}/{retries})")
        time.sleep(interval)
    _die(f"{label} did not become ready at {url}")

def _run_bg(*cmd, log_file: str | None = None, cwd=None, env=None) -> subprocess.Popen:
    out = open(ROOT / log_file, "a") if log_file else subprocess.DEVNULL
    p = subprocess.Popen(
        cmd, stdout=out, stderr=out,
        cwd=str(cwd or ROOT), env=env,
        start_new_session=True,
    )
    PROCS.append(p)
    return p

def _run(cmd: list[str], check=True, capture=False, cwd=None):
    return subprocess.run(
        cmd, check=check,
        capture_output=capture,
        cwd=str(cwd or ROOT),
    )


# ── Step 0: Write .env if missing ────────────────────────────────────────────
def setup_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        print("[env] .env exists — skipping creation")
        return

    print("[env] Creating .env ...")
    # Ask for bot token
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        token = input("  Enter your TELEGRAM_BOT_TOKEN: ").strip()
    if not token:
        _die("TELEGRAM_BOT_TOKEN is required for the Telegram bot")

    env_path.write_text(f"""\
# Redmine (fill when available)
REDMINE_URL=http://localhost:3000
REDMINE_API_KEY=

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_DOCS_COLLECTION=ehc_module_docs

# Ollama via SSH tunnel from GPU server
OLLAMA_BASE_URL=http://127.0.0.1:{OLLAMA_LOCAL}
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_GRADER_MODEL=qwen3.5:9b
OLLAMA_REWRITE_MODEL=qwen3.5:9b
OLLAMA_VISION_MODEL=qwen3.5:9b

# Model provider
MODEL_PROVIDER=ollama
ROLLOUT_ENABLED=false
ROLLOUT_PERCENT_VLLM=0
ROLLOUT_STICKY_KEY=session_id

# Multi-source retrieval
DOCS_RETRIEVAL_ENABLED=true
DOCS_RETRIEVAL_TOP_K=4

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ingestion
CHUNK_SIZE=500
CHUNK_OVERLAP=100
BATCH_SIZE=10
DOCX_INPUT_DIR=data/module_docs_raw
DOCX_OCR_ENABLED=true
DOCX_OCR_MAX_IMAGES=6
DOCX_OCR_BACKEND=auto

# Telegram
TELEGRAM_BOT_TOKEN={token}
HELPDESK_API_URL=http://localhost:8000/api/v1/ask
TELEGRAM_API_TIMEOUT_SECONDS=240
""")
    print("  ✓ .env created")


# ── Step 1: Create directories ────────────────────────────────────────────────
def setup_dirs():
    for d in ["logs", "tmp", "data/module_docs_raw", "data/docx_images"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("[dirs] ✓ logs/ data/ tmp/ ready")


# ── Step 2: Check local Ollama ───────────────────────────────────────────────────
def check_ollama_ready():
    print(f"[ollama] Checking local Ollama on localhost:{OLLAMA_LOCAL}...")

    if _port_open("127.0.0.1", OLLAMA_LOCAL):
        print(f"  ✓ Ollama is running on localhost:{OLLAMA_LOCAL}")
        return

    print(f"\n⚠️  Ollama is not running on port {OLLAMA_LOCAL}")
    print("\n  Start Ollama with:")
    print("  $ ollama serve")
    print("\n  Or run Ollama via Docker:")
    print("  $ docker run -d --gpus all -p 11434:11434 ollama/ollama\n")
    _die("Local Ollama not found. Please start Ollama first.")


# ── Step 2b: Warm up embedding model ─────────────────────────────────────────
def warmup_embedding_model():
    import json
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
    url = f"http://127.0.0.1:{OLLAMA_LOCAL}/api/embed"
    print(f"[ollama] Warming up embedding model: {model} ...")
    for attempt in range(1, 7):
        try:
            data = json.dumps({"model": model, "input": "warmup"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()
            print(f"  ✓ {model} loaded into memory")
            return
        except Exception as e:
            print(f"  … warmup attempt {attempt}/6: {e}")
            time.sleep(5)
    print(f"  ⚠ Could not warm up {model} — retrieval may be slow on first query")


# ── Step 3: Qdrant via Docker ─────────────────────────────────────────────────
def start_qdrant():
    print("[qdrant] Starting Qdrant...")

    if _http_ok("http://localhost:6333/collections"):
        print("  ✓ Qdrant already running")
        return

    # Check docker
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
    except Exception:
        _die("Docker is not running. Start Docker first.")

    # Check if container exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    existing = result.stdout.strip().splitlines()

    if "ehc_qdrant" in existing:
        subprocess.run(["docker", "start", "ehc_qdrant"], check=True, capture_output=True)
    else:
        subprocess.run([
            "docker", "run", "-d",
            "--name", "ehc_qdrant",
            "--restart", "unless-stopped",
            "-p", "6333:6333",
            "-v", "ehc_qdrant_data:/qdrant/storage",
            "qdrant/qdrant:latest",
        ], check=True, capture_output=True)

    _wait_for("http://localhost:6333/collections", "Qdrant", retries=20, interval=2)


# ── Step 4: Python venv + deps ────────────────────────────────────────────────
def setup_venv():
    venv = ROOT / "venv"
    pip  = venv / "bin" / "pip"
    py   = venv / "bin" / "python3"
    deps_stamp_file = venv / ".deps_stamp"

    if not venv.exists():
        print("[venv] Creating virtual environment...")
        _run([sys.executable, "-m", "venv", str(venv)])
        print("  ✓ venv created")
    else:
        print("[venv] venv exists")

    if not (venv / "lib").exists():
        _die("venv creation failed")

    extra_packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic-settings",
        "python-multipart",
        "python-dotenv",
        "langgraph>=0.1.0",
        "langchain>=0.2.0",
        "langchain-core>=0.2.0",
        "langchain-ollama>=0.1.0",
        "langchain-openai>=0.1.0",
        "langchain-text-splitters>=0.2.0",
        "python-docx",
        "Pillow",
        "pytesseract",
        "httpx",
        "python-telegram-bot>=20.0",
        "psutil>=5.9.0",
    ]
    deps_signature = hashlib.sha256(
        (
            (ROOT / "shared/py/requirements.txt").read_text(encoding="utf-8")
            + "\n"
            + "\n".join(extra_packages)
        ).encode("utf-8")
    ).hexdigest()
    force_reinstall = os.getenv("EHC_REINSTALL_DEPS", "0").strip().lower() in {"1", "true", "yes"}
    if not force_reinstall and deps_stamp_file.exists():
        current_stamp = deps_stamp_file.read_text(encoding="utf-8").strip()
        if current_stamp == deps_signature:
            print("[deps] Unchanged — skipping install")
            return py

    print("[deps] Installing dependencies (this may take a few minutes)...")
    _run([str(pip), "install", "--quiet", "--upgrade", "pip"])
    _run([str(pip), "install", "--quiet", "-r", "shared/py/requirements.txt"])
    # Ensure qdrant-client is not downgraded by other packages
    _run([str(pip), "install", "--quiet", "qdrant-client>=1.7.0"])
    _run([str(pip), "install", "--quiet", *extra_packages])
    deps_stamp_file.write_text(deps_signature + "\n", encoding="utf-8")
    print("  ✓ All dependencies installed")
    return py


# ── Step 5: Check local Ollama ────────────────────────────────────────────────
def check_ollama_ready():
    print(f"[ollama] Checking local Ollama on localhost:{OLLAMA_LOCAL}...")
    if _port_open("127.0.0.1", OLLAMA_LOCAL):
        print(f"  ✓ Ollama running on localhost:{OLLAMA_LOCAL}")
        return
    print(f"\n  Ollama not found on port {OLLAMA_LOCAL}.")
    print("  Start it with:  ollama serve")
    print("  Or via Docker:  docker run -d --gpus all -p 11434:11434 ollama/ollama")
    _die("Ollama not running. Please start Ollama first.")


# ── Step 6: Start API Gateway ─────────────────────────────────────────────────
def start_api(py: Path):
    print("[api] Starting API Gateway...")

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(ROOT),
        str(ROOT / "apps" / "api_gateway"),
        str(ROOT / "apps" / "agent_runtime"),
    ])
    # Load .env values into env
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env.setdefault(k.strip(), v.strip())

    _run_bg(
        str(py), "apps/api_gateway/main.py",
        log_file="logs/api.log",
        env=env,
    )
    _wait_for("http://localhost:8000/api/v1/health", "API Gateway", retries=30, interval=2)


# ── Step 7: Start Telegram Bot ────────────────────────────────────────────────
BOT_PY: Path | None = None
BOT_ENV: dict | None = None

def _kill_stale_bot():
    """Kill any existing bot.py process to avoid Telegram getUpdates conflict."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "apps/telegram_bot/bot.py"],
            capture_output=True, text=True,
        )
        for pid_str in result.stdout.strip().splitlines():
            try:
                pid = int(pid_str)
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
                    print(f"  [bot] Killed stale bot process (pid {pid})")
                    time.sleep(1)
            except Exception:
                pass
    except FileNotFoundError:
        pass  # pgrep not available — skip


def start_bot(py: Path):
    global BOT_PY, BOT_ENV
    print("[bot] Starting Telegram Bot...")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env.setdefault(k.strip(), v.strip())

    token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("  ⚠ TELEGRAM_BOT_TOKEN not set in .env — skipping bot")
        return

    BOT_PY = py
    BOT_ENV = env

    _kill_stale_bot()

    _run_bg(
        str(py), "apps/telegram_bot/bot.py",
        log_file="logs/bot.log",
        env=env,
    )
    time.sleep(3)
    print("  ✓ Telegram bot started (polling)")


def _restart_bot():
    """Spawn a fresh bot process and register it in PROCS."""
    if BOT_PY is None or BOT_ENV is None:
        return
    print("[bot] Auto-restarting Telegram bot...")
    _kill_stale_bot()
    time.sleep(2)
    _run_bg(
        str(BOT_PY), "apps/telegram_bot/bot.py",
        log_file="logs/bot.log",
        env=BOT_ENV,
    )
    print("[bot] ✓ Bot restarted")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  EHC AI MVP — Full Stack Startup")
    print("=" * 55)

    setup_env()

    # Load .env into os.environ early so MODEL_PROVIDER is available
    env_path = ROOT / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

    setup_dirs()

    provider = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
    if provider != "vllm":
        check_ollama_ready()
    else:
        print("[provider] MODEL_PROVIDER=vllm — skipping Ollama check")

    warmup_embedding_model()
    start_qdrant()
    py = setup_venv()
    start_api(py)
    start_bot(py)

    print()
    print("=" * 55)
    print("  ✅ All services running!")
    print(f"  API:      http://localhost:8000")
    print(f"  Docs:     http://localhost:8000/docs")
    print(f"  Health:   http://localhost:8000/api/v1/health")
    print(f"  Qdrant:   http://localhost:6333")
    print(f"  Ollama:   http://localhost:{OLLAMA_LOCAL}  (qwen3.5:9b + bge-m3)")
    print(f"  Logs:     {ROOT}/logs/")
    print("  Press Ctrl+C to stop all services")
    print("=" * 55)

    # Keep alive — watch child processes; auto-restart bot on exit
    bot_script = str(ROOT / "apps/telegram_bot/bot.py")
    while True:
        dead = [p for p in PROCS if p.poll() is not None]
        for p in dead:
            name = " ".join(str(a) for a in p.args)
            print(f"\n⚠ Process exited (code {p.returncode}): {name}")
            print(f"  Check logs in {ROOT}/logs/")
            PROCS.remove(p)
            if bot_script in name:
                _restart_bot()
        time.sleep(5)


if __name__ == "__main__":
    main()
