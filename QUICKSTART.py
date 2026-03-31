"""EHC AI MVP - Development Console

Quick commands for testing the system:
"""

# === 1. SETUP ===
# Copy .env template
# cp .env.example .env

# === 2. START INFRASTRUCTURE ===
# bash scripts/dev_up.sh

# === 3. SETUP PYTHON ENV (optional, for local testing) ===
# python3 -m venv venv
# source venv/bin/activate
# pip install -r shared/py/requirements.txt
# pip install -r apps/api_gateway/requirements.txt

# === 4. RUN INGESTION ===
# bash scripts/run_ingestion.sh --project FAQ

# === 5. TEST API ===

# Health check
# curl http://localhost:8000/health

# Simple query
# curl -X POST http://localhost:8000/api/v1/ask \
#   -H "Content-Type: application/json" \
#   -d '{"query": "Làm sao để kiểm tra tồn kho?"}'

# === 6. SHUTDOWN ===
# bash scripts/dev_down.sh

print("""
╔════════════════════════════════════════════════════════════════╗
║         EHC AI Helpdesk MVP - Quick Start Guide               ║
╚════════════════════════════════════════════════════════════════╝

✓ All modules created and ready for deployment!

NEXT STEPS:
1. Verify .env configuration
2. Start infrastructure: bash scripts/dev_up.sh
3. Run ingestion: bash scripts/run_ingestion.sh --project FAQ
4. Test API: curl http://localhost:8000/health

DOCUMENTATION: See README.md for full details
API Docs: http://localhost:8000/docs (after starting)
""")
