# EHC AI Helpdesk MVP

Vietnamese helpdesk FAQ chatbot powered by RAG (Retrieval-Augmented Generation).

**Stack:**
- **Data**: Redmine → Qdrant vector database (bge-m3 embeddings)
- **LLM**: Ollama over SSH tunnel (`qwen2.5:14b` main answer, `qwen3.5:4b` OCR, `bge-m3` embeddings)
- **Workflow**: LangGraph Self-RAG with rewrite loop and grader
- **API**: FastAPI with async support
- **Deploy**: Docker Compose (Qdrant + Redis + API)

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /home/phungkien/ehc_ai_mvp
cp .env.example .env
# Edit .env with your Redmine API key and Ollama URL
```

### 2. Start Infrastructure

```bash
./scripts/dev_up.sh
```

Wait for all services to be healthy:
- ✓ Qdrant: http://localhost:6333
- ✓ API: http://localhost:8000

### 2.1. Start Ollama Tunnel

```bash
./scripts/start_ollama_tunnel.sh
```

### 2.2. Start vLLM Canary (10%)

This starts API Gateway in native process mode with deterministic canary routing.

```bash
# Requires local forwards:
# 127.0.0.1:18000 -> vLLM LLM
# 127.0.0.1:18001 -> vLLM Embedding/Vision
./scripts/start_canary_rollout.sh
```

Verify sticky routing:

```bash
python3 scripts/verify_canary_routing.py --runs 10 --rollout-percent 10
```

Rollback canary quickly:

```bash
./scripts/rollback_canary.sh
python3 scripts/verify_canary_routing.py --runs 6 --rollout-percent 0
```

### 2.3. Decommission Ollama Path (after full vLLM cutover)

```bash
./scripts/decommission_ollama.sh
```

This will:
- archive rollout logs to `logs/archive/`
- stop local Ollama SSH tunnel
- restart API in `MODEL_PROVIDER=vllm` mode

One-command rollback back to Ollama:

```bash
./scripts/rollback_to_ollama.sh
```

### 2.4. Unified Daily Operations

Start all runtime services (API + Telegram bot):

```bash
# Full vLLM mode
./scripts/start_services.sh vllm

# Canary mode (default rollout from env/script)
./scripts/start_services.sh canary

# Full Ollama mode
./scripts/start_services.sh ollama
```

Status / stop / log cleanup:

```bash
./scripts/status_services.sh
./scripts/stop_services.sh
./scripts/clean_logs.sh
```

Daily checklist runbook:
- `docs/OPS_RUNBOOK_DAILY.md`

Override target when you rent a new GPU:

```bash
EHC_GPU_HOST=n2.ckey.vn \
EHC_GPU_SSH_PORT=1848 \
EHC_GPU_USER=root \
./scripts/start_ollama_tunnel.sh
```

### 3. Ingest FAQ Data

```bash
# From Redmine project "FAQ"
./scripts/run_ingestion.sh --project FAQ

# Or dry-run to test without writing to Qdrant
./scripts/run_ingestion.sh --project FAQ --dry-run
```

### 4. Ask Questions

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Làm sao để kiểm tra tồn kho?"}'
```

**Via Swagger Docs:**
Visit http://localhost:8000/docs

### 5. Open Operations Dashboard

```bash
cd /home/phungkien/ehc_ai_mvp/apps/dashboard-ui
npm run build
```

Then open http://localhost:8000/ui/

Dashboard tabs:
- Overview: health, uptime, active model list, average latency
- Logs: realtime stream with level/source filters
- Visual Workflow: Mermaid rendering from docs/workflow.mmd

## 📁 Project Structure

```
ehc_ai_mvp/
├── shared/py/               # Shared utilities (data models, clients, utils)
│   ├── models/
│   ├── clients/
│   └── utils/
├── pipelines/ingestion/     # Redmine → Qdrant ingestion
│   └── app/
│       ├── core/            # Config loader
│       ├── sources/         # Redmine API client
│       ├── processors/      # Chunking logic
│       └── sinks/           # Qdrant writer
├── apps/agent_runtime/      # LangGraph Self-RAG workflow
│   └── app/
│       ├── graph/           # Nodes, state, router
│       └── core/            # Config
├── apps/api_gateway/        # FastAPI server
│   └── app/
│       ├── api/             # Routes
│       └── core/            # Config
├── docker/                  # Docker Compose & Dockerfiles
├── scripts/                 # Bootstrap & utility scripts
└── tests/                   # Test suites
```

## 🔄 Data Flow

```
User Query (JSON)
    ↓
[API Gateway] POST /api/v1/ask
    ↓
[LangGraph Workflow]
  ├─ extract_ocr_if_image  → OCR + merge text when image exists
  ├─ agent                 → Decide direct answer vs retrieval tool
  ├─ tools                 → search_faq_tool against Qdrant
  ├─ grade_documents       → LLM + heuristic relevance grading
  ├─ route_after_grading   → Generate or rewrite
  ├─ rewrite_query         → Retry retrieval with refined query
  └─ generate_final_answer → Final synthesis or safe fallback
    ↓
User receives JSON response with answer + sources
```

**Total execution time:** 3-10 seconds

## 🛠️ Configuration

All settings via `.env`:

```bash
# Redmine
REDMINE_URL=http://localhost:3000
REDMINE_API_KEY=your_key

# Qdrant
QDRANT_URL=http://qdrant:6333       # Docker: use service name
QDRANT_API_KEY=                      # Optional

# Ollama (embeddings + vision)
OLLAMA_BASE_URL=http://localhost:11434

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ingestion
CHUNK_SIZE=500
CHUNK_OVERLAP=100
BATCH_SIZE=10
```

## 📊 Module Overview

### Module 1: Shared Utilities (✓ Complete)
- Data models (FAQ, chunks, Qdrant payload)
- Qdrant client wrapper
- Ollama embedding & vision clients
- Text processing (normalization, chunking)
- Structured logging

### Module 2: Ingestion Pipeline (✓ Complete)
- Redmine API client → fetch FAQ issues
- FAQ chunking with 500-char overlap
- Batch embedding (bge-m3)
- Qdrant upsert with metadata

### Module 3: Agent Runtime (✓ Complete)
- **extract_ocr_if_image** → vision OCR + query merge
- **agent** → tool decision and direct-answer shortcut
- **tools** → retrieval against Qdrant
- **grade_documents** → relevance grading
- **rewrite_query** → bounded retry loop
- **generate_final_answer** → grounded answer or fallback
- Structured `TRACE|node=...` markers for dashboard realtime workflow

### Module 4: FastAPI Gateway (✓ Complete)
- `POST /api/v1/ask` → query with optional image
- `GET /health` → Qdrant + Ollama health
- CORS enabled for cross-origin requests
- Swagger/OpenAPI docs at `/docs`

### Module 5: Docker & Config (✓ Complete)
- Docker Compose with Qdrant, Redis, API
- Dockerfiles for API gateway
- Bootstrap scripts (dev_up.sh, dev_down.sh)
- Ingestion runner script

### Module 6: Tests (Next)
- Health checks
- API endpoint tests
- Workflow tests
- Smoke tests

## 🧪 Testing

### Dashboard APIs
```bash
curl http://localhost:8000/api/v1/dashboard/overview
curl "http://localhost:8000/api/v1/dashboard/logs?limit=20"
curl http://localhost:8000/api/v1/dashboard/workflow
```

### Health Check
```bash
curl http://localhost:8000/api/v1/health
# Output: {"status": "ok", "qdrant_ok": true, "ollama_ok": true, ...}
```

### Simple Query
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cách xem hóa đơn?",
    "image_base64": null
  }'
```

### With Image (OCR)
```bash
python3 -c "
import base64
import json
with open('screenshot.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
print(json.dumps({'query': 'Giải thích chiếc biểu đồ này', 'image_base64': img_b64}))
" | curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d @-
```

## 🔧 Troubleshooting

### Qdrant not connecting
```bash
# Check Docker logs
docker-compose -f docker/docker-compose.yml logs qdrant

# Verify connection
curl http://localhost:6333/health
```

### Ollama not found
```bash
# Ensure Ollama is running on your machine/server
curl http://localhost:11434/api/tags

# If using Docker, may need to use host.docker.internal:11434
# Edit docker-compose.yml OLLAMA_BASE_URL
```

### Ingestion script hangs
```bash
# Increase timeout in shared/py/clients/ollama_client.py
# or reduce BATCH_SIZE in .env
```

## 📚 API Reference

### POST /api/v1/ask

Request:
```json
{
  "query": "Câu hỏi tiếng Việt",
  "image_base64": "iVBORw0KGgoAAAANS..."  // Optional
}
```

Response:
```json
{
  "answer": "Trả lời từ FAQ...",
  "sources": [
    {
      "issue_id": "123",
      "snippet": "...",
      "url": "http://redmine/issues/123",
      "score": 0.87
    }
  ],
  "image_urls": ["http://..."],
  "error": null,
  "execution_time_ms": 2345.67
}
```

### GET /health

Response:
```json
{
  "status": "ok",
  "qdrant_ok": true,
  "ollama_ok": true,
  "message": "Qdrant: ✓, Ollama: ✓"
}
```

## 🎯 Next Steps (Phase 2)

- [ ] Conversation memory (Redis + chat history)
- [ ] Zalo webhook integration
- [ ] Multi-language support (Vietnamese + English)
- [ ] LLM-based answer synthesis (instead of snippet)
- [ ] Feedback collection & model fine-tuning
- [ ] Production hardening (auth, rate limiting, monitoring)

## 📝 License

Internal EHC project - 2026
