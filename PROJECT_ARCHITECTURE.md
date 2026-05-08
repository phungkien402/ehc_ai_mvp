# EHC AI MVP — Project Deep Dive

**Ngày quét:** 22/04/2026  
**Trạng thái:** MVP hoàn chỉnh (6 module)

---

## 🎯 Mục tiêu & Context

**Vấn đề:** EHC (nhà phát triển phần mềm bệnh viện) có hàng nghìn ticket Redmine nhưng không có công cụ tìm kiếm thông minh → nhân viên IT lúc thúc tìm lại cách xử lý lỗi cũ.

**Giải pháp:** Chatbot RAG tích hợp Telegram/API
- Tìm kiếm ngữ nghĩa (semantic) trên Qdrant vector DB
- Đọc ảnh chụp màn hình (OCR qua Ollama vision)
- Tổng hợp câu trả lời bằng LLM tiếng Việt (Qwen2.5)
- Rewrite query tự động nếu kết quả không tốt (Self-RAG)

---

## 🏗️ Architecture 

```
┌─────────────────────────────────────────────────────────┐
│                      User Layer                         │
│   Telegram Bot  │  FastAPI /api/v1/ask  │  Dashboard    │
└────────────────┬────────────────────────┬────────────────┘
                 │                        │
                 ▼                        ▼
        ┌──────────────────────────────────┐
        │   FastAPI Gateway (:8000)        │
        │   - Request/response validation  │
        │   - Session memory (2h TTL)      │
        │   - CORS + Swagger docs          │
        └──────────────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────────┐
        │    LangGraph Agent Runtime       │
        │  ┌────────────────────────────┐  │
        │  │ 7 Nodes (Self-RAG):        │  │
        │  │ 1. extract_ocr_if_image   │  │
        │  │ 2. call_agent             │  │
        │  │ 3. grade_documents        │  │
        │  │ 4. rewrite_query          │  │
        │  │ 5. generate_final_answer  │  │
        │  │ 6. natural_chat           │  │
        │  │ 7. llm_unavailable        │  │
        │  └────────────────────────────┘  │
        └──────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┬────────────────────┐
        │                     │                    │
        ▼                     ▼                    ▼
   ┌─────────┐         ┌──────────┐         ┌──────────┐
   │  Qdrant │         │  Ollama  │         │  Ollama  │
   │ (Vector │         │   LLM    │         │  Vision  │
   │   DB)   │         │  qwen2.5 │         │ qwen3.5  │
   │ :6333   │         │ :11434   │         │ :11434   │
   └─────────┘         └──────────┘         └──────────┘
```

**Data Source:** Redmine FAQ Project (100+ issues/page pagination)

---

## 📁 Project Structure

```
ehc_ai_mvp/
│
├── 🟦 shared/py/                    # Module 1: Shared Utilities
│   ├── models/
│   │   └── faq.py                   # FAQSource, FAQChunk, ModuleDocSection, QdrantPayload, WorkflowState
│   ├── clients/
│   │   ├── qdrant_client.py         # QdrantWrapper (search, upsert, collection mgmt)
│   │   └── ollama_client.py         # OllamaEmbeddings, OllamaVision, OllamaChat
│   └── utils/
│       ├── text.py                  # Vietnamese normalize, chunking, merging
│       └── logging.py               # Structured logging setup
│
├── 🟦 pipelines/ingestion/          # Module 2: Data Ingestion
│   ├── main.py                      # CLI: --project FAQ --source docx --dry-run
│   └── app/
│       ├── core/config.py           # Pydantic Settings (env-based)
│       ├── sources/
│       │   ├── redmine_client.py    # Fetch issues from Redmine API
│       │   └── docx_parser.py       # Parse DOCX + extract text/images ⭐ NEW
│       ├── processors/
│       │   └── chunker.py           # Split into 500-char chunks (overlap 100)
│       ├── sinks/
│       │   └── qdrant_writer.py     # Batch embed (bge-m3) + upsert
│       └── utils/
│           └── image_storage.py     # ImageStorage + ImageExtractor ⭐ NEW
│
├── 🟦 apps/agent_runtime/          # Module 3: LangGraph Workflow
│   ├── main.py                      # Test/debug entry point
│   └── app/
│       ├── core/config.py           # Config loader
│       ├── graph/
│       │   ├── state.py             # WorkflowState TypedDict
│       │   ├── nodes.py             # 7 node implementations
│       │   ├── tools.py             # search_faq_tool + scoring logic
│       │   ├── router.py            # Conditional routing (relevant/not relevant)
│       │   └── workflow.py          # StateGraph compilation
│       └── ...
│
├── 🟦 apps/api_gateway/            # Module 4: FastAPI Server
│   ├── main.py                      # Uvicorn entry point
│   └── app/
│       ├── core/config.py           # Settings from env
│       ├── core/session_memory.py   # In-memory session store (TTL 2h)
│       ├── api/routes.py            # Endpoints: /ask, /ocr, /health, /dashboard/*
│       └── schemas.py               # Pydantic request/response models
│
├── 🟦 apps/telegram_bot/           # Module 5: Telegram Integration
│   ├── bot.py                       # Long-polling bot with image caching
│   └── ...
│
├── 🟦 docker/                      # Module 6: Container & Deployment
│   ├── docker-compose.yml           # Qdrant + Redis + API
│   ├── Dockerfile.api               # Python 3.11 slim
│   └── Dockerfile.telegram          # (optional)
│
├── 🟦 scripts/                     # Bootstrap & Operations
│   ├── dev_up.sh                    # Start all services
│   ├── dev_down.sh                  # Stop services
│   ├── run_ingestion.sh             # Execute ingestion pipeline
│   ├── start_ollama_tunnel.sh       # SSH port forward GPU
│   ├── start_services.sh            # Start API + bot (unified)
│   ├── stop_services.sh             # Shutdown cleanly
│   ├── status_services.sh           # Health check all
│   ├── clean_logs.sh                # Archive old logs
│   └── ...
│
├── 🟦 docs/                        # Documentation
│   ├── PROJECT_OVERVIEW.md          # Vietnamese docs
│   ├── DOCX_IMAGE_EXTRACTION.md     # ⭐ NEW - Image handling guide
│   ├── OPS_RUNBOOK_DAILY.md         # Daily operations checklist
│   ├── workflow.mmd                 # Mermaid diagram
│   └── workflow-graph.mmd           # LangGraph visualization
│
├── tests/                           # Unit tests (pytest)
├── reports/eval/                    # Evaluation reports (455 test cases)
├── logs/                            # Runtime logs
├── qdrant_data/                     # Persistent Qdrant storage
│
├── 📋 .env.example                 # Config template
├── 📋 .env                         # Actual config (GITIGNORED)
├── 📋 requirements.txt              # Python deps
├── 📋 README.md                     # Full guide (400+ lines)
├── 📋 QUICKSTART.py                 # Quick reference
├── 📋 BUILD_COMPLETE.md             # Build summary
└── 📋 DOCX_IMAGE_EXTRACTION.md      # ⭐ NEW

```

---

## 🔄 Data Flow (Happy Path)

### 1. **Ingestion** (Manual or Cron)
```bash
python pipelines/ingestion/main.py --source both --project FAQ
```

**Steps:**
1. `RedmineClient` → Fetch all FAQ issues from Redmine (paginated 100/page)
2. `DocxParser` → Parse DOCX files, extract text + images
   - Images saved to `data/docx_images/images/` with hash IDs
   - Metadata tracked in `data/docx_images/image_metadata.json`
3. `FAQChunker` → Split into 500-char chunks (overlap 100)
   - Associates `image_ids` with chunks
4. `QdrantWriter` → Batch embed (bge-m3 @ Ollama) + upsert to Qdrant
   - Stores `image_ids` in Qdrant payload

**Output:** Qdrant collection `ehc_faq` or `ehc_docs` with 1024-dim vectors + metadata

---

### 2. **Query Time** (3-10 seconds)

```
User asks:  "Lỗi không in được báo cáo PTTT?"
                         ▼
           [FastAPI /api/v1/ask]
                         ▼
[extract_ocr_if_image] (skip, no image)
                         ▼
[call_agent]  (LLM + tool binding)
        │
        └─► tool_calls["search_faq_tool"]
                         ▼
[tools] node
  ├─ Embed query → 1024-dim vector (bge-m3)
  ├─ Search Qdrant (cosine similarity, threshold 0.5)
  └─ Return top 5 chunks + scores
  └─ Also return associated image_ids
                         ▼
[grade_documents]  (LLM grader + heuristic)
  ├─ Primary: LLM JSON grader (yes/no)
  ├─ Fallback: (vector_score ≥0.78 AND lexical ≥0.45)
  ├─ Ambiguity detect: score <0.75 & gap ≤0.03 → suggest candidates
  └─ Return: relevant=True/False
                         ▼
            Is relevant?
           /             \
         YES             NO
         │               │
         ▼               ▼
    [generate_       [rewrite_query]
    final_answer]    (synonym subst, retry)
       │                    │
       │            (loop → call_agent, max N)
       │                    │
       └────────┬───────────┘
                │
                ▼
        Response to user:
        {
          "answer": "Để in báo cáo PTTT...",
          "sources": [{
            "issue_id": 19135,
            "score": 0.92,
            "snippet": "...",
            "image_ids": ["img_a1b2c3d4"]  # ⭐ NEW
          }],
          "execution_time_ms": 3420,
          "rewrite_count": 0
        }
```

**LangGraph Nodes:**
| Node | Chức năng | Timeout |
|------|-----------|---------|
| `extract_ocr_if_image` | Vision model → [TEXT][ISSUE][UI] | 40s |
| `call_agent` | LLM + force tool call | 60s |
| `grade_documents` | Relevance check | 30s |
| `rewrite_query` | Synonym + retry | 30s |
| `generate_final_answer` | Synthesize answer | 30s |
| `natural_chat` | Small-talk (chào, cảm ơn) | — |
| `llm_unavailable` | Graceful fallback | — |

---

## 🛠️ Configuration (.env)

```bash
# Redmine
REDMINE_URL=http://localhost:3000
REDMINE_API_KEY=your_key

# Qdrant Vector DB
QDRANT_URL=http://localhost:6333    # Docker service name
QDRANT_API_KEY=                     # Optional

# Ollama (Self-hosted on GPU)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_VISION_MODEL=qwen3.5:4b
OLLAMA_CHAT_MODEL=qwen2.5:14b

# Model Provider (ollama or vllm)
MODEL_PROVIDER=ollama

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000

# Ingestion
CHUNK_SIZE=500
CHUNK_OVERLAP=100
BATCH_SIZE=10

# DOCX Processing ⭐ NEW
DOCX_INPUT_DIR=data/documents
DOCX_OCR_ENABLED=true
DOCX_OCR_MAX_IMAGES=6
```

---

## 📊 Module Breakdown

### Module 1: Shared Utilities ✅
**What:** Data models, clients, utilities
- `FAQSource` → Redmine issue (ID, subject, description, attachments)
- `ModuleDocSection` → DOCX section (title, content, **image_ids**) ⭐
- `FAQChunk` → Embedable unit (content, metadata, **image_ids**) ⭐
- `QdrantPayload` → Stored in Qdrant (includes **image_ids**) ⭐
- `QdrantWrapper` → Search, upsert, collection management
- `OllamaEmbeddings` → Batch embed via Ollama API
- `OllamaVision` → OCR via Ollama vision model

### Module 2: Ingestion Pipeline ✅
**What:** Redmine → Qdrant → Searchable Vector DB

**Entry:** `python pipelines/ingestion/main.py`

**Components:**
- `RedmineClient` → Fetch issues (pagination, attachments)
- `DocxParser` → Parse DOCX + extract text/images ⭐
- `ImageStorage` + `ImageExtractor` → Manage image files ⭐
- `FAQChunker` → Split into chunks (with image_ids) ⭐
- `QdrantWriter` → Batch embed + upsert

**Output:** 
- Qdrant collections: `ehc_faq`, `ehc_docs`
- Image storage: `data/docx_images/` ⭐

### Module 3: Agent Runtime ✅
**What:** LangGraph Self-RAG workflow

**7 Nodes:**
1. `extract_ocr_if_image` → Vision OCR if image exists
2. `call_agent` → LLM decides tool vs direct answer
3. `grade_documents` → Relevance check (LLM + heuristic)
4. `rewrite_query` → Retry with synonym substitution
5. `generate_final_answer` → Final synthesis
6. `natural_chat` → Small-talk handling
7. `llm_unavailable` → Graceful degradation

**Grading Logic:**
```
IF LLM.grade == "yes":
    RETURN relevant
ELIF (vector_score ≥ 0.78) AND (lexical_score ≥ 0.45):
    RETURN relevant
ELIF (lexical_score ≥ 0.70):
    RETURN relevant
ELSE:
    RETURN not_relevant → rewrite_query
```

**Ambiguity Detection:** If score < 0.75 AND gap with 2nd place ≤ 0.03 → suggest candidates

### Module 4: FastAPI Gateway ✅
**What:** REST API entry point

**Endpoints:**
- `POST /api/v1/ask` → Query with optional image
- `GET /api/v1/health` → System health
- `GET /api/v1/dashboard/overview` → Status, uptime, latency
- `GET /api/v1/dashboard/logs` → Paginated logs
- `WS /api/v1/dashboard/logs/stream` → Real-time logs
- `GET /ui` → React dashboard

**Features:**
- Async/await (Uvicorn)
- CORS enabled
- Session memory (2h TTL, max 2000 sessions)
- Swagger docs at `/docs`

### Module 5: Telegram Bot ✅
**What:** Chat interface via Telegram

**Features:**
- Long-polling (no webhook needed)
- Image caching (5 min per chat)
- Follow-up detection (regex tiếng Việt)
- Exponential backoff retry
- Error handling (timeout, connection error)

### Module 6: Docker & Operations ✅
**What:** Container orchestration + scripts

**Services:**
- Qdrant (vector DB, port 6333)
- Redis (caching, port 6379)
- API Gateway (port 8000)

**Scripts:**
- `dev_up.sh` → Start all services
- `dev_down.sh` → Stop services
- `run_ingestion.sh` → Run pipeline
- `start_ollama_tunnel.sh` → SSH port forward to GPU
- `start_services.sh` → Unified startup (API + bot)
- `clean_logs.sh` → Archive old logs

---

## 🚀 Quick Start

### 1. Initialize
```bash
cd ehc_ai_mvp
cp .env.example .env
# Edit .env with your Redmine API key
```

### 2. Start Infrastructure
```bash
bash scripts/dev_up.sh
# Waits for health checks:
# ✓ Qdrant: http://localhost:6333
# ✓ API: http://localhost:8000
```

### 3. Ingest FAQ Data
```bash
# Both FAQ + DOCX with images
bash scripts/run_ingestion.sh --source both --project FAQ

# Or just DOCX
bash scripts/run_ingestion.sh --source docx --docx-dir data/documents

# Or with dry-run
bash scripts/run_ingestion.sh --project FAQ --dry-run
```

### 4. Test API
```bash
# Health check
curl http://localhost:8000/health

# Ask question
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Làm sao để kiểm tra tồn kho?"}'

# Swagger
open http://localhost:8000/docs
```

### 5. Shutdown
```bash
bash scripts/dev_down.sh
```

---

## ⭐ NEW: DOCX + Image Extraction

### What Changed
- **DocxParser** now extracts images + text separately
- **ImageStorage** manages image files with hash-based IDs (img_XXXXX)
- **Image metadata** tracked in JSON for easy lookup
- **Vector DB** stores image_ids alongside text chunks
- **RAG responses** include image_ids for multimodal LLM

### Image Flow
```
DOCX → DocxParser
  ├─ Text → FAQChunk.content → Vector DB
  └─ Images → ImageStorage → data/docx_images/
              └─ image_ids → QdrantPayload.image_ids
```

### Metadata Example
```json
{
  "img_a1b2c3d4": {
    "image_id": "img_a1b2c3d4",
    "filename": "image_a1b2c3d4.jpg",
    "original_filename": "media1.jpg",
    "source_file": "user_guide.docx",
    "alt_text": "Screenshot showing menu options",
    "size_bytes": 45320
  }
}
```

### Usage in RAG
```python
# Search returns chunks with image_ids
result = qdrant.search(query_vector)
image_ids = result.payload["image_ids"]

# Fetch images from storage
for img_id in image_ids:
    image_bytes = image_storage.get_image_bytes(img_id)
    # Send to multimodal LLM alongside text
```

See `DOCX_IMAGE_EXTRACTION.md` for complete guide.

---

## 🔐 Security Considerations

- **API Key** (Redmine, Qdrant) → stored in .env (GITIGNORED)
- **SSH Tunnel** → GPU access via port forwarding
- **Session Memory** → In-process, not persistent
- **CORS** → Configured for frontend domain
- **Logging** → No sensitive data in logs

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Query execution time | 3-10 seconds |
| Vector search | <100ms |
| Embedding (batch 10) | 1-2 seconds |
| OCR (if image) | 5-10 seconds |
| Chunk retrieval limit | Top 5 |
| Rewrite loop max | 3 iterations |
| Session TTL | 2 hours |
| Max concurrent sessions | 2000 |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Specific module
pytest tests/api/ -v

# With coverage
pytest --cov=apps --cov=pipelines tests/
```

---

## 📚 Documentation

- **README.md** → Full guide (400+ lines)
- **QUICKSTART.py** → Quick reference
- **BUILD_COMPLETE.md** → Build summary
- **PROJECT_OVERVIEW.md** → Vietnamese docs
- **DOCX_IMAGE_EXTRACTION.md** → ⭐ NEW - Image handling
- **OPS_RUNBOOK_DAILY.md** → Daily operations
- **docs/workflow.mmd** → Mermaid diagrams

---

## 🎯 Next Steps (Recommendations)

1. **Image Retrieval in RAG** → Modify LangGraph to fetch actual images
2. **Image Display in API** → Return base64 images in responses
3. **Dashboard Image View** → Show retrieved images in UI
4. **CLIP-based Image Search** → Search by image content, not just text
5. **Image Quality Metrics** → Track OCR accuracy, image relevance
6. **Persistent Session Storage** → Replace in-memory with Redis
7. **Multi-language Support** → Extend beyond Vietnamese
8. **Fine-tuning** → Train models on EHC-specific domain

---

## 📞 Key Files at a Glance

| File | Purpose |
|------|---------|
| `.env` | Configuration (secrets) |
| `shared/py/models/faq.py` | Data models (includes image_ids) ⭐ |
| `pipelines/ingestion/main.py` | Ingestion CLI |
| `apps/agent_runtime/app/graph/workflow.py` | LangGraph compilation |
| `apps/api_gateway/app/api/routes.py` | API endpoints |
| `apps/telegram_bot/bot.py` | Telegram bot |
| `docker-compose.yml` | Container orchestration |
| `DOCX_IMAGE_EXTRACTION.md` | ⭐ NEW - Image guide |

---

**Project Status:** ✅ MVP Complete (6/6 modules)  
**Latest Addition:** DOCX + Image Extraction (22/04/2026)  
**Ready for:** Deployment, fine-tuning, integration testing
