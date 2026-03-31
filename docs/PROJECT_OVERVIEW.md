# EHC AI Helpdesk MVP — Tổng quan dự án

> Tài liệu tổng hợp từ codebase thực tế (31/03/2026).

---

## 1. Mục tiêu & bối cảnh

Hệ thống **AI Helpdesk** nội bộ cho **EHC (Công ty phần mềm bệnh viện)**, hỗ trợ nhân viên IT tuyến đầu tra cứu giải pháp kỹ thuật từ cơ sở dữ liệu ticket hỗ trợ lịch sử (Redmine).

**Vấn đề cần giải quyết:**
- Nhân viên mất nhiều thời gian tìm lại cách xử lý lỗi đã từng giải quyết
- Ticket Redmine có hàng nghìn mục, không có công cụ tìm kiếm ngữ nghĩa
- Nhiều lỗi phần mềm bệnh viện có ảnh chụp màn hình — cần đọc và phân tích ảnh

**Giải pháp:** Chatbot Telegram tích hợp RAG (Retrieval-Augmented Generation) với khả năng đọc ảnh (OCR), tìm kiếm ngữ nghĩa trên vector DB, và tổng hợp câu trả lời bằng LLM hiểu tiếng Việt.

---

## 2. Kiến trúc tổng quan

```
[Nhân viên]
    │
    ▼ Telegram
[Telegram Bot]
    │  HTTP POST /api/v1/ask
    ▼
[FastAPI Gateway :8000]
    │
    ▼
[LangGraph Agent Runtime]
    ├─► [Ollama Vision qwen2.5vl:7b]  ── đọc ảnh màn hình
    ├─► [Qdrant :6333]                ── tìm kiếm ngữ nghĩa
    └─► [Ollama LLM qwen2.5:14b]      ── tổng hợp câu trả lời

[Dashboard UI :8000/ui]
    React + Vite — giám sát runtime, logs, workflow

[Ingestion Pipeline]  (chạy thủ công / cron)
    Redmine API ──► Chunker ──► Ollama Embeddings ──► Qdrant
```

### GPU qua SSH Tunnel
- **GPU**: RTX 3090 24GB tại `n3.ckey.vn:1534`
- **Tunnel**: `localhost:11435 → GPU:11434` (Ollama)
- **Lệnh**: `sshpass -p '...' ssh -N -L 11435:localhost:11434 -p 1534 root@n3.ckey.vn &`
- **Lưu ý**: Tunnel **không tự reconnect** — cần khởi động lại thủ công sau reboot

---

## 3. Technology Stack

| Tầng | Công nghệ | Ghi chú |
|------|-----------|---------|
| Ngôn ngữ | Python 3.12 | — |
| API Framework | FastAPI + Uvicorn | REST + WebSocket |
| Agent Orchestration | LangGraph 0.1+ | Self-RAG với rewrite loop |
| LLM Framework | LangChain 0.2+ | Tool binding, message history |
| LLM Inference | Ollama 0.19.0 | Self-hosted trên GPU |
| Vector DB | Qdrant 1.7+ | Cosine similarity, 1024-dim |
| Frontend | React + Vite + TypeScript | Dashboard UI |
| Bot | python-telegram-bot | Long polling |
| Data Validation | Pydantic v2 | Schemas, config |
| Nguồn dữ liệu | Redmine REST API | Ticket hỗ trợ lịch sử |
| Container | Docker Compose | Qdrant + Redis (Phase 2) |

### Models sử dụng

| Model | Vai trò |
|-------|---------|
| `qwen2.5:14b` | LLM chính: tổng hợp câu trả lời, đánh giá relevance, rewrite query |
| `qwen2.5vl:7b` | Vision model: đọc và mô tả ảnh chụp màn hình |
| `bge-m3:latest` | Embedding: vector hóa query và chunk (1024 dim) |

---

## 4. Cấu trúc thư mục

```
ehc_ai_mvp/
├── apps/
│   ├── agent_runtime/           # LangGraph workflow engine
│   │   └── app/graph/
│   │       ├── nodes.py         # 7 node xử lý (OCR, agent, grader, rewrite, answer)
│   │       ├── workflow.py      # Compile StateGraph, kết nối edges
│   │       ├── router.py        # Conditional routing logic
│   │       ├── state.py         # WorkflowState TypedDict
│   │       └── tools.py         # search_faq_tool + lexical scoring
│   ├── api_gateway/             # FastAPI server (entry point)
│   │   └── app/
│   │       ├── api/routes.py    # Endpoints: /ask, /ocr, /health, /dashboard/*
│   │       ├── schemas.py       # Pydantic request/response models
│   │       └── core/
│   │           ├── config.py    # Env-based configuration (Pydantic Settings)
│   │           └── session_memory.py  # In-process session store (TTL 2h)
│   ├── telegram_bot/
│   │   └── bot.py               # Telegram polling bot
│   └── dashboard-ui/            # React dashboard (Vite)
│       └── src/App.tsx          # Tabs: Overview, Logs, Workflow, Vision Test
├── shared/py/clients/
│   ├── ollama_client.py         # OllamaEmbeddings, OllamaVision, OllamaChat
│   └── qdrant_client.py         # QdrantWrapper (search, upsert, collection mgmt)
├── pipelines/ingestion/         # ETL: Redmine → Qdrant
│   ├── main.py                  # CLI orchestrator (--project, --dry-run)
│   └── app/
│       ├── sources/redmine_client.py   # Redmine API (paginated, 100/trang)
│       ├── processors/chunker.py       # Sliding window chunking
│       └── sinks/qdrant_writer.py      # Batch embed + upsert
├── docker/
│   └── docker-compose.yml       # Qdrant + Redis + API gateway
├── docs/                        # Tài liệu & Mermaid diagrams
├── scripts/                     # Shell + Python tiện ích
├── tests/                       # pytest test suite
├── reports/eval/                # Kết quả đánh giá (455 test cases)
├── .env                         # Secrets — GITIGNORED
├── .env.example                 # Template cho .env
└── requirements.txt             # Python deps
```

---

## 5. Agent Workflow (LangGraph Self-RAG)

Kiến trúc Self-RAG với vòng lặp rewrite khi kết quả không đủ relevance:

```
START
  │
  ▼
[extract_ocr_if_image]     ← Có ảnh? Vision model → [TEXT][ISSUE][UI]
  │
  ├─ small-talk ──────────► [natural_chat] ──────────────────► END
  ├─ LLM offline ─────────► [llm_unavailable] ──────────────► END
  └─ normal query ────────► [agent] (LLM + tool binding)
                                  │
                       ┌──────────┴──────────┐
               tool_calls?                 direct answer
                  ▼                              ▼
              [tools]                  [generate_final_answer] ──► END
          (search_faq_tool
           → Qdrant search)
                  │
          [grade_documents]
          (LLM + heuristic)
                  │
          ┌───────┴───────────────┐
        relevant              not relevant
          ▼                        ▼
[generate_final_answer]      [rewrite_query]
       ──► END               (loop → agent, max N lần)
```

### Các node

| Node | Chức năng |
|------|-----------|
| `extract_ocr_if_image` | Vision model đọc ảnh → `[TEXT]`, `[ISSUE]`, `[UI]`; xây RAG query từ 3 section |
| `call_agent` | LLM với tools bound; force gọi tool ở lần đầu |
| `grade_documents` | LLM grader + heuristic fallback: vector ≥0.78 × lexical ≥0.45 |
| `rewrite_query` | Rewrite query khi grader reject; synonym substitution domain-specific |
| `generate_final_answer` | Tổng hợp từ top chunks, phát hiện ambiguity, guided fallback |
| `natural_chat` | Xử lý chào hỏi, cảm ơn (không cần retrieval) |
| `llm_unavailable` | Graceful degradation khi Ollama offline |

### Grading Logic

```
Primary:  LLM JSON grader (yes/no + reason)
Fallback: (vector_score >= 0.78 AND lexical_score >= 0.45)
       OR  lexical_only >= 0.70
```

### Ambiguity Detection
Khi top chunk score < 0.75 **và** khoảng cách với candidate thứ 2 ≤ 0.03 → trả về danh sách candidates + hướng dẫn người dùng làm rõ, thay vì ép 1 kết quả.

### Lexical Scoring (tiếng Việt)
- Token ≥3 ký tự, bỏ stopword (177 từ tiếng Việt)
- **Giữ marker trạng thái**: `"đã"` / `"chưa"` → bonus +0.06 (cùng trạng thái), penalty -0.10 (ngược)
- **Rerank cuối**: vector 60% + lexical 40%

---

## 6. Ingestion Pipeline

```bash
# Đồng bộ toàn bộ FAQ từ Redmine vào Qdrant
python pipelines/ingestion/main.py --project FAQ

# Dry run (chỉ xem, không ghi)
python pipelines/ingestion/main.py --project FAQ --dry-run
```

| Bước | Component | Chi tiết |
|------|-----------|---------|
| **Fetch** | `RedmineClient` | Pagination 100 issues/trang, bao gồm attachments |
| **Chunk** | `FAQChunker` | Sliding window 500 chars, overlap 100 |
| **Embed** | `QdrantWriter` | Batch 10, bge-m3 → 1024-dim vector |
| **Store** | `QdrantWrapper` | Upsert vào collection `ehc_faq` (auto-create) |

---

## 7. API Endpoints

| Endpoint | Method | Chức năng |
|----------|--------|-----------|
| `/api/v1/ask` | POST | Q&A chính: nhận `query` + `image_base64` (optional) |
| `/api/v1/ocr` | POST | Test OCR standalone |
| `/api/v1/health` | GET | Kiểm tra Qdrant + Ollama connectivity |
| `/api/v1/dashboard/overview` | GET | Uptime, service status, active models, avg latency |
| `/api/v1/dashboard/logs` | GET | Paginated logs với cursor |
| `/api/v1/dashboard/logs/stream` | WS | Realtime log stream (WebSocket) |
| `/api/v1/dashboard/workflow` | GET | Mermaid diagram workflow |
| `/ui` | GET | Dashboard React SPA |

**Request `/api/v1/ask`:**
```json
{
  "query": "Lỗi không in được báo cáo PTTT",
  "image_base64": "...",
  "user_id": "123456",
  "channel": "telegram"
}
```

**Response:**
```json
{
  "answer": "Để in báo cáo PTTT, bạn cần...",
  "sources": [{"issue_id": 19135, "score": 0.92, "snippet": "..."}],
  "ocr_text": "[TEXT]...[ISSUE]...[UI]...",
  "execution_time_ms": 3420,
  "rewrite_count": 0
}
```

---

## 8. Session Memory

- **Lưu trữ**: In-process dict (mất khi restart API)
- **Key**: `channel:user_id` — vd `telegram:123456`
- **TTL**: 2 giờ; max 2000 sessions đồng thời
- **Nội dung**: 8 turns gần nhất + last ticket reference
- **Follow-up detection**: Regex "lỗi này", "cái này", "chi tiết hơn" → tự append ticket context vào query

---

## 9. Telegram Bot

- **Mode**: Long polling (không dùng webhook)
- **Image flow**: Upload ảnh → cache 5 phút per chat → hỏi text sau → tự gắn ảnh cached
- **Follow-up**: Regex tiếng Việt → tự include ảnh cached nếu có
- **Citation**: Đã tắt — không hiển thị "Nguồn: Ticket #..."
- **Error handling**: Timeout, connect error, HTTP error → message tiếng Việt phù hợp
- **Retry**: Exponential backoff cho API calls

---

## 10. Dashboard UI (React/Vite)

Phục vụ tại `http://<server>:8000/ui`

| Tab | Chức năng |
|-----|-----------|
| **Overview** | System status (API/Qdrant/Ollama), uptime, active models, avg latency |
| **Logs** | WebSocket realtime hoặc polling; filter theo level/logger |
| **Visual Workflow** | Mermaid diagram + highlight node đang active theo log stream |
| **Vision Test** | Upload ảnh + câu hỏi → test toàn bộ OCR + RAG pipeline |

---

## 11. Cấu hình (.env)

File `.env` tại root project — **KHÔNG được commit lên Git**:

```env
# Redmine (ingestion)
REDMINE_URL=http://co.ehc.vn:81/redmine
REDMINE_API_KEY=<key>

# Vector DB
QDRANT_URL=http://localhost:6333

# Ollama (qua SSH tunnel đến GPU)
OLLAMA_BASE_URL=http://localhost:11435
OLLAMA_VISION_MODEL=qwen2.5vl:7b
OLLAMA_LLM_MODEL=qwen2.5:14b
OLLAMA_GRADER_MODEL=qwen2.5:14b
OLLAMA_REWRITE_MODEL=qwen2.5:14b
OLLAMA_EMBEDDING_MODEL=bge-m3:latest

# Telegram
TELEGRAM_BOT_TOKEN=<token>

# API
API_HOST=0.0.0.0
API_PORT=8000

# Ingestion params
CHUNK_SIZE=500
CHUNK_OVERLAP=100
BATCH_SIZE=10
```

---

## 12. Khởi động hệ thống

```bash
# Bước 1: SSH tunnel đến GPU (cần chạy lại sau mỗi lần server/tunnel chết)
sshpass -p 'Admin@123' ssh -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -N -L 11435:localhost:11434 -p 1534 root@n3.ckey.vn &

# Kiểm tra tunnel
curl http://localhost:11435/api/tags

# Bước 2: Qdrant vector DB
docker compose -f docker/docker-compose.yml up qdrant -d

# Bước 3: API Gateway
mkdir -p logs
nohup venv/bin/python3 apps/api_gateway/main.py >> logs/api_gateway.stdout.log 2>&1 &

# Bước 4: Telegram Bot
nohup venv/bin/python3 apps/telegram_bot/bot.py >> logs/bot.stdout.log 2>&1 &

# Bước 5: Kiểm tra
curl http://localhost:8000/
# → {"status": "running", ...}
```

---

## 13. Tiến trình phát triển

| Phase | Nội dung | Trạng thái |
|-------|---------|-----------|
| **Phase 1 — Core RAG** | Ingestion Redmine→Qdrant; LangGraph Self-RAG cơ bản; FastAPI; Telegram bot | ✅ Done |
| **Phase 2 — Vision** | `qwen2.5vl:7b` đọc ảnh; structured output [TEXT][ISSUE][UI]; Vision Test tab | ✅ Done |
| **Phase 3 — Quality** | Lexical scoring tiếng Việt; status marker đã/chưa; ambiguity detection; guided fallback | ✅ Done |
| **Phase 4 — UX** | Tắt citation ticket; follow-up detection; bot image caching 5 phút | ✅ Done |
| **Phase 5 — Infra** | Dashboard React/Vite; WebSocket log stream; Docker compose; GitHub repo | ✅ Done |
| **Phase 6 — Production** | autossh tunnel; Redis session; rate limiting; webhook bot | 🔲 TODO |

---

## 14. Known Issues & TODO

| Issue | Mức độ | Giải pháp đề xuất |
|-------|--------|------------------|
| SSH tunnel không tự reconnect | **Cao** | `autossh` hoặc `systemd` service |
| Session memory mất khi restart | Trung bình | Redis (Phase 6) |
| `qdrant_data/` thuộc `root:root` | Thấp | `chown -R phungkien:phungkien qdrant_data/` |
| Bot dùng long polling | Thấp | Webhook khi cần scale |
| Chưa có rate limiting trên API | Trung bình | `slowapi` hoặc nginx limit |

---

## 15. Evaluation

Bộ test cases và kết quả nằm trong `reports/eval/`:

| File | Nội dung |
|------|---------|
| `bot_eval_report_455.json` | Đánh giá tự động 455 test cases |
| `bot_eval_report_455.md` | Report dạng Markdown |
| `source_miss_analysis_455.md` | Phân tích cases bị miss nguồn |
| `ticket_eval_cases_full.json` | Toàn bộ bộ test cases từ Redmine |

---

*GitHub: [github.com/phungkien402/ehc_ai_mvp](https://github.com/phungkien402/ehc_ai_mvp) — branch `main` (31/03/2026)*
