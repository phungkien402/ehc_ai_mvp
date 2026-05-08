# EHC AI MVP — Claude Code Instructions

## Project Context

This is a Vietnamese internal IT-helpdesk chatbot using Self-RAG (agentic RAG with LangGraph).
Key components:
- `apps/api_gateway/` — FastAPI on port 8000 (`/api/v1/ask`, `/health`, `/dashboard/*`)
- `apps/agent_runtime/app/graph/` — LangGraph workflow: OCR → agent → tools → grade → generate
- `pipelines/ingestion/` — CLI: Redmine FAQ + DOCX → chunk → embed → Qdrant
- `shared/py/` — shared clients (Ollama, Qdrant) and utilities

The main business logic is in `apps/agent_runtime/app/graph/nodes.py`.
LLM provider is set via `MODEL_PROVIDER` env var: `ollama` (local) or `vllm` (OpenAI-compatible).

## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

## Python Coding Style

- Follow **PEP 8**. Use **black** for formatting, **ruff** for linting, **isort** for imports.
- Add **type annotations** on all function signatures.
- Prefer `dataclass(frozen=True)` or `TypedDict` for structured data (e.g., `WorkflowState`).
- No unused imports. No wildcard imports (`from x import *`).
- Keep functions focused; split at ~80 lines.

## Security Rules (MANDATORY)

**Never hardcode secrets.** This project uses:
- `REDMINE_API_KEY`, `QDRANT_URL`, `OLLAMA_BASE_URL`, `VLLM_API_KEY`, etc.
- All secrets MUST come from environment variables via `config` (see `app/core/config.py`).
- If you discover a hardcoded secret: STOP, remove it, rotate it.

Before any commit verify:
- [ ] No API keys, tokens, or passwords in source code
- [ ] No secrets in log statements (use `.setLevel(WARNING)` for noisy libs)
- [ ] All user inputs validated at API boundary (FastAPI request schemas handle this)
- [ ] Error responses don't leak internal paths or stack traces to callers

## LangGraph Patterns

- Every node function signature: `def node_name(state: WorkflowState) -> dict:`
- Return only the state keys you mutate — LangGraph merges them.
- Always call `emit_trace(logger, "node_name", "start"|"end", state, ...)` at node entry/exit.
- Nodes must be pure functions — no global side effects, no direct `asyncio.run()`.
- Tool functions registered in `tools.py` must be synchronous (LangGraph ToolNode handles async wrapping).

## Git Workflow

Commit message format:
```
<type>: <description>

<optional body>
```
Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

Examples:
- `feat: add reranking for module_doc chunks`
- `fix: handle OCR timeout in extract_ocr_if_image node`

## Testing

- Tests live in `tests/`. Run with: `pytest tests/`
- Unit tests for graph routing go in `tests/agent/`.
- Do not mock Qdrant in integration tests — use a real test collection.
- Each new node or router function needs at least one test.

## Do Not

- Do not modify `shared/py/` without checking all callers in `apps/` and `pipelines/`.
- Do not add `sys.path.insert` calls in new files — add proper package structure instead.
- Do not log full `image_bytes` content or raw base64 strings.
- Do not introduce `asyncio.run()` inside node functions — keep them sync or fully async.

---

## Behavioral Guidelines (Karpathy Principles)

**Tradeoff:** These bias toward caution over speed. Use judgment for trivial tasks.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Add feature X" → define what correct output looks like, then implement.
- For multi-step tasks, state a brief plan with a verify step for each.
