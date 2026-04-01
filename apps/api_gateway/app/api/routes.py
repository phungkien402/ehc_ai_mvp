"""API Gateway routes."""

import asyncio
import base64
import hashlib
import logging
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

import sys
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from app.schemas import (
    AskRequest,
    AskResponse,
    DashboardLogEntry,
    DashboardLogsResponse,
    DashboardOverviewResponse,
    HealthResponse,
    OCRRequest,
    OCRResponse,
    SourceInfo,
    WorkflowDiagramResponse,
)
from shared.py.utils.logging import setup_logging
from apps.agent_runtime.app.graph.workflow import get_agent
from shared.py.clients.qdrant_client import QdrantWrapper
from shared.py.clients.ollama_client import DEFAULT_OCR_PROMPT, OllamaEmbeddings, OllamaVision
from app.core.config import config
from app.core.session_memory import session_memory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

START_TIME = time.time()
LATENCY_HISTORY = deque(maxlen=100)
LOG_LINE_PATTERN = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\s-\s"
    r"(?P<logger>.+?)\s-\s(?P<level>[A-Z]+)\s-\s"
    r"(?:[^-]+?\s-\s)?(?P<message>.*)$"
)

LOG_SOURCE_FILES = {
    "api": "logs/api.log",
    "agent": "logs/agent.log",
    "bot": "logs/bot.log",
}


def _resolve_log_file(log_source: Optional[str]) -> str:
    source = (log_source or "api").strip().lower()
    return LOG_SOURCE_FILES.get(source, LOG_SOURCE_FILES["api"])


def _read_last_lines(file_path: str, max_lines: int = 3000) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return list(deque(f, maxlen=max_lines))


def _parse_log_line(line: str, cursor: int) -> DashboardLogEntry:
    clean = line.rstrip("\n")
    match = LOG_LINE_PATTERN.match(clean)
    if not match:
        return DashboardLogEntry(
            cursor=cursor,
            timestamp="",
            logger="",
            level="",
            message=clean,
            raw=clean,
        )
    return DashboardLogEntry(
        cursor=cursor,
        timestamp=match.group("ts").strip(),
        logger=match.group("logger").strip(),
        level=match.group("level").strip(),
        message=match.group("message").strip(),
        raw=clean,
    )


def _get_active_models() -> list[str]:
    if config.MODEL_PROVIDER == "vllm":
        configured = [
            os.getenv("OLLAMA_LLM_MODEL", ""),
            os.getenv("OLLAMA_VISION_MODEL", ""),
            os.getenv("OLLAMA_EMBEDDING_MODEL", ""),
        ]
        return [name for name in dict.fromkeys(configured) if name]

    try:
        response = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/ps",
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
    except Exception:
        return []


def _tail_log_file(log_path: str, from_offset: int) -> tuple[list[str], int]:
    path = Path(log_path)
    if not path.exists():
        return [], 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if from_offset > file_size:
            from_offset = 0
        f.seek(from_offset)
        lines = f.readlines()
        new_offset = f.tell()
    return lines, new_offset


def _sticky_rollout_key(request: AskRequest, session_key: str) -> str:
    sticky = (config.ROLLOUT_STICKY_KEY or "session_id").strip().lower()
    if sticky == "user_id":
        return (request.user_id or session_key or "anonymous").strip() or "anonymous"
    if sticky == "channel_user":
        channel = (request.channel or "api").strip() or "api"
        user_id = (request.user_id or "anonymous").strip() or "anonymous"
        return f"{channel}:{user_id}"
    return session_key or "anonymous"


def _resolve_provider_for_request(request: AskRequest, session_key: str) -> tuple[str, int]:
    base_provider = (config.MODEL_PROVIDER or "ollama").strip().lower()
    if base_provider == "vllm":
        return "vllm", 100

    if not config.ROLLOUT_ENABLED:
        return base_provider, -1

    pct = max(0.0, min(float(config.ROLLOUT_PERCENT_VLLM), 100.0))
    if pct <= 0:
        return base_provider, -1
    if pct >= 100:
        return "vllm", 0

    key = _sticky_rollout_key(request, session_key)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    provider = "vllm" if bucket < pct else base_provider
    return provider, bucket


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an answer from FAQ.
    
    Request:
        query: str — Question in Vietnamese
        image_base64: Optional[str] — Base64-encoded image (screenshot)
    
    Response:
        answer: str — Generated answer
        sources: List[SourceInfo] — Sources used
        image_urls: List[str] — Attachment URLs from FAQ
        error: Optional[str] — Error message if any
        execution_time_ms: float — Total execution time
    """
    
    start_time = time.time()
    
    try:
        # Decode image if provided
        image_bytes = None
        if request.image_base64:
            try:
                image_bytes = base64.b64decode(request.image_base64)
            except Exception as e:
                logger.error(f"Failed to decode image: {e}")
                # Continue without image
        
        # Build session identity for short-term chat memory.
        session_key = request.session_id
        if not session_key:
            channel = (request.channel or "api").strip() or "api"
            user_id = (request.user_id or "anonymous").strip() or "anonymous"
            session_key = f"{channel}:{user_id}"

        selected_provider, rollout_bucket = _resolve_provider_for_request(request, session_key)

        history_context, last_source_hint = session_memory.build_context(session_key)
        enriched_query = session_memory.enrich_query(session_key, request.query)

        # Invoke workflow
        agent = get_agent()
        
        state = {
            "messages": [],
            "trace_id": uuid.uuid4().hex[:8],
            "model_provider": selected_provider,
            "rollout_bucket": rollout_bucket,
            "query_text": enriched_query,
            "image_bytes": image_bytes,
            "conversation_context": history_context,
            "last_source_hint": last_source_hint,
            "cleaned_query": "",
            "ocr_text": None,
            "merged_query": "",
            "active_query": "",
            "rewritten_query": None,
            "rewrite_attempts": 0,
            "max_rewrite_attempts": config.MAX_REWRITE_ATTEMPTS,
            "retrieved_chunks": [],
            "is_relevant": "no",
            "grading_reason": None,
            "retrieval_debug": [],
            "last_tool_name": None,
            "final_answer": "",
            "sources": [],
            "image_urls": [],
            "error": None,
            "duration_seconds": 0.0
        }
        
        logger.info(
            "Processing query: %s | provider=%s | rollout_bucket=%s",
            request.query[:100],
            selected_provider,
            rollout_bucket,
        )
        result = await agent.ainvoke(state)
        
        # Format response
        sources = [
            SourceInfo(
                issue_id=s.get("issue_id", ""),
                snippet=s.get("snippet", ""),
                url=s.get("url", ""),
                score=s.get("score", 0)
            )
            for s in result.get("sources", [])
        ]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        LATENCY_HISTORY.append(elapsed_ms)

        session_memory.remember_turn(
            session_key=session_key,
            user_text=request.query,
            answer_text=result.get("final_answer", ""),
            sources=result.get("sources", []),
        )

        return AskResponse(
            answer=result.get("final_answer", ""),
            sources=sources,
            image_urls=result.get("image_urls", []),
            error=result.get("error"),
            execution_time_ms=elapsed_ms,
            is_relevant=result.get("is_relevant"),
            rewrite_attempts=result.get("rewrite_attempts", 0),
            rewritten_query=result.get("rewritten_query"),
            grading_reason=result.get("grading_reason"),
            ocr_text=result.get("ocr_text"),
            provider=result.get("model_provider", selected_provider),
            rollout_bucket=rollout_bucket if rollout_bucket >= 0 else None,
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        elapsed_ms = (time.time() - start_time) * 1000
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", response_model=OCRResponse)
async def test_ocr(request: OCRRequest):
    """Run standalone OCR against an uploaded screenshot without retrieval/generation."""

    start_time = time.time()

    try:
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}") from e

        selected_model = request.model or config.OLLAMA_VISION_MODEL
        vision_base_url = config.VLLM_VISION_URL if config.MODEL_PROVIDER == "vllm" else config.OLLAMA_BASE_URL
        vision_client = OllamaVision(base_url=vision_base_url, model=selected_model, timeout=120)
        ocr_text = vision_client.extract_text_from_image(
            image_bytes=image_bytes,
            prompt=DEFAULT_OCR_PROMPT,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        return OCRResponse(
            ocr_text=ocr_text,
            model=selected_model,
            execution_time_ms=elapsed_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Standalone OCR failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check system health: Qdrant and Ollama connectivity.
    """
    
    qdrant_ok = False
    ollama_ok = False
    
    try:
        qdrant_wrapper = QdrantWrapper(url=config.QDRANT_URL)
        info = qdrant_wrapper.collection_info(config.QDRANT_COLLECTION)
        qdrant_ok = bool(info)
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
    
    try:
        embedding_base_url = config.VLLM_EMBEDDING_URL if config.MODEL_PROVIDER == "vllm" else config.OLLAMA_BASE_URL
        embeddings = OllamaEmbeddings(
            base_url=embedding_base_url,
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3"),
            provider=config.MODEL_PROVIDER,
        )
        vector = embeddings.embed_query("test")
        ollama_ok = len(vector) == 1024
    except Exception as e:
        logger.warning(f"Model provider health check failed: {e}")
    
    status = "ok" if (qdrant_ok and ollama_ok) else "degraded"
    provider_label = config.MODEL_PROVIDER.upper()
    message = f"Qdrant: {'✓' if qdrant_ok else '✗'}, {provider_label}: {'✓' if ollama_ok else '✗'}"
    
    return HealthResponse(
        status=status,
        qdrant_ok=qdrant_ok,
        ollama_ok=ollama_ok,
        message=message
    )


@router.get("/dashboard/overview", response_model=DashboardOverviewResponse)
async def dashboard_overview():
    """Return high-level runtime status for dashboard cards."""

    qdrant_ok = False
    ollama_ok = False

    try:
        qdrant_wrapper = QdrantWrapper(url=config.QDRANT_URL)
        qdrant_ok = bool(qdrant_wrapper.collection_info(config.QDRANT_COLLECTION))
    except Exception:
        qdrant_ok = False

    try:
        if config.MODEL_PROVIDER == "vllm":
            response = requests.get(f"{config.VLLM_LLM_URL.rstrip('/')}/v1/models", timeout=5)
        else:
            response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        ollama_ok = True
    except Exception:
        ollama_ok = False

    return DashboardOverviewResponse(
        api_status="ok",
        uptime_seconds=int(time.time() - START_TIME),
        qdrant_ok=qdrant_ok,
        ollama_ok=ollama_ok,
        active_models=_get_active_models(),
        recent_latency_ms=(sum(LATENCY_HISTORY) / len(LATENCY_HISTORY)) if LATENCY_HISTORY else None,
    )


@router.get("/dashboard/logs", response_model=DashboardLogsResponse)
async def dashboard_logs(
    log_source: Optional[str] = "api",
    source: Optional[str] = None,
    level: Optional[str] = None,
    limit: int = 100,
    cursor: int = 0,
):
    """Return parsed API log entries with lightweight cursor paging."""

    limit = max(1, min(limit, 500))
    level_filter = level.upper() if level else None
    log_path = _resolve_log_file(log_source)

    lines = _read_last_lines(log_path, max_lines=3000)
    parsed = [_parse_log_line(line, idx + 1) for idx, line in enumerate(lines)]

    filtered = []
    for entry in parsed:
        if source and source.lower() not in entry.logger.lower():
            continue
        if level_filter and entry.level != level_filter:
            continue
        filtered.append(entry)

    newer = [entry for entry in filtered if entry.cursor > cursor]
    entries = newer[-limit:]
    next_cursor = filtered[-1].cursor if filtered else cursor

    return DashboardLogsResponse(entries=entries, next_cursor=next_cursor)


@router.websocket("/dashboard/logs/stream")
async def dashboard_logs_stream(websocket: WebSocket):
    """Stream appended API log lines; frontend can filter client-side."""

    await websocket.accept()
    log_source = websocket.query_params.get("log_source", "api")
    log_path = _resolve_log_file(log_source)
    offset = 0
    try:
        while True:
            lines, offset = _tail_log_file(log_path, from_offset=offset)
            for line in lines:
                entry = _parse_log_line(line, cursor=int(time.time() * 1000))
                await websocket.send_json(entry.model_dump())
            await websocket.send_json({"type": "heartbeat", "ts": int(time.time())})
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("Dashboard logs stream disconnected")
    except Exception as e:
        logger.warning(f"Dashboard logs stream error: {e}")
        await websocket.close()


@router.get("/dashboard/workflow", response_model=WorkflowDiagramResponse)
async def dashboard_workflow():
    """Serve Mermaid workflow source for dashboard visualization tab."""

    workflow_path = Path("docs/workflow.mmd")
    if not workflow_path.exists():
        raise HTTPException(status_code=404, detail="Workflow diagram not found")

    content = workflow_path.read_text(encoding="utf-8")
    modified = int(workflow_path.stat().st_mtime * 1000)
    return WorkflowDiagramResponse(content=content, last_modified_epoch_ms=modified)
