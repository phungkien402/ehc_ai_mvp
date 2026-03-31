"""Conditional routing helpers for the Self-RAG workflow."""

import logging
import re
import time

import requests

from langgraph.graph import END

from apps.agent_runtime.app.core.config import config
from apps.agent_runtime.app.graph.state import WorkflowState
from apps.agent_runtime.app.graph.tracing import emit_trace

logger = logging.getLogger(__name__)

_LLM_HEALTH_CACHE_TTL_SECONDS = 5.0
_llm_health_cached_at = 0.0
_llm_health_cached_value = False


# Single-word tokens that must match as a whole word (prevents "hi" matching inside "nhiêu", etc.)
_SMALL_TALK_SINGLE = {"chao", "chào", "hello", "hi", "ok", "oke"}
# Multi-word phrases where substring match is safe
_SMALL_TALK_PHRASES = [
    "xin chao", "xin chào",
    "cảm ơn", "cam on",
    "thanks", "thank you",
    "bạn là ai", "ban la ai",
]


def _is_small_talk(query: str) -> bool:
    lowered = (query or "").strip().lower()
    if not lowered:
        return False

    # Split on whitespace/punctuation for whole-word comparison
    words = set(re.split(r"[\s,.!?;:\-\+/\\]+", lowered))
    if any(token in words for token in _SMALL_TALK_SINGLE):
        return True
    return any(phrase in lowered for phrase in _SMALL_TALK_PHRASES)


def _is_llm_available() -> bool:
    global _llm_health_cached_at
    global _llm_health_cached_value

    now = time.time()
    if (now - _llm_health_cached_at) < _LLM_HEALTH_CACHE_TTL_SECONDS:
        return _llm_health_cached_value

    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
        _llm_health_cached_value = response.status_code == 200
    except Exception:
        _llm_health_cached_value = False

    _llm_health_cached_at = now
    return _llm_health_cached_value


def route_after_ocr(state: WorkflowState) -> str:
    """Route from OCR stage to natural chat, agent flow, or degraded LLM-unavailable path."""

    emit_trace(logger, "route_after_ocr", "start", state)

    query_text = state.get("query_text", "")
    if _is_small_talk(query_text):
        emit_trace(logger, "route_after_ocr", "end", state, route="natural_chat")
        return "natural_chat"

    if not _is_llm_available():
        logger.warning("LLM endpoint unavailable; routing to llm_unavailable node")
        emit_trace(logger, "route_after_ocr", "end", state, route="llm_unavailable")
        return "llm_unavailable"

    emit_trace(logger, "route_after_ocr", "end", state, route="agent")
    return "agent"


def should_continue(state: WorkflowState) -> str:
    """Route from the agent node to tools or final answer generation."""

    emit_trace(logger, "should_continue", "start", state)

    messages = list(state.get("messages", []))
    if not messages:
        logger.info("Agent produced no messages; finishing")
        emit_trace(logger, "should_continue", "end", state, route="end", reason="no_messages")
        return "end"

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if tool_calls:
        logger.info("Agent requested %s tool call(s)", len(tool_calls))
        emit_trace(logger, "should_continue", "end", state, route="continue", tool_calls=len(tool_calls))
        return "continue"

    logger.info("Agent decided to respond without tools")
    emit_trace(logger, "should_continue", "end", state, route="end", reason="direct_answer")
    return "end"


def route_after_grading(state: WorkflowState) -> str:
    """Route from grading to rewrite loop or final synthesis."""

    emit_trace(logger, "route_after_grading", "start", state, decision=state.get("is_relevant", "no"))

    if state.get("is_relevant") == "yes":
        logger.info("Document grader accepted retrieval; generating final answer")
        emit_trace(logger, "route_after_grading", "end", state, route="generate")
        return "generate"

    rewrite_attempts = state.get("rewrite_attempts", 0)
    max_attempts = state.get("max_rewrite_attempts", 0)
    if rewrite_attempts < max_attempts:
        logger.info("Document grader rejected retrieval; rewriting query (attempt %s/%s)", rewrite_attempts + 1, max_attempts)
        emit_trace(logger, "route_after_grading", "end", state, route="rewrite", next_attempt=rewrite_attempts + 1, max_attempts=max_attempts)
        return "rewrite"

    logger.info("Rewrite budget exhausted; generating fallback-style final answer")
    emit_trace(logger, "route_after_grading", "end", state, route="generate", reason="rewrite_budget_exhausted")
    return "generate"


__all__ = ["END", "route_after_ocr", "should_continue", "route_after_grading"]
