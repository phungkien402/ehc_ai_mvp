"""Conditional routing helpers for the Self-RAG workflow."""

import logging

from langgraph.graph import END

from apps.agent_runtime.app.graph.state import WorkflowState
from apps.agent_runtime.app.graph.tracing import emit_trace

logger = logging.getLogger(__name__)


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


__all__ = ["END", "should_continue", "route_after_grading"]
