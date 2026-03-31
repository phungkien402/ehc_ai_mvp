"""Structured trace helpers for workflow observability."""

from __future__ import annotations

import logging
from typing import Any, Mapping


def _clean(value: Any) -> str:
    text = str(value)
    return text.replace("|", "/").replace("\n", " ").strip()


def emit_trace(
    logger: logging.Logger,
    node: str,
    phase: str,
    state: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {
        "node": node,
        "phase": phase,
    }

    if state:
        trace_id = state.get("trace_id")
        if trace_id:
            payload["trace_id"] = trace_id
        rewrite_attempts = state.get("rewrite_attempts")
        if rewrite_attempts is not None:
            payload["rewrite_attempts"] = rewrite_attempts

    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value

    parts = [f"{key}={_clean(value)}" for key, value in payload.items()]
    logger.info("TRACE|%s", "|".join(parts))
