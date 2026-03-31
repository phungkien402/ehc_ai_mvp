"""Lightweight in-process session memory for short chat context."""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SessionState:
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    last_issue_id: str | None = None
    last_issue_snippet: str | None = None
    last_issue_url: str | None = None
    updated_at: float = field(default_factory=time.time)


class SessionMemoryStore:
    def __init__(self) -> None:
        self._store: dict[str, SessionState] = {}
        self._lock = Lock()
        self._ttl_seconds = int(os.getenv("SESSION_MEMORY_TTL_SECONDS", "7200"))
        self._max_sessions = int(os.getenv("SESSION_MEMORY_MAX_SESSIONS", "2000"))

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items() if (now - v.updated_at) > self._ttl_seconds]
        for key in expired:
            self._store.pop(key, None)

    def _maybe_trim_size(self) -> None:
        if len(self._store) <= self._max_sessions:
            return
        # Drop oldest sessions first when cap exceeded.
        ordered = sorted(self._store.items(), key=lambda kv: kv[1].updated_at)
        to_drop = len(self._store) - self._max_sessions
        for key, _ in ordered[:to_drop]:
            self._store.pop(key, None)

    def _get_or_create(self, session_key: str) -> SessionState:
        state = self._store.get(session_key)
        if state is None:
            state = SessionState()
            self._store[session_key] = state
        state.updated_at = time.time()
        return state

    def build_context(self, session_key: str) -> tuple[str, str]:
        with self._lock:
            self._evict_expired()
            state = self._store.get(session_key)
            if state is None:
                return "", ""

            history_lines = []
            for item in list(state.history)[-6:]:
                role = item.get("role", "user")
                text = (item.get("text", "") or "").strip()
                if text:
                    history_lines.append(f"{role}: {text}")
            history_text = "\n".join(history_lines)

            if state.last_issue_id:
                snippet = (state.last_issue_snippet or "").strip()
                source_hint = f"ticket #{state.last_issue_id}"
                if snippet:
                    source_hint += f" | {snippet[:220]}"
            else:
                source_hint = ""

            return history_text, source_hint

    def remember_turn(self, session_key: str, user_text: str, answer_text: str, sources: list[dict] | None) -> None:
        with self._lock:
            self._evict_expired()
            state = self._get_or_create(session_key)

            user_text = (user_text or "").strip()
            answer_text = (answer_text or "").strip()
            if user_text:
                state.history.append({"role": "user", "text": user_text[:500]})
            if answer_text:
                state.history.append({"role": "assistant", "text": answer_text[:900]})

            if sources:
                top = sources[0] or {}
                issue_id = str(top.get("issue_id", "") or "").strip()
                if issue_id:
                    state.last_issue_id = issue_id
                    state.last_issue_snippet = str(top.get("snippet", "") or "")
                    state.last_issue_url = str(top.get("url", "") or "")

            state.updated_at = time.time()
            self._maybe_trim_size()

    def enrich_query(self, session_key: str, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return query

        lowered = query.lower()
        followup = bool(
            re.search(r"\b(chi tiết|cụ thể|nói rõ|giải thích thêm|thế còn|cái đó|vậy còn|nữa)\b", lowered)
            or len(lowered.split()) <= 5
        )
        if not followup:
            return query

        with self._lock:
            self._evict_expired()
            state = self._store.get(session_key)
            if state is None or not state.last_issue_id:
                return query

            snippet = (state.last_issue_snippet or "").strip()
            if snippet:
                return (
                    f"{query}\n"
                    f"Ngữ cảnh hội thoại gần nhất: ticket #{state.last_issue_id} - {snippet[:220]}"
                )
            return f"{query}\nNgữ cảnh hội thoại gần nhất: ticket #{state.last_issue_id}"


session_memory = SessionMemoryStore()
