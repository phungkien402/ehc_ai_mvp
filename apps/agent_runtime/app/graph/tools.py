"""Retrieval tools and helpers for the Self-RAG workflow."""

import json
import logging
import re
import sys
from contextvars import ContextVar
from typing import List

from langchain_core.tools import tool

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from apps.agent_runtime.app.core.config import config
from apps.agent_runtime.app.graph.tracing import emit_trace
from shared.py.clients.ollama_client import OllamaEmbeddings
from shared.py.clients.qdrant_client import QdrantWrapper
from shared.py.utils.text import normalize_vietnamese

logger = logging.getLogger(__name__)
_CURRENT_MODEL_PROVIDER: ContextVar[str | None] = ContextVar("current_model_provider", default=None)

_STOPWORDS = {
    "cho", "cua", "voi", "tren", "duoc", "khong", "dang", "toi", "ban", "em", "anh", "chi", "bsi",
    "bac", "si", "giu", "giup", "lam", "nay", "kia", "mot", "cac", "nhung", "khi", "neu", "vao",
    "trong", "phan", "mem", "he", "thong", "loi", "sao", "the", "nao", "roi", "nua", "de", "va",
}

# Keep negation/status markers out of stopword removal to preserve intent contrast.
_STATUS_MARKERS = {"da", "dã", "đã", "chua", "chưa"}


def _tokenize_text(text: str) -> set[str]:
    normalized = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return {
        token
        for token in normalized.split()
        if len(token) >= 3 or token in _STATUS_MARKERS
    }


def _extract_status_windows(text: str) -> dict[tuple[str, ...], list[int]]:
    """Extract short status windows after markers like da/chua for contrast-aware rerank."""
    normalized = re.sub(r"[^\w\s]", " ", (text or "").lower())
    words = [w for w in normalized.split() if w]

    status_map: dict[tuple[str, ...], list[int]] = {}
    for idx, token in enumerate(words):
        if token in {"da", "dã", "đã"}:
            sign = 1
        elif token in {"chua", "chưa"}:
            sign = -1
        else:
            continue

        tail = tuple(words[idx + 1 : idx + 4])
        if not tail:
            continue
        status_map.setdefault(tail, []).append(sign)

    return status_map


def _status_alignment_bonus(query: str, content: str) -> float:
    """Reward same-status matches and penalize opposite-status matches on similar tails."""
    q_windows = _extract_status_windows(query)
    c_windows = _extract_status_windows(content)
    if not q_windows or not c_windows:
        return 0.0

    bonus = 0.0
    for tail, q_signs in q_windows.items():
        c_signs = c_windows.get(tail, [])
        if not c_signs:
            continue
        for q_sign in q_signs:
            for c_sign in c_signs:
                if q_sign == c_sign:
                    bonus += 0.06
                else:
                    bonus -= 0.10

    return max(-0.2, min(0.2, bonus))


def lexical_overlap_ratio(query: str, content: str) -> float:
    # Preserve status markers (da/chua) even if they appear in stopword-like lists.
    query_tokens = {
        token for token in _tokenize_text(query)
        if (token not in _STOPWORDS) or (token in _STATUS_MARKERS)
    }
    if not query_tokens:
        return 0.0
    content_tokens = _tokenize_text(content)
    base = len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))
    adjusted = base + _status_alignment_bonus(query, content)
    return max(0.0, min(1.0, adjusted))


def set_runtime_model_provider(provider: str | None) -> None:
    if provider:
        _CURRENT_MODEL_PROVIDER.set(provider.strip().lower())


def get_runtime_model_provider() -> str:
    return (_CURRENT_MODEL_PROVIDER.get() or config.MODEL_PROVIDER or "ollama").strip().lower()


def search_faq_chunks(query: str) -> List[dict]:
    """Run vector retrieval without early threshold gating so the grader can decide relevance."""

    cleaned_query = normalize_vietnamese((query or "").strip())
    if not cleaned_query:
        return []

    provider = get_runtime_model_provider()
    if provider == "vllm":
        embedding_base_url = config.VLLM_EMBEDDING_URL
    else:
        embedding_base_url = config.OLLAMA_BASE_URL

    embeddings = OllamaEmbeddings(
        base_url=embedding_base_url,
        model=config.OLLAMA_EMBEDDING_MODEL,
        timeout=60,
        provider=provider,
    )
    query_vector = embeddings.embed_query(cleaned_query)

    qdrant = QdrantWrapper(url=config.QDRANT_URL)
    chunks = qdrant.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=config.RETRIEVAL_TOP_K,
        score_threshold=None,
    )

    for chunk in chunks:
        chunk["lexical_score"] = round(lexical_overlap_ratio(cleaned_query, chunk.get("content_brief", "")), 4)

    logger.info("search_faq_chunks returned %s chunks", len(chunks))
    return chunks


def parse_tool_payload(payload: str) -> dict:
    """Parse serialized retrieval payload emitted by tools."""

    raw = (payload or "").strip()
    if not raw:
        return {"query": "", "chunks": [], "error": "Empty tool payload"}

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Recover from markdown fences or prefixed/suffixed text by extracting the
    # most likely JSON object block that contains retrieval fields.
    candidates = []
    if "```" in raw:
        for block in re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE):
            candidates.append(block)
    candidates.extend(re.findall(r"\{[\s\S]*\}", raw))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                if "chunks" in data or "query" in data:
                    return data
        except Exception:
            continue

    logger.warning("Failed to parse tool payload as JSON")
    return {"query": "", "chunks": [], "error": "Malformed tool payload"}


@tool
def search_faq_tool(query: str) -> str:
    """Tra cứu các FAQ helpdesk liên quan tới câu hỏi hiện tại của người dùng."""

    try:
        provider = get_runtime_model_provider()
        emit_trace(logger, "tools", "start", None, query=query[:120])
        chunks = search_faq_chunks(query)
        emit_trace(logger, "tools", "end", None, chunks=len(chunks), top_issue=(chunks[0].get("issue_id") if chunks else None))
        return json.dumps(
            {
                "query": query,
                "provider": provider,
                "chunks": chunks,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("search_faq_tool failed: %s", exc)
        emit_trace(logger, "tools", "end", None, chunks=0, error=str(exc))
        return json.dumps(
            {
                "query": query,
                "chunks": [],
                "error": str(exc),
            },
            ensure_ascii=False,
        )


available_tools = [search_faq_tool]