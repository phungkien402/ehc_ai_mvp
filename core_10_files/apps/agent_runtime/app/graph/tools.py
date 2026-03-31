"""Retrieval tools and helpers for the Self-RAG workflow."""

import json
import logging
import re
import sys
from typing import List

from langchain_core.tools import tool

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from apps.agent_runtime.app.core.config import config
from apps.agent_runtime.app.graph.tracing import emit_trace
from shared.py.clients.ollama_client import OllamaEmbeddings
from shared.py.clients.qdrant_client import QdrantWrapper
from shared.py.utils.text import normalize_vietnamese

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "cho", "cua", "voi", "tren", "duoc", "khong", "dang", "toi", "ban", "em", "anh", "chi", "bsi",
    "bac", "si", "giu", "giup", "lam", "nay", "kia", "mot", "cac", "nhung", "khi", "neu", "vao",
    "trong", "phan", "mem", "he", "thong", "loi", "sao", "the", "nao", "roi", "nua", "de", "va",
}


def _tokenize_text(text: str) -> set[str]:
    normalized = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return {token for token in normalized.split() if len(token) >= 3}


def lexical_overlap_ratio(query: str, content: str) -> float:
    query_tokens = {token for token in _tokenize_text(query) if token not in _STOPWORDS}
    if not query_tokens:
        return 0.0
    content_tokens = _tokenize_text(content)
    return len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))


def search_faq_chunks(query: str) -> List[dict]:
    """Run vector retrieval without early threshold gating so the grader can decide relevance."""

    cleaned_query = normalize_vietnamese((query or "").strip())
    if not cleaned_query:
        return []

    embeddings = OllamaEmbeddings(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_EMBEDDING_MODEL,
        timeout=60,
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

    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.warning("Failed to parse tool payload as JSON")
    return {"query": "", "chunks": [], "error": "Malformed tool payload"}


@tool
def search_faq_tool(query: str) -> str:
    """Tra cứu các FAQ helpdesk liên quan tới câu hỏi hiện tại của người dùng."""

    try:
        emit_trace(logger, "tools", "start", None, query=query[:120])
        chunks = search_faq_chunks(query)
        emit_trace(logger, "tools", "end", None, chunks=len(chunks), top_issue=(chunks[0].get("issue_id") if chunks else None))
        return json.dumps(
            {
                "query": query,
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