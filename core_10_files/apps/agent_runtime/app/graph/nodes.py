"""LangGraph nodes implementation for the Self-RAG workflow."""

import json
import logging
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from apps.agent_runtime.app.core.config import config
from apps.agent_runtime.app.graph.state import WorkflowState
from apps.agent_runtime.app.graph.tracing import emit_trace
from apps.agent_runtime.app.graph.tools import lexical_overlap_ratio, parse_tool_payload
from shared.py.clients.ollama_client import DEFAULT_OCR_PROMPT, OllamaChat, OllamaVision
from shared.py.utils.text import merge_query_with_ocr, normalize_vietnamese

logger = logging.getLogger(__name__)


def _build_agent_model() -> ChatOllama:
    return ChatOllama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=180,
    )


def _latest_tool_message(state: WorkflowState) -> ToolMessage | None:
    for message in reversed(list(state.get("messages", []))):
        if isinstance(message, ToolMessage):
            return message
    return None


def _latest_ai_text(state: WorkflowState) -> str:
    for message in reversed(list(state.get("messages", []))):
        if isinstance(message, AIMessage):
            content = message.content if isinstance(message.content, str) else str(message.content)
            return content.strip()
    return ""


def _append_retrieval_debug(state: WorkflowState, item: dict) -> list[dict]:
    debug_items = list(state.get("retrieval_debug", []))
    debug_items.append(item)
    return debug_items


def _parse_grader_output(raw_output: str) -> tuple[str, str | None]:
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            is_relevant = str(parsed.get("is_relevant", "")).strip().lower()
            if is_relevant in {"yes", "no"}:
                return is_relevant, str(parsed.get("reason", "")).strip() or None
    except Exception:
        pass

    lowered = raw_output.lower()
    if "yes" in lowered:
        return "yes", raw_output.strip() or None
    if "no" in lowered:
        return "no", raw_output.strip() or None
    return "", None


def _confidence_label(score: float) -> str:
    """Convert a vector similarity score to a human-readable confidence label."""
    if score >= 0.70:
        return "Cao"
    if score >= 0.55:
        return "Trung bình"
    return "Thấp"


def _rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank chunks by combined (vector * 0.6 + lexical * 0.4) score."""
    scored = []
    for chunk in chunks:
        vec = float(chunk.get("score", 0.0) or 0.0)
        lex = lexical_overlap_ratio(query, chunk.get("content_brief", ""))
        combined = vec * 0.6 + lex * 0.4
        scored.append((combined, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _best_chunk_metrics(query: str, chunks: list[dict]) -> tuple[float, float]:
    if not chunks:
        return 0.0, 0.0
    best_chunk = chunks[0]
    best_score = float(best_chunk.get("score", 0.0) or 0.0)
    lexical_score = float(best_chunk.get("lexical_score", 0.0) or lexical_overlap_ratio(query, best_chunk.get("content_brief", "")))
    return best_score, lexical_score


def extract_ocr_if_image(state: WorkflowState) -> dict:
    """Extract text from image if present."""
    cleaned_query = normalize_vietnamese(state.get("query_text", "").strip())
    emit_trace(logger, "extract_ocr_if_image", "start", state, has_image=bool(state.get("image_bytes")))
    
    if state.get("image_bytes") is None:
        logger.info("No image; skipping OCR")
        emit_trace(logger, "extract_ocr_if_image", "end", state, outcome="skipped")
        return {
            "cleaned_query": cleaned_query,
            "ocr_text": None,
            "merged_query": cleaned_query,
            "active_query": cleaned_query,
        }
    
    try:
        vision_client = OllamaVision(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_VISION_MODEL,
            timeout=90
        )
        
        logger.info("Calling vision model for OCR...")
        ocr_text = vision_client.extract_text_from_image(
            image_bytes=state["image_bytes"],
            prompt=DEFAULT_OCR_PROMPT,
        )
        
        merged = merge_query_with_ocr(state["cleaned_query"], ocr_text)
        logger.info(f"OCR result: {len(ocr_text)} chars; merged query: {len(merged)} chars")
        emit_trace(logger, "extract_ocr_if_image", "end", state, outcome="ocr_ok", ocr_chars=len(ocr_text))
        
        return {
            "cleaned_query": cleaned_query,
            "ocr_text": ocr_text,
            "merged_query": merged,
            "active_query": merged,
        }
    
    except TimeoutError:
        logger.warning("OCR timeout; using text-only query")
        emit_trace(logger, "extract_ocr_if_image", "end", state, outcome="ocr_timeout")
        return {
            "cleaned_query": cleaned_query,
            "ocr_text": None,
            "merged_query": cleaned_query,
            "active_query": cleaned_query,
            "error": "OCR timeout"
        }
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        emit_trace(logger, "extract_ocr_if_image", "end", state, outcome="ocr_failed", error=str(e))
        return {
            "cleaned_query": cleaned_query,
            "ocr_text": None,
            "merged_query": cleaned_query,
            "active_query": cleaned_query,
            "error": f"OCR failed: {e}"
        }


def call_agent(state: WorkflowState) -> dict:
    """Let the agent decide whether to call retrieval tools or answer directly."""

    active_query = state.get("active_query") or state.get("rewritten_query") or state.get("merged_query") or state.get("query_text", "")
    emit_trace(logger, "agent", "start", state, query=active_query[:120])

    llm_with_tools = _build_agent_model().bind_tools(__import__("apps.agent_runtime.app.graph.tools", fromlist=["available_tools"]).available_tools)
    ocr_text = state.get("ocr_text") or "Khong co"

    system_prompt = (
        "Ban la agent Self-RAG cho helpdesk noi bo. "
        "Voi cau hoi nghiep vu ho tro, uu tien goi tool search_faq_tool truoc khi ket luan. "
        "Chi tra loi truc tiep neu do la chao hoi ngan, hoac cau hoi khong can tra cuu. "
        "Khong duoc tu che nguon; neu can tim lai, hay dung tool."
    )

    messages = list(state.get("messages", []))
    if not messages:
        user_message = HumanMessage(
            content=(
                f"Cau hoi goc cua user: {state.get('query_text', '')}\n"
                f"Truy van hien tai de tim kiem: {active_query}\n"
                f"Noi dung OCR: {ocr_text}"
            )
        )
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt), user_message])
        logger.info("Agent generated first-step response")
        emit_trace(logger, "agent", "end", state, tool_calls=len(getattr(response, "tool_calls", None) or []), outcome="first_step")
        return {"messages": [user_message, response]}

    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages[-8:])
    logger.info("Agent generated follow-up response")
    emit_trace(logger, "agent", "end", state, tool_calls=len(getattr(response, "tool_calls", None) or []), outcome="follow_up")
    return {"messages": [response]}


def grade_documents(state: WorkflowState) -> dict:
    """Evaluate whether retrieved documents are relevant enough to answer the query."""

    emit_trace(logger, "grade_documents", "start", state)

    tool_message = _latest_tool_message(state)
    if tool_message is None:
        logger.warning("grade_documents called without a tool result")
        emit_trace(logger, "grade_documents", "end", state, decision="no", reason="missing_tool_result")
        return {
            "retrieved_chunks": [],
            "is_relevant": "no",
            "grading_reason": "Agent khong tra ve ket qua tool.",
            "last_tool_name": None,
        }

    payload = parse_tool_payload(str(tool_message.content))
    chunks = payload.get("chunks", []) if isinstance(payload.get("chunks"), list) else []
    active_query = state.get("active_query") or payload.get("query") or state.get("merged_query") or state.get("query_text", "")

    if not chunks:
        reason = payload.get("error") or "Tool retrieval khong tim thay chunk phu hop."
        emit_trace(logger, "grade_documents", "end", state, decision="no", chunks=0, reason=reason)
        return {
            "retrieved_chunks": [],
            "is_relevant": "no",
            "grading_reason": reason,
            "last_tool_name": getattr(tool_message, "name", None),
            "retrieval_debug": _append_retrieval_debug(
                state,
                {"query": active_query, "decision": "no", "reason": reason, "chunks": 0},
            ),
        }

    context_lines = []
    for index, chunk in enumerate(chunks[:3], 1):
        # Use up to 300 chars of content_brief for grader context
        content_preview = chunk.get('content_brief', '')[:300]
        context_lines.append(
            f"{index}. issue_id={chunk.get('issue_id', 'unknown')}, score={float(chunk.get('score', 0)):.2f}\n   Noi dung: {content_preview}"
        )

    # Log chunk previews so we can debug what was graded
    for i, chunk in enumerate(chunks[:3], 1):
        logger.debug(
            "TRACE|node=grade_documents|phase=chunk_preview|rank=%d|issue_id=%s|score=%.4f|content=%s",
            i, chunk.get('issue_id', '?'), float(chunk.get('score', 0)),
            chunk.get('content_brief', '')[:200],
        )

    prompt = (
        "Ban la giam khao cho he thong RAG helpdesk noi bo. "
        "Tra ve JSON duy nhat voi schema {\"is_relevant\":\"yes|no\",\"reason\":\"...\"}.\n\n"
        f"Cau hoi user: {active_query}\n\n"
        "Tai lieu truy xuat:\n"
        f"{chr(10).join(context_lines)}\n\n"
        "Tra 'yes' neu bat ky tai lieu nao co lien quan den chu de cau hoi, du chi mot phan. "
        "Tra 'no' chi khi TAT CA tai lieu hoan toan khong lien quan den chu de cau hoi."
    )

    grader = OllamaChat(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_GRADER_MODEL,
        timeout=60,
    )

    grader_reason = None
    grade = ""
    raw_grader_output = ""
    try:
        raw_grader_output = grader.generate(
            prompt=prompt,
            system_prompt="Ban chi duoc tra ve JSON hop le, khong giai thich them.",
        )
        logger.info("TRACE|node=grade_documents|phase=llm_raw|output=%s", raw_grader_output[:300].replace("|", "/"))
        grade, grader_reason = _parse_grader_output(raw_grader_output)
    except Exception as exc:
        logger.warning("LLM grader failed, using heuristic fallback: %s", exc)

    best_score, lexical_score = _best_chunk_metrics(active_query, chunks)

    # Heuristic override: if metrics are strong, trust them over LLM grader
    OVERRIDE_VECTOR_THRESHOLD = 0.55
    OVERRIDE_LEXICAL_THRESHOLD = 0.40
    if grade == "no" and (
        best_score >= OVERRIDE_VECTOR_THRESHOLD
        or lexical_score >= OVERRIDE_LEXICAL_THRESHOLD
    ):
        old_grade = grade
        grade = "yes"
        grader_reason = (
            f"Heuristic override: LLM said '{old_grade}' but "
            f"vector={best_score:.2f}>={OVERRIDE_VECTOR_THRESHOLD} or "
            f"lexical={lexical_score:.2f}>={OVERRIDE_LEXICAL_THRESHOLD}"
        )
        logger.info("TRACE|node=grade_documents|phase=heuristic_override|reason=%s", grader_reason)
    elif grade not in {"yes", "no"}:
        grade = "yes" if (best_score >= config.GRADE_MIN_RELEVANCE or lexical_score >= config.GRADE_MIN_LEXICAL_OVERLAP) else "no"
        grader_reason = (
            f"Heuristic fallback used (vector={best_score:.2f}, lexical={lexical_score:.2f})"
        )

    emit_trace(
        logger,
        "grade_documents",
        "end",
        state,
        decision=grade,
        chunks=len(chunks),
        best_score=round(best_score, 4),
        lexical_score=round(lexical_score, 4),
        grader_reason=(grader_reason or "")[:120],
    )

    return {
        "retrieved_chunks": chunks,
        "is_relevant": grade,
        "grading_reason": grader_reason,
        "last_tool_name": getattr(tool_message, "name", None),
        "retrieval_debug": _append_retrieval_debug(
            state,
            {
                "query": active_query,
                "decision": grade,
                "reason": grader_reason,
                "chunks": len(chunks),
                "best_score": round(best_score, 4),
                "best_lexical_score": round(lexical_score, 4),
            },
        ),
    }


def rewrite_query(state: WorkflowState) -> dict:
    """Rewrite the current query when retrieved documents are not relevant enough."""

    current_query = state.get("active_query") or state.get("merged_query") or state.get("query_text", "")
    rewrite_attempts = state.get("rewrite_attempts", 0) + 1
    emit_trace(logger, "rewrite_query", "start", state, current_query=current_query[:120], next_attempt=rewrite_attempts)
    grading_reason = state.get("grading_reason") or "Tai lieu tim duoc chua dung y cau hoi."
    ocr_text = state.get("ocr_text") or "Khong co"

    prompt = (
        f"Cau hoi goc cua user: {state.get('query_text', '')}\n"
        f"Truy van hien tai: {current_query}\n"
        f"Noi dung OCR: {ocr_text}\n"
        f"Ly do bi reject: {grading_reason}\n\n"
        "Hay viet lai MOT truy van tim kiem moi bang tieng Viet, uu tien tu dong nghia chuyen mon y te va tu khoa cu the hon. "
        "Chi tra ve truy van moi, khong giai thich."
    )

    rewriter = OllamaChat(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_REWRITE_MODEL,
        timeout=60,
    )

    rewritten_query = current_query
    try:
        rewritten_query = rewriter.generate(
            prompt=prompt,
            system_prompt="Ban la bo may toi uu truy van retrieval. Tra ve mot dong duy nhat.",
        ).strip().splitlines()[0].strip('"\' ')
    except Exception as exc:
        logger.warning("rewrite_query failed, keeping current query: %s", exc)

    if not rewritten_query:
        rewritten_query = current_query

    emit_trace(logger, "rewrite_query", "end", state, rewritten_query=rewritten_query[:120], next_attempt=rewrite_attempts)

    retry_message = HumanMessage(
        content=(
            f"Thu lai retrieval voi truy van da viet lai: {rewritten_query}. "
            f"Ly do lan truoc that bai: {grading_reason}"
        )
    )

    return {
        "rewritten_query": rewritten_query,
        "active_query": rewritten_query,
        "rewrite_attempts": rewrite_attempts,
        "messages": [retry_message],
    }


def generate_final_answer(state: WorkflowState) -> dict:
    """Generate the final answer once retrieval has been accepted or exhausted."""

    emit_trace(logger, "generate_final_answer", "start", state, relevant=state.get("is_relevant", "no"))

    if not state.get("retrieved_chunks") or state.get("is_relevant") != "yes":
        agent_text = _latest_ai_text(state)
        if agent_text and not getattr(list(state.get("messages", []))[-1], "tool_calls", None):
            emit_trace(logger, "generate_final_answer", "end", state, outcome="agent_direct", sources=0)
            return {
                "final_answer": agent_text,
                "sources": [],
                "image_urls": [],
            }

        error_reason = state.get("grading_reason") or state.get("error") or "Khong tim thay tai lieu du phu hop"
        logger.warning("Final answer falling back: %s", error_reason)
        emit_trace(logger, "generate_final_answer", "end", state, outcome="fallback", error=error_reason, sources=0)
        return {
            "final_answer": (
                "Xin lỗi, em chưa tìm thấy thông tin đủ sát để trả lời chắc chắn. "
                "Bạn vui lòng mô tả rõ hơn hoặc liên hệ đội hỗ trợ kỹ thuật ạ."
            ),
            "sources": [],
            "image_urls": [],
            "error": error_reason,
        }

    # Rerank: combine vector score + lexical overlap so the most specific chunk wins
    active_query = state.get("active_query") or state.get("query_text", "")
    ranked_chunks = _rerank_chunks(active_query, state["retrieved_chunks"])
    best_chunk = ranked_chunks[0]

    answer_text = best_chunk.get("content_brief", "")
    issue_id = best_chunk.get("issue_id", "unknown")
    attachment_urls = best_chunk.get("attachment_urls", [])
    source_url = best_chunk.get("source_url", "")
    score = best_chunk.get("score", 0)
    confidence = _confidence_label(float(score))
    logger.info(
        "TRACE|node=generate_final_answer|phase=best_chunk|issue_id=%s|vec_score=%.4f|confidence=%s",
        issue_id, float(score), confidence,
    )

    top_chunks = ranked_chunks[:2]
    context_lines = []
    for idx, chunk in enumerate(top_chunks, 1):
        chunk_issue_id = chunk.get("issue_id", "unknown")
        chunk_score = chunk.get("score", 0)
        chunk_brief = chunk.get("content_brief", "").strip()
        context_lines.append(
            f"{idx}. issue_id={chunk_issue_id}, score={chunk_score:.2f}, content_brief={chunk_brief}"
        )

    context_text = "\n".join(context_lines)
    user_query = state.get("active_query") or state.get("query_text", "")
    ocr_text = state.get("ocr_text")
    grading_reason = state.get("grading_reason") or ""

    prompt = (
        "Cau hoi nguoi dung:\n"
        f"{user_query}\n\n"
        "Van ban OCR (neu co):\n"
        f"{ocr_text or 'Khong co'}\n\n"
        "Ngu canh tu cac chunk truy xuat:\n"
        f"{context_text}\n\n"
        "Danh gia retrieval:\n"
        f"{grading_reason or 'Khong co'}\n\n"
        "Yeu cau:\n"
        "- Tra loi bang tieng Viet, ro rang, de hieu.\n"
        "- Chi dua tren ngu canh da cung cap.\n"
        "- Neu cau hoi la thao tac/quy trinh, phai liet ke DAY DU cac buoc theo dung thu tu, khong luoc bo buoc.\n"
        "- Neu ngu canh co danh sach buoc danh so, uu tien giu nguyen thu tu va noi dung cac buoc do.\n"
        "- Neu thong tin chua duac hoac khong chac chan, noi ro muc do khong chac chan.\n"
        "- Khong chen thong tin ben ngoai ngu canh."
    )

    final_answer = ""
    try:
        chat_client = OllamaChat(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_LLM_MODEL,
            timeout=120
        )
        llm_answer = chat_client.generate(
            prompt=prompt,
            system_prompt="Ban la tro ly helpdesk noi bo, uu tien tinh chinh xac va ngan gon."
        )
        final_answer = f"{llm_answer}\n\n_(Tham khảo ticket #{issue_id} · Độ phù hợp: {confidence})_"
        logger.info(f"Synthesized answer with LLM from chunk #{issue_id}")
        emit_trace(logger, "generate_final_answer", "end", state, outcome="llm", issue_id=issue_id, sources=1)
    except Exception as e:
        logger.warning(f"LLM synthesis failed, fallback to deterministic answer: {e}")
        final_answer = f"{answer_text}\n\n_(Tham khảo ticket #{issue_id} · Độ phù hợp: {confidence})_"
        emit_trace(logger, "generate_final_answer", "end", state, outcome="deterministic", issue_id=issue_id, sources=1, error=str(e))
    
    sources = [
        {
            "issue_id": issue_id,
            "snippet": answer_text[:150],
            "url": source_url,
            "score": score
        }
    ]
    
    return {
        "final_answer": final_answer,
        "sources": sources,
        "image_urls": attachment_urls
    }
