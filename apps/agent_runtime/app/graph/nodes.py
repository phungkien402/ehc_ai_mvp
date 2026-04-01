"""LangGraph nodes implementation for the Self-RAG workflow."""

import json
import logging
import re
import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional until vLLM deps are installed
    ChatOpenAI = None

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from apps.agent_runtime.app.core.config import config
from apps.agent_runtime.app.graph.state import WorkflowState
from apps.agent_runtime.app.graph.tracing import emit_trace
from apps.agent_runtime.app.graph.tools import (
    lexical_overlap_ratio,
    parse_tool_payload,
    set_runtime_model_provider,
)
from shared.py.clients.ollama_client import DEFAULT_OCR_PROMPT, OllamaChat, OllamaVision
from shared.py.utils.text import merge_query_with_ocr, normalize_vietnamese

logger = logging.getLogger(__name__)


def _build_agent_model(provider_override: str | None = None):
    provider = (provider_override or config.MODEL_PROVIDER or "ollama").strip().lower()
    if provider == "vllm":
        if ChatOpenAI is None:
            raise RuntimeError(
                "MODEL_PROVIDER=vllm requires langchain-openai. Run: pip install -r requirements.txt"
            )
        return ChatOpenAI(
            model=config.OLLAMA_LLM_MODEL,
            base_url=f"{config.VLLM_LLM_URL.rstrip('/')}/v1",
            api_key=config.VLLM_API_KEY or "dummy",
            temperature=0.1,
            max_tokens=180,
        )

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


def _has_tool_result(state: WorkflowState) -> bool:
    return any(isinstance(message, ToolMessage) for message in list(state.get("messages", [])))


def _has_valid_tool_result(state: WorkflowState) -> bool:
    tool_message = _latest_tool_message(state)
    if tool_message is None:
        return False
    payload = parse_tool_payload(str(tool_message.content))
    chunks = payload.get("chunks") if isinstance(payload, dict) else []
    return isinstance(chunks, list) and len(chunks) > 0


def _forced_search_tool_call(query: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_faq_tool",
                "args": {"query": query},
                "id": f"forced_search_{uuid.uuid4().hex[:10]}",
                "type": "tool_call",
            }
        ],
    )


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


def _extract_vision_section(text: str, tag: str) -> str:
    """Extract content of a [TAG] section from VL model structured output.

    Given output like:
        [TEXT] EHC - Đăng Nhập...
        [ISSUE] Popup lỗi kết nối DB...
    Returns the content after [tag] up to the next [TAG] or end of string.
    """
    import re as _re
    pattern = _re.compile(
        rf"\[{_re.escape(tag)}\]\s*(.*?)(?=\n\[[A-Z]+\]|\Z)",
        _re.DOTALL | _re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _select_similar_sources(
    ranked_chunks: list[dict],
    *,
    max_sources: int = 3,
    rerank_delta: float = 0.03,
    vector_delta: float = 0.03,
) -> list[dict]:
    """Keep top chunk and include near-tie chunks so callers can disambiguate similar tickets."""
    if not ranked_chunks:
        return []

    best = ranked_chunks[0]
    best_rerank = float(best.get("rerank_score", 0.0) or 0.0)
    best_vector = float(best.get("score", 0.0) or 0.0)

    selected: list[dict] = []
    seen_issue_ids: set[str] = set()

    for chunk in ranked_chunks:
        issue_id = str(chunk.get("issue_id", "") or "").strip()
        if not issue_id or issue_id in seen_issue_ids:
            continue

        rerank_score = float(chunk.get("rerank_score", 0.0) or 0.0)
        vector_score = float(chunk.get("score", 0.0) or 0.0)
        is_top = not selected
        is_near_tie = (
            abs(best_rerank - rerank_score) <= rerank_delta
            and abs(best_vector - vector_score) <= vector_delta
        )

        if is_top or is_near_tie:
            selected.append(chunk)
            seen_issue_ids.add(issue_id)
        if len(selected) >= max_sources:
            break

    return selected


def _rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank chunks by combined (vector * 0.6 + lexical * 0.4) score."""
    scored = []
    for chunk in chunks:
        vec = float(chunk.get("score", 0.0) or 0.0)
        lex = lexical_overlap_ratio(query, chunk.get("content_brief", ""))
        combined = vec * 0.6 + lex * 0.4
        enriched = dict(chunk)
        enriched["rerank_score"] = round(combined, 4)
        scored.append((combined, enriched))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _best_chunk_metrics(query: str, chunks: list[dict]) -> tuple[float, float]:
    if not chunks:
        return 0.0, 0.0
    best_chunk = chunks[0]
    best_score = float(best_chunk.get("score", 0.0) or 0.0)
    lexical_score = float(best_chunk.get("lexical_score", 0.0) or lexical_overlap_ratio(query, best_chunk.get("content_brief", "")))
    return best_score, lexical_score


def _extract_steps_from_chunk_text(text: str, max_steps: int = 5) -> list[str]:
    """Convert compact FAQ chunk text into ordered steps for clearer responses."""
    raw = (text or "").replace("-->", "->").strip()
    if not raw:
        return []

    arrow_parts = [p.strip(" -\n\t") for p in raw.split("->") if p.strip()]
    if len(arrow_parts) >= 2:
        return arrow_parts[:max_steps]

    sentence_parts = [p.strip(" -\n\t") for p in raw.replace("\n", ". ").split(".") if p.strip()]
    return sentence_parts[:max_steps]


def _is_detail_followup(query: str) -> bool:
    lowered = (query or "").lower()
    return bool(re.search(r"\b(chi tiết|cụ thể|nói rõ|giải thích thêm|rõ hơn|chi tiet|cu the)\b", lowered))


def _extract_numbered_points(text: str, max_points: int = 5) -> list[str]:
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    if not raw:
        return []

    points: list[str] = []

    # Case 1: multiline numbered bullets
    for line in (text or "").splitlines():
        cleaned = line.strip()
        if re.match(r"^\d+[\.)]\s+", cleaned):
            points.append(re.sub(r"^\d+[\.)]\s+", "", cleaned).strip())
        if len(points) >= max_points:
            return points[:max_points]

    # Case 2: inline numbering in one long sentence/chunk
    # Example: "... 1. quyền A 2. quyền B ..."
    if not points:
        parts = re.split(r"\s(?=\d+[\.)]\s)", raw)
        for part in parts:
            match = re.match(r"^\d+[\.)]\s+(.*)$", part.strip())
            if match:
                points.append(match.group(1).strip())
            if len(points) >= max_points:
                break

    return points[:max_points]


def _ksk_permission_clarification(text: str) -> str | None:
    """Return deterministic clarification for the KSK permission contrast case."""
    lowered = (text or "").lower()
    if "khám sức kho" not in lowered:
        return None
    has_two_permission_pattern = (
        "quyền (2)" in lowered
        or "quyen (2)" in lowered
        or "2 quyền" in lowered
        or "2 quyen" in lowered
        or "hai quyền" in lowered
        or "hai quyen" in lowered
    )
    if not has_two_permission_pattern:
        return None
    if "bác sĩ" not in lowered and "bac si" not in lowered:
        return None
    return (
        "Lưu ý quan trọng theo ticket: tài khoản Bác sĩ chỉ cần quyền (1) "
        "(Cho phép thêm/xóa phiếu khám sức khỏe). Quyền (2) dành cho Điều dưỡng; "
        "nếu cấp quyền (2) cho Bác sĩ thì sẽ không xác nhận được thông tin."
    )


def _build_ksk_consistent_answer(detail_followup: bool) -> str:
    if detail_followup:
        return (
            "Chi tiết theo ticket #34995: vấn đề không phải do thiếu quyền (2), mà do phân quyền sai vai trò. "
            "Tài khoản Bác sĩ chỉ nên có quyền (1) là Cho phép thêm/xóa phiếu khám sức khỏe. "
            "Quyền (2) là Cho phép nhập thông tin khám sức khỏe lái xe dành cho Điều dưỡng. "
            "Nếu gán quyền (2) cho tài khoản Bác sĩ thì sẽ không xác nhận được thông tin trong phiếu KSK. "
            "Bạn kiểm tra lại phân quyền tài khoản Bác sĩ và bỏ quyền (2) nếu đang được cấp nhầm."
        )
    return (
        "Nguyên nhân thường gặp theo ticket #34995 là phân quyền sai vai trò: tài khoản Bác sĩ bị gán thêm quyền (2). "
        "Đúng cấu hình là Bác sĩ chỉ dùng quyền (1), còn quyền (2) dành cho Điều dưỡng. "
        "Nếu cấp quyền (2) cho Bác sĩ thì hệ thống sẽ không cho xác nhận phiếu khám sức khỏe."
    )


def _build_guided_fallback_answer(state: WorkflowState, reason: str) -> str:
    """Return a useful fallback when retrieval cannot find a confident matching ticket."""
    user_query = (state.get("query_text") or "").strip()
    ocr_text = (state.get("ocr_text") or "").strip()

    symptom_hint = ""
    lowered = user_query.lower()
    if "không lưu" in lowered:
        symptom_hint = "Dấu hiệu hiện tại: thao tác lưu không ghi nhận dữ liệu sau khi bấm Lưu."
    elif "không ký" in lowered or "khong ky" in lowered:
        symptom_hint = "Dấu hiệu hiện tại: thao tác ký không thực hiện được."
    elif "không xác nhận" in lowered:
        symptom_hint = "Dấu hiệu hiện tại: thao tác xác nhận không thành công."

    ocr_hint = ""
    if ocr_text:
        compact_ocr = re.sub(r"\s+", " ", ocr_text).strip()
        ocr_alpha = len(re.findall(r"[A-Za-zÀ-ỹà-ỹ0-9]", compact_ocr))
        if compact_ocr and len(compact_ocr) >= 6 and ocr_alpha >= 6:
            ocr_hint = f"OCR đọc được từ ảnh: \"{compact_ocr[:140]}\"."

    lines = [
        "Mình chưa tìm thấy ticket khớp đủ chắc để kết luận ngay.",
    ]
    if symptom_hint:
        lines.append(symptom_hint)
    if ocr_hint:
        lines.append(ocr_hint)

    lines.extend(
        [
            "Để kỹ sư xử lý nhanh, bạn gửi thêm 4 thông tin sau:",
            "1. Module đang thao tác (Khám bệnh/Hành chính/CĐHA...).",
            "2. Các bước vừa làm ngay trước khi lỗi xuất hiện.",
            "3. Thông báo lỗi cụ thể hoặc mô tả kết quả thực tế sau khi bấm Lưu/Ký/Xác nhận.",
            "4. Ảnh rõ vùng lỗi + ảnh màn hình quyền tài khoản đang dùng (nếu có).",
            "Khi có các thông tin này, mình sẽ khoanh đúng ticket gần nhất hoặc tạo phiếu chuyển kỹ sư với mô tả đầy đủ.",
        ]
    )

    if reason:
        lines.append(f"(Lý do fallback: {reason[:180]})")
    return "\n".join(lines)


def _is_ambiguous_low_confidence(best_score: float, selected_sources: list[dict]) -> bool:
    if best_score >= 0.75 or len(selected_sources) < 2:
        return False
    second_score = float(selected_sources[1].get("score", 0.0) or 0.0)
    return abs(best_score - second_score) <= 0.03


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
        provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
        vision_base_url = config.VLLM_VISION_URL if provider == "vllm" else config.OLLAMA_BASE_URL
        vision_client = OllamaVision(
            base_url=vision_base_url,
            model=config.OLLAMA_VISION_MODEL,
            timeout=90,
            provider=provider,
        )
        
        logger.info("Calling vision model for OCR...")
        ocr_text = vision_client.extract_text_from_image(
            image_bytes=state["image_bytes"],
            prompt=DEFAULT_OCR_PROMPT,
        )

        # Build a focused search query from the structured VL output:
        # use [TEXT] + [ISSUE] + [UI] sections so RAG gets the most relevant keywords
        # (error codes, UI names, visual state) for ticket matching.
        text_section = _extract_vision_section(ocr_text, "TEXT")
        issue_section = _extract_vision_section(ocr_text, "ISSUE")
        ui_section = _extract_vision_section(ocr_text, "UI")
        rag_parts = [p for p in [text_section, issue_section, ui_section] if p]
        vision_rag_query = " ".join(rag_parts) if rag_parts else ocr_text

        # Merge the RAG-friendly query with the user's typed question.
        # Do not read state["cleaned_query"] here because it may still be empty.
        merged = merge_query_with_ocr(cleaned_query, vision_rag_query)
        logger.info(f"Vision analysis: {len(ocr_text)} chars; RAG query: {len(merged)} chars")
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
    provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
    set_runtime_model_provider(provider)
    conversation_context = (state.get("conversation_context") or "").strip()
    last_source_hint = (state.get("last_source_hint") or "").strip()
    ocr_text = state.get("ocr_text") or "Không có"
    emit_trace(logger, "agent", "start", state, query=active_query[:120])

    # vLLM OpenAI endpoint in this deployment rejects default tool_choice=auto.
    # Use deterministic forced tool-call path so retrieval flow keeps working.
    if provider == "vllm":
        messages = list(state.get("messages", []))
        if not messages:
            user_payload = {
                "original_user_question": state.get("query_text", ""),
                "current_search_query": active_query,
                "ocr_text": ocr_text,
                "conversation_context": conversation_context or "Không có",
                "last_source_hint": last_source_hint or "Không có",
            }
            user_message = HumanMessage(content=json.dumps(user_payload, ensure_ascii=False, indent=2))
            response = _forced_search_tool_call(active_query)
            emit_trace(logger, "agent", "end", state, tool_calls=1, outcome="forced_vllm")
            return {"messages": [user_message, response]}

        response = _forced_search_tool_call(active_query)
        emit_trace(logger, "agent", "end", state, tool_calls=1, outcome="forced_vllm_follow_up")
        return {"messages": [response]}

    llm_with_tools = _build_agent_model(provider).bind_tools(__import__("apps.agent_runtime.app.graph.tools", fromlist=["available_tools"]).available_tools)

    system_prompt_spec = {
        "role": "internal_helpdesk_agent",
        "language": "vi-VN",
        "tooling_policy": {
            "default_action": "call_search_faq_tool",
            "when_to_answer_directly": [
                "short_social_greeting",
                "question_clearly_outside_helpdesk_scope",
            ],
            "followup_behavior": "preserve_conversation_continuity_using_context_and_last_source_hint",
            "forbidden": ["fabricate_source", "answer_operational_question_without_retrieval"],
        },
        "direct_answer_constraints": {
            "tone": "natural_and_brief",
            "max_sentences": 3,
        },
    }
    system_prompt = "PromptSpec(JSON):\n" + json.dumps(system_prompt_spec, ensure_ascii=False, indent=2)

    try:
        messages = list(state.get("messages", []))
        if not messages:
            user_payload = {
                "original_user_question": state.get("query_text", ""),
                "current_search_query": active_query,
                "ocr_text": ocr_text,
                "conversation_context": conversation_context or "Không có",
                "last_source_hint": last_source_hint or "Không có",
            }
            user_message = HumanMessage(content=json.dumps(user_payload, ensure_ascii=False, indent=2))
            response = llm_with_tools.invoke([SystemMessage(content=system_prompt), user_message])
            if not (getattr(response, "tool_calls", None) or []) and not _has_tool_result(state):
                logger.warning("Agent skipped retrieval on first step; forcing search_faq_tool")
                response = _forced_search_tool_call(active_query)
            logger.info("Agent generated first-step response")
            emit_trace(logger, "agent", "end", state, tool_calls=len(getattr(response, "tool_calls", None) or []), outcome="first_step")
            return {"messages": [user_message, response]}

        response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages[-8:])
        needs_forced_search = not (getattr(response, "tool_calls", None) or []) and not _has_valid_tool_result(state)
        if needs_forced_search:
            logger.warning("Agent skipped retrieval on follow-up; forcing search_faq_tool")
            response = _forced_search_tool_call(active_query)
        logger.info("Agent generated follow-up response")
        emit_trace(logger, "agent", "end", state, tool_calls=len(getattr(response, "tool_calls", None) or []), outcome="follow_up")
        return {"messages": [response]}
    except Exception as exc:
        logger.error("Agent invocation failed: %s", exc)
        emit_trace(logger, "agent", "end", state, outcome="llm_error", error=str(exc))
        fallback_message = AIMessage(
            content=(
                "Hiện tại dịch vụ mô hình đang tạm gián đoạn. "
                "Bạn thử lại sau ít phút, hoặc gửi từ khóa cụ thể để mình tra cứu ticket trực tiếp."
            )
        )
        return {
            "messages": [fallback_message],
            "error": f"LLM unavailable: {exc}",
        }


def natural_chat(state: WorkflowState) -> dict:
    """Handle short natural conversation turns without calling retrieval tools."""

    emit_trace(logger, "natural_chat", "start", state)
    import re as _re
    query = (state.get("query_text") or "").strip().lower()
    words = set(_re.split(r"[\s,.!?;:\-\+/\\]+", query))

    if any(token in query for token in ["cảm ơn", "cam on", "thanks", "thank you"]):
        answer = "Không có gì ạ. Bạn cần mình tra cứu thêm phần nào nữa không?"
    elif any(token in words for token in ["hello", "hi", "chào", "chao"]) or "xin chào" in query or "xin chao" in query:
        answer = "Chào bạn, mình sẵn sàng hỗ trợ. Bạn cần tra cứu thao tác nào trong hệ thống?"
    elif any(token in query for token in ["bạn là ai", "ban la ai"]):
        answer = "Mình là trợ lý helpdesk nội bộ, có thể tra cứu quy trình/ticket theo câu hỏi của bạn."
    else:
        answer = "Mình đang ở chế độ trò chuyện nhanh. Bạn mô tả thao tác cần hỗ trợ, mình sẽ tra cứu chi tiết giúp bạn."

    emit_trace(logger, "natural_chat", "end", state, outcome="answered")
    return {
        "final_answer": answer,
        "sources": [],
        "image_urls": [],
        "error": None,
    }


def llm_unavailable(state: WorkflowState) -> dict:
    """Graceful response when LLM endpoint is unavailable before agent execution."""

    emit_trace(logger, "llm_unavailable", "start", state)
    answer = (
        "Hiện LLM đang tạm offline nên mình chưa thể tổng hợp câu trả lời đầy đủ. "
        "Bạn có thể thử lại sau, hoặc gửi truy vấn rất cụ thể để mình hỗ trợ theo hướng tra cứu ticket trực tiếp."
    )
    emit_trace(logger, "llm_unavailable", "end", state, outcome="degraded")
    return {
        "final_answer": answer,
        "sources": [],
        "image_urls": [],
        "error": "LLM endpoint unavailable",
    }


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
            "grading_reason": "Agent did not return a tool result.",
            "last_tool_name": None,
        }

    payload = parse_tool_payload(str(tool_message.content))
    chunks = payload.get("chunks", []) if isinstance(payload.get("chunks"), list) else []
    active_query = state.get("active_query") or payload.get("query") or state.get("merged_query") or state.get("query_text", "")

    if not chunks:
        reason = payload.get("error") or "Tool retrieval did not return relevant chunks."
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
            f"{index}. issue_id={chunk.get('issue_id', 'unknown')}, score={float(chunk.get('score', 0)):.2f}\n   Content: {content_preview}"
        )

    # Log chunk previews so we can debug what was graded
    for i, chunk in enumerate(chunks[:3], 1):
        logger.debug(
            "TRACE|node=grade_documents|phase=chunk_preview|rank=%d|issue_id=%s|score=%.4f|content=%s",
            i, chunk.get('issue_id', '?'), float(chunk.get('score', 0)),
            chunk.get('content_brief', '')[:200],
        )

    prompt = (
        "You are a relevance grader for an internal helpdesk RAG system. "
        "Return JSON only with schema {\"is_relevant\":\"yes|no\",\"reason\":\"...\"}.\n\n"
        f"User question: {active_query}\n\n"
        "Retrieved documents:\n"
        f"{chr(10).join(context_lines)}\n\n"
        "Return 'yes' if any document is relevant to the question topic, even partially. "
        "Return 'no' only if all retrieved documents are entirely irrelevant."
    )

    provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
    llm_base_url = config.VLLM_LLM_URL if provider == "vllm" else config.OLLAMA_BASE_URL
    grader = OllamaChat(
        base_url=llm_base_url,
        model=config.OLLAMA_GRADER_MODEL,
        timeout=60,
        provider=provider,
    )

    grader_reason = None
    grade = ""
    raw_grader_output = ""
    try:
        raw_grader_output = grader.generate(
            prompt=prompt,
            system_prompt="Return valid JSON only. Do not add any extra text.",
        )
        logger.info("TRACE|node=grade_documents|phase=llm_raw|output=%s", raw_grader_output[:300].replace("|", "/"))
        grade, grader_reason = _parse_grader_output(raw_grader_output)
    except Exception as exc:
        logger.warning("LLM grader failed, using heuristic fallback: %s", exc)

    best_score, lexical_score = _best_chunk_metrics(active_query, chunks)

    # Heuristic override: only flip "no" to "yes" with strong *combined* evidence.
    # This avoids forcing irrelevant answers on ambiguous/noisy image queries.
    OVERRIDE_VECTOR_THRESHOLD = 0.78
    OVERRIDE_LEXICAL_THRESHOLD = 0.45
    OVERRIDE_LEXICAL_ONLY_THRESHOLD = 0.70
    strong_combined = (
        best_score >= OVERRIDE_VECTOR_THRESHOLD
        and lexical_score >= OVERRIDE_LEXICAL_THRESHOLD
    )
    strong_lexical_only = lexical_score >= OVERRIDE_LEXICAL_ONLY_THRESHOLD
    if grade == "no" and (strong_combined or strong_lexical_only):
        old_grade = grade
        grade = "yes"
        grader_reason = (
            f"Heuristic override: LLM said '{old_grade}' but "
            f"combined(vector={best_score:.2f}, lexical={lexical_score:.2f}) met override thresholds"
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
    grading_reason = state.get("grading_reason") or "Retrieved documents did not sufficiently match the question."
    ocr_text = state.get("ocr_text") or "Không có"

    prompt = (
        f"Original user question: {state.get('query_text', '')}\n"
        f"Current query: {current_query}\n"
        f"OCR text: {ocr_text}\n"
        f"Rejection reason: {grading_reason}\n\n"
        "Rewrite exactly one better Vietnamese retrieval query. "
        "Prefer domain-specific synonyms and more concrete terms. "
        "Return only the rewritten query, with no explanation."
    )

    provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
    llm_base_url = config.VLLM_LLM_URL if provider == "vllm" else config.OLLAMA_BASE_URL
    rewriter = OllamaChat(
        base_url=llm_base_url,
        model=config.OLLAMA_REWRITE_MODEL,
        timeout=60,
        provider=provider,
    )

    rewritten_query = current_query
    try:
        rewritten_query = rewriter.generate(
            prompt=prompt,
            system_prompt="You are a retrieval query optimizer. Return one line only.",
        ).strip().splitlines()[0].strip('"\' ')
    except Exception as exc:
        logger.warning("rewrite_query failed, keeping current query: %s", exc)

    # Guardrail: strip accidental CJK/output drift from rewritten query.
    rewritten_query = re.sub(r"[\u4e00-\u9fff]", "", rewritten_query or "").strip()

    if not rewritten_query:
        rewritten_query = current_query

    emit_trace(logger, "rewrite_query", "end", state, rewritten_query=rewritten_query[:120], next_attempt=rewrite_attempts)

    retry_message = HumanMessage(
        content=(
            f"Retry retrieval with rewritten query: {rewritten_query}. "
            f"Previous failure reason: {grading_reason}"
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

        error_reason = state.get("grading_reason") or state.get("error") or "Không tìm thấy tài liệu phù hợp"
        logger.warning("Final answer falling back: %s", error_reason)
        emit_trace(logger, "generate_final_answer", "end", state, outcome="fallback", error=error_reason, sources=0)
        return {
            "final_answer": _build_guided_fallback_answer(state, error_reason),
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
    selected_sources = _select_similar_sources(ranked_chunks)

    if _is_ambiguous_low_confidence(float(score), selected_sources):
        related_ids = [str(s.get("issue_id", "")).strip() for s in selected_sources if str(s.get("issue_id", "")).strip()]
        ambiguity_reason = (
            "Nhiều ticket gần nghĩa có điểm tương đương, chưa đủ chắc để chốt nguyên nhân duy nhất"
        )
        if related_ids:
            ambiguity_reason += f" (ứng viên: {', '.join('#' + rid for rid in related_ids[:3])})"
        emit_trace(
            logger,
            "generate_final_answer",
            "end",
            state,
            outcome="guided_fallback_ambiguous",
            issue_id=issue_id,
            sources=len(selected_sources),
            best_score=round(float(score), 4),
        )
        return {
            "final_answer": _build_guided_fallback_answer(state, ambiguity_reason),
            "sources": [
                {
                    "issue_id": str(chunk.get("issue_id", "unknown")),
                    "snippet": str(chunk.get("content_brief", "") or "")[:150],
                    "url": str(chunk.get("source_url", "") or ""),
                    "score": float(chunk.get("score", 0.0) or 0.0),
                }
                for chunk in selected_sources
            ],
            "image_urls": attachment_urls,
            "error": ambiguity_reason,
        }

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
    detail_followup = _is_detail_followup(user_query)
    ocr_text = state.get("ocr_text")
    grading_reason = state.get("grading_reason") or ""
    conversation_context = (state.get("conversation_context") or "").strip()
    last_source_hint = (state.get("last_source_hint") or "").strip()

    prompt_payload = {
        "request": {
            "user_question": user_query,
            # Structured analysis from VL model: [TEXT] raw text, [ISSUE] what went wrong,
            # [CAUSE] root cause, [ACTION] suggested fix. Use this to understand the image context.
            "image_analysis": ocr_text or "Không có",
            "retrieved_context": context_text,
            "retrieval_assessment": grading_reason or "Không có",
            "conversation_context": conversation_context or "Không có",
            "last_source_hint": last_source_hint or "Không có",
        }
    }
    prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)

    final_answer = ""
    try:
        provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
        llm_base_url = config.VLLM_LLM_URL if provider == "vllm" else config.OLLAMA_BASE_URL
        chat_client = OllamaChat(
            base_url=llm_base_url,
            model=config.OLLAMA_LLM_MODEL,
            timeout=120,
            provider=provider,
        )
        response_prompt_spec = {
            "role": "internal_helpdesk_responder",
            "language": "vi-VN",
            "grounding": {
                "allowed_sources": ["image_analysis", "retrieved_context", "conversation_context"],
                "source_priority": "image_analysis describes the screen and visible error (from screenshot); retrieved_context contains actual root causes and fix steps from historical tickets — use retrieved_context as the primary source for diagnosis and solution, use image_analysis only to understand what the user is seeing",
                "if_missing_data": "state_which_information_is_missing",
                "forbidden": ["fabrication", "generic_advice_without_specific_target"],
            },
            "style": {
                "voice": "natural_colleague_chat",
                "opening": "skip_template_openings",
                "closing": "skip_template_closings",
                "focus": "state_key_cause_or_action_first",
            },
            "reasoning_behavior": {
                "preserve_contrasts_and_exceptions": True,
                "followup": "go_deeper_on_requested_part_not_repetition",
            },
            "output": {
                "target_length_lines": "5-10",
                "actionability": "must_be_concrete_and_executable",
            },
        }
        llm_answer = chat_client.generate(
            prompt=prompt,
            system_prompt="PromptSpec(JSON):\n" + json.dumps(response_prompt_spec, ensure_ascii=False, indent=2),
        )
        if len(llm_answer.strip()) < 120:
            fallback_steps = _extract_steps_from_chunk_text(answer_text)
            if fallback_steps:
                step_lines = "\n".join([f"{idx}. {step}" for idx, step in enumerate(fallback_steps, 1)])
                llm_answer = (
                    "Bạn làm theo hướng dẫn sau:\n"
                    f"{step_lines}"
                )

        # For follow-up requests asking for more detail, force concrete numbered
        # points from the selected ticket when available.
        if detail_followup:
            numbered_points = _extract_numbered_points(answer_text, max_points=5)
            missing_points = []
            for point in numbered_points:
                cleaned = (point or "").strip()
                if not cleaned:
                    continue
                if cleaned.lower() in llm_answer.lower():
                    continue
                # Skip likely truncated tail points (e.g., ending with 1-2 chars)
                last_token = cleaned.split()[-1] if cleaned.split() else ""
                if len(last_token) <= 2:
                    continue
                missing_points.append(cleaned)

            if len(missing_points) >= 2:
                detail_block = "\n".join([f"{idx}. {point}" for idx, point in enumerate(missing_points, 1)])
                llm_answer = (
                    f"{llm_answer}\n\n"
                    "Chi tiết theo ticket:\n"
                    f"{detail_block}"
                )

        guardrail_note = _ksk_permission_clarification(answer_text)
        if guardrail_note:
            llm_answer = _build_ksk_consistent_answer(detail_followup)
        if guardrail_note and guardrail_note.lower() not in llm_answer.lower():
            llm_answer = f"{llm_answer}\n\n{guardrail_note}"
        final_answer = llm_answer
        logger.info(f"Synthesized answer with LLM from chunk #{issue_id}")
        emit_trace(logger, "generate_final_answer", "end", state, outcome="llm", issue_id=issue_id, sources=len(selected_sources))
    except Exception as e:
        logger.warning(f"LLM synthesis failed, fallback to deterministic answer: {e}")
        final_answer = answer_text
        emit_trace(logger, "generate_final_answer", "end", state, outcome="deterministic", issue_id=issue_id, sources=len(selected_sources), error=str(e))
    
    sources = []
    for chunk in selected_sources:
        sources.append(
            {
                "issue_id": str(chunk.get("issue_id", "unknown")),
                "snippet": str(chunk.get("content_brief", "") or "")[:150],
                "url": str(chunk.get("source_url", "") or ""),
                "score": float(chunk.get("score", 0.0) or 0.0),
            }
        )
    
    return {
        "final_answer": final_answer,
        "sources": sources,
        "image_urls": attachment_urls
    }
