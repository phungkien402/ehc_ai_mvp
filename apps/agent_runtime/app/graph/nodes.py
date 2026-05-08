"""LangGraph nodes implementation for the Self-RAG workflow."""

import base64
import json
import logging
import re
import sys
import uuid
from pathlib import Path

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
    search_faq_chunks,
    set_runtime_model_provider,
)
from shared.py.clients.ollama_client import DEFAULT_OCR_PROMPT, OllamaChat, OllamaVision
from shared.py.utils.text import merge_query_with_ocr, normalize_vietnamese
from pipelines.ingestion.app.utils.image_storage import ImageStorage

_REPO_ROOT = Path(__file__).resolve().parents[4]

logger = logging.getLogger(__name__)


def _build_agent_model(provider_override: str | None = None):
    provider = (provider_override or config.MODEL_PROVIDER or "ollama").strip().lower()
    if provider == "vllm":
        if ChatOpenAI is None:
            raise RuntimeError(
                "MODEL_PROVIDER=vllm requires langchain-openai. Run: pip install -r requirements.txt"
            )
        return ChatOpenAI(
            model=config.VLLM_LLM_MODEL,
            base_url=f"{config.VLLM_LLM_URL.rstrip('/')}/v1",
            api_key=config.VLLM_API_KEY or "EMPTY",
            temperature=0.1,
            max_tokens=512,
        )

    return ChatOllama(
        model=config.OLLAMA_LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=180,
        think=False,
    )


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks, appending captured content to /tmp/ehc_thinking.log."""
    import time as _time

    def _capture(m):
        block = m.group(0)
        try:
            with open("/tmp/ehc_thinking.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{_time.strftime('%H:%M:%S')}] {block}\n")
        except Exception:
            pass
        return ""

    text = re.sub(r"<think>.*?</think>", _capture, text, flags=re.DOTALL)
    # Unclosed tag (truncated by max_tokens) — capture and drop the rest
    text = re.sub(r"<think>.*", _capture, text, flags=re.DOTALL)
    text = re.sub(r"Thinking Process:.*", "", text, flags=re.DOTALL)
    return text.strip()


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
    seen_source_ids: set[str] = set()

    for chunk in ranked_chunks:
        source_id = str(chunk.get("source_id", "") or chunk.get("issue_id", "") or "").strip()
        if not source_id or source_id in seen_source_ids:
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
            seen_source_ids.add(source_id)
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


def _is_doc_flow_query(query: str) -> bool:
    """Detect when user asks for procedural/full-flow guidance from documentation."""
    lowered = (query or "").lower()
    return bool(
        re.search(
            r"\b(hướng dẫn|huong dan|chi tiết|chi tiet|từ đầu|tu dau|toàn bộ|toan bo|quy trình|quy trinh|các bước|cac buoc)\b",
            lowered,
        )
    )


def _collect_module_doc_context_chunks(ranked_chunks: list[dict], max_chunks: int = 8) -> list[dict]:
    """Collect module_doc chunks from the same guide family for richer procedural context."""
    if not ranked_chunks:
        return []

    anchor = ranked_chunks[0]
    anchor_title = str(anchor.get("source_title", "") or "")
    guide_prefix = anchor_title.split("/")[0].strip().lower() if anchor_title else ""

    selected: list[dict] = []
    seen_ids: set[str] = set()
    for chunk in ranked_chunks:
        source_id = str(chunk.get("source_id", chunk.get("issue_id", "")) or "").strip()
        title = str(chunk.get("source_title", "") or "").lower()
        source_type = str(chunk.get("source_type", "faq") or "faq").strip()

        if source_type != "module_doc":
            continue
        if guide_prefix and guide_prefix not in title:
            continue
        if source_id and source_id in seen_ids:
            continue

        selected.append(chunk)
        if source_id:
            seen_ids.add(source_id)
        if len(selected) >= max_chunks:
            break

    if selected:
        return selected[:max_chunks]
    return [c for c in ranked_chunks if str(c.get("source_type", "faq")) == "module_doc"][:max_chunks] or ranked_chunks[:max_chunks]


def _augment_module_doc_flow_chunks(base_chunks: list[dict], query: str, max_chunks: int = 8) -> list[dict]:
    """Augment module_doc context by issuing broader procedural retrieval queries."""
    merged: list[dict] = []
    seen_source_ids: set[str] = set()

    def _add_chunk(chunk: dict) -> None:
        source_id = str(chunk.get("source_id", chunk.get("issue_id", "")) or "").strip()
        source_type = str(chunk.get("source_type", "faq") or "faq").strip()
        if source_type != "module_doc":
            return
        if source_id and source_id in seen_source_ids:
            return
        merged.append(chunk)
        if source_id:
            seen_source_ids.add(source_id)

    for c in base_chunks:
        _add_chunk(c)

    supplement_queries = [
        query,
        f"{query} hướng dẫn chi tiết từng bước",
        f"{query} toàn bộ quy trình",
    ]
    for q in supplement_queries:
        try:
            for c in search_faq_chunks(q):
                _add_chunk(c)
                if len(merged) >= max_chunks:
                    return merged[:max_chunks]
        except Exception as exc:
            logger.warning("Supplement module_doc retrieval failed for '%s': %s", q, exc)

    return merged[:max_chunks]


def _build_stepwise_answer_from_chunks(chunks: list[dict], max_steps: int = 10) -> str:
    """Build deterministic stepwise answer from retrieved doc chunks when LLM output is too short."""
    steps: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        full_text = str(chunk.get("content_full", "") or chunk.get("content_brief", "") or "")
        for raw_line in full_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not (line.startswith("-") or line.startswith("+") or re.match(r"^\d+[\.)]", line)):
                continue

            cleaned = re.sub(r"^[\-\+\d\.)\s]+", "", line).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if len(cleaned) < 8:
                continue

            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            steps.append(cleaned)
            if len(steps) >= max_steps:
                break
        if len(steps) >= max_steps:
            break

    if not steps:
        return ""

    lines = ["Bạn làm từ đầu theo thứ tự sau:"]
    lines.extend([f"{idx}. {step}" for idx, step in enumerate(steps, 1)])
    return "\n".join(lines)


def _build_doc_flow_answer_by_sections(chunks: list[dict], max_sections: int = 5, max_points_per_section: int = 3) -> str:
    """Build deterministic guide answer grouped by document sections."""
    if not chunks:
        return ""

    seen_section: set[str] = set()
    section_blocks: list[tuple[str, list[str]]] = []

    for chunk in chunks:
        section_title = str(chunk.get("section_path", "") or chunk.get("source_title", "") or "").strip()
        if not section_title:
            continue
        key = section_title.lower()
        if key in seen_section:
            continue
        seen_section.add(key)

        full_text = str(chunk.get("content_full", "") or chunk.get("content_brief", "") or "")
        points: list[str] = []
        seen_points: set[str] = set()
        for raw_line in full_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not (line.startswith("-") or line.startswith("+") or re.match(r"^\d+[\.)]", line)):
                continue

            cleaned = re.sub(r"^[\-\+\d\.)\s]+", "", line).strip()
            cleaned = _clean_docx_step_text(cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 8:
                continue

            lower = cleaned.lower()
            if lower in seen_points:
                continue
            seen_points.add(lower)
            points.append(cleaned)
            if len(points) >= max_points_per_section:
                break

        if points:
            section_blocks.append((section_title, points))
        if len(section_blocks) >= max_sections:
            break

    if not section_blocks:
        return ""

    lines = ["Bạn làm theo từng bước như sau:"]
    for idx, (title, points) in enumerate(section_blocks, 1):
        lines.append(f"Bước {idx}: {title}")
        for p in points:
            lines.append(f"- {p}")
    return "\n".join(lines)


def _is_progress_followup_query(query: str) -> bool:
    lowered = (query or "").lower()
    return bool(
        re.search(
            r"\b(mỗi vậy thôi|moi vay thoi|còn nữa không|con nua khong|tiếp theo|tiep theo|rồi sao|roi sao|đã xong|da xong|xong bước|xong buoc)\b",
            lowered,
        )
    )


def _progress_skip_keywords(query: str) -> list[str]:
    lowered = (query or "").lower()
    keywords: list[str] = []
    if "chuẩn bị" in lowered or "chuan bi" in lowered:
        keywords.extend(["chuẩn bị", "máy trạm", "worklist"])
    if "cài đặt thông số" in lowered or "cai dat thong so" in lowered:
        keywords.extend(["thông số", "pacs name", "pacs ae", "pacs port", "worklist port"])
    return keywords


def _build_progress_followup_answer(chunks: list[dict], query: str) -> str:
    """Return next-step guidance for conversational follow-up without repeating completed steps."""
    raw = _build_stepwise_answer_from_chunks(chunks, max_steps=12)
    if not raw:
        return ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    step_lines = [ln for ln in lines if re.match(r"^\d+\.\s+", ln)]
    if not step_lines:
        return ""

    skip_keys = _progress_skip_keywords(query)
    filtered: list[str] = []
    for ln in step_lines:
        lowered = ln.lower()
        if skip_keys and any(k in lowered for k in skip_keys):
            continue
        filtered.append(ln)

    if not filtered:
        filtered = step_lines[1:]
    if not filtered:
        filtered = step_lines

    next_steps = filtered[:3]
    if not next_steps:
        return ""

    out = ["Chưa hết, bạn làm tiếp các bước sau:"]
    out.extend(next_steps)
    out.append("Xong các bước này thì nhắn mình, mình chỉ bước tiếp theo ngay.")
    return "\n".join(out)


def _clean_docx_step_text(step: str) -> str:
    """Remove noisy DOCX-derived prefixes and normalize compact bullet text."""
    cleaned = re.sub(r"\s+", " ", (step or "")).strip(" -\n\t")
    if not cleaned:
        return ""

    # Drop boilerplate prefixes frequently seen in chunk previews.
    cleaned = re.sub(r"^hdsd[_\s-]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^hướng dẫn[_\s-]*", "", cleaned, flags=re.IGNORECASE)

    # If line still starts with title-like phrase, keep the actionable tail after ":" or "-".
    lowered = cleaned.lower()
    if "kết nối minipacs" in lowered and (":" in cleaned or " - " in cleaned):
        if ":" in cleaned:
            cleaned = cleaned.split(":", 1)[1].strip()
        elif " - " in cleaned:
            cleaned = cleaned.split(" - ", 1)[1].strip()

    return cleaned


def _pick_best_docx_source_for_query(selected_sources: list[dict], query: str) -> dict:
    """Pick the source whose section/title best matches explicit query intent."""
    if not selected_sources:
        return {}

    lowered_query = (query or "").lower()
    has_prepare_intent = bool(re.search(r"\b(chuẩn bị|chuan bi|chuẩn bị gì|can chuan bi)\b", lowered_query))
    has_connect_intent = bool(re.search(r"\b(kết nối|ket noi|cấu hình|cau hinh)\b", lowered_query))

    best = selected_sources[0]
    best_score = -1.0
    for src in selected_sources:
        section = str(src.get("section_path", "") or "").lower()
        title = str(src.get("source_title", "") or "").lower()
        score = float(src.get("score", 0.0) or 0.0)

        if has_prepare_intent and ("chuẩn bị" in section or "chuan bi" in section or "chuẩn bị" in title or "chuan bi" in title):
            score += 0.25
        if has_connect_intent and ("kết nối" in section or "ket noi" in section or "kết nối" in title or "ket noi" in title):
            score += 0.08

        if score > best_score:
            best_score = score
            best = src

    return best


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


def _should_force_hybrid_for_ambiguous_query(state: WorkflowState, selected_sources: list[dict]) -> bool:
    intent = str(state.get("retrieval_intent", "") or "").strip().lower()
    if intent != "ambiguous":
        return False

    query = (state.get("active_query") or state.get("query_text") or "").strip()
    if len(query) > 16:
        return False

    has_docx_sources = any(str(s.get("source_type", "faq")) == "module_doc" for s in selected_sources)
    return has_docx_sources and len(selected_sources) >= 1


def _build_docx_hybrid_answer(selected_sources: list[dict], query: str) -> str:
    """Return quick practical guidance + deep-dive options for ambiguous docx queries."""
    lowered_query = (query or "").lower()
    is_prepare_query = bool(re.search(r"\b(chuẩn bị|chuan bi|cần chuẩn bị|can chuan bi)\b", lowered_query))

    stepwise = _build_stepwise_answer_from_chunks(selected_sources, max_steps=10)
    step_lines = [
        re.sub(r"^\d+\.\s+", "", ln).strip()
        for ln in stepwise.splitlines()
        if re.match(r"^\d+\.\s+", ln)
    ]
    step_lines = [_clean_docx_step_text(s) for s in step_lines]
    step_lines = [s for s in step_lines if s]

    if is_prepare_query:
        preferred = [
            s for s in step_lines
            if any(k in s.lower() for k in ["chuẩn bị", "chuan bi", "máy trạm", "worklist", "key", "his"])
        ]
        if preferred:
            step_lines = preferred

    lines = ["Mình tóm tắt nhanh các bước chính như sau:"]
    if step_lines:
        for idx, step in enumerate(step_lines[:3], 1):
            lines.append(f"{idx}. {step}")
    else:
        top = _pick_best_docx_source_for_query(selected_sources, query)
        section_title = str(top.get("section_path", "") or top.get("source_title", "") or "tài liệu liên quan").strip()
        lines.append(f"1. Mở mục: {section_title}.")
        lines.append("2. Thực hiện lần lượt các thao tác trong mục này và lưu cấu hình.")

    lines.append("Nếu cần, mình bung riêng từng mục 1, 2 hoặc 3.")
    return "\n".join(lines)


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
        vision_model = config.VLLM_VISION_MODEL if provider == "vllm" else config.OLLAMA_VISION_MODEL
        vision_client = OllamaVision(
            base_url=vision_base_url,
            model=vision_model,
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

    context_block = f"\n\nLịch sử hội thoại gần nhất:\n{conversation_context}" if conversation_context else ""
    source_block = f"\nTin nhắn trước liên quan đến: {last_source_hint}" if last_source_hint else ""

    system_prompt = (
        "Bạn là trợ lý helpdesk EHC, hỗ trợ nhân viên y tế dùng phần mềm HIS.\n\n"
        "Hãy đọc lịch sử hội thoại bên dưới trước khi trả lời.\n"
        "- Nếu câu hỏi là chào hỏi, cảm ơn, hỏi lại câu trước, hoặc hội thoại thông thường → trả lời trực tiếp, tự nhiên, KHÔNG gọi tool.\n"
        "- Nếu câu hỏi cần tra cứu quy trình, lỗi phần mềm, hoặc hướng dẫn kỹ thuật → gọi search_faq_tool.\n"
        "Không bịa thông tin kỹ thuật."
        f"{context_block}{source_block}"
    )

    # vLLM: small model cannot reliably call tools, so always force retrieval.
    # Conversational/greeting queries are handled naturally in generate_final_answer
    # when retrieval returns nothing — the LLM answers from context there.
    if provider == "vllm":
        messages = list(state.get("messages", []))
        user_message = HumanMessage(content=active_query)
        if not messages:
            response = _forced_search_tool_call(active_query)
            emit_trace(logger, "agent", "end", state, tool_calls=1, outcome="forced_vllm")
            return {"messages": [user_message, response]}
        response = _forced_search_tool_call(active_query)
        emit_trace(logger, "agent", "end", state, tool_calls=1, outcome="forced_vllm_follow_up")
        return {"messages": [response]}

    llm_with_tools = _build_agent_model(provider).bind_tools(
        __import__("apps.agent_runtime.app.graph.tools", fromlist=["available_tools"]).available_tools
    )

    try:
        from shared.py.clients.ollama_client import _write_thinking_log as _wtl

        def _clean_response(msg) -> object:
            # Capture Qwen3 reasoning_content before stripping
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if not reasoning:
                reasoning = (getattr(msg, "additional_kwargs", {}) or {}).get("reasoning_content", "")
            if reasoning:
                _wtl(f"<think>{reasoning}</think>")
            if isinstance(msg.content, str):
                msg.content = _strip_thinking(msg.content)
            return msg

        messages = list(state.get("messages", []))
        if not messages:
            user_message = HumanMessage(content=active_query)
            response = _clean_response(llm_with_tools.invoke([SystemMessage(content=system_prompt), user_message]))
            logger.info("Agent generated first-step response")
            emit_trace(logger, "agent", "end", state, tool_calls=len(getattr(response, "tool_calls", None) or []), outcome="first_step")
            return {"messages": [user_message, response]}

        response = _clean_response(llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages[-8:]))
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
    retrieval_intent = str(payload.get("intent", "") or "")
    active_query = state.get("active_query") or payload.get("query") or state.get("merged_query") or state.get("query_text", "")

    if not chunks:
        reason = payload.get("error") or "Tool retrieval did not return relevant chunks."
        emit_trace(logger, "grade_documents", "end", state, decision="no", chunks=0, reason=reason)
        return {
            "retrieved_chunks": [],
            "is_relevant": "no",
            "grading_reason": reason,
            "last_tool_name": getattr(tool_message, "name", None),
            "retrieval_intent": retrieval_intent,
            "retrieval_debug": _append_retrieval_debug(
                state,
                {"query": active_query, "decision": "no", "reason": reason, "chunks": 0},
            ),
        }

    context_lines = []
    for index, chunk in enumerate(chunks[:3], 1):
        # Use up to 300 chars of content_brief for grader context
        content_preview = chunk.get('content_brief', '')[:300]
        source_key = chunk.get("source_id") or chunk.get("issue_id", "unknown")
        context_lines.append(
            f"{index}. source_id={source_key}, score={float(chunk.get('score', 0)):.2f}\n   Content: {content_preview}"
        )

    # Log chunk previews so we can debug what was graded
    for i, chunk in enumerate(chunks[:3], 1):
        logger.debug(
            "TRACE|node=grade_documents|phase=chunk_preview|rank=%d|source_id=%s|score=%.4f|content=%s",
            i, chunk.get('source_id', chunk.get('issue_id', '?')), float(chunk.get('score', 0)),
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
    grader_model = config.VLLM_LLM_MODEL if provider == "vllm" else config.OLLAMA_GRADER_MODEL
    grader = OllamaChat(
        base_url=llm_base_url,
        model=grader_model,
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
        "retrieval_intent": retrieval_intent,
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
    rewrite_model = config.VLLM_LLM_MODEL if provider == "vllm" else config.OLLAMA_REWRITE_MODEL
    rewriter = OllamaChat(
        base_url=llm_base_url,
        model=rewrite_model,
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

        # Try LLM direct answer before showing the template — handles greetings,
        # meta-questions, and anything that simply didn't match any document.
        try:
            provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
            llm_base_url = config.VLLM_LLM_URL if provider == "vllm" else config.OLLAMA_BASE_URL
            conversation_context = (state.get("conversation_context") or "").strip()
            context_block = f"\n\nLịch sử hội thoại:\n{conversation_context}" if conversation_context else ""
            fallback_system = (
                "Bạn là trợ lý helpdesk EHC. Trả lời tự nhiên, thân thiện bằng tiếng Việt."
                " Nếu là câu hỏi chào hỏi hoặc hội thoại thông thường, trả lời bình thường."
                " Nếu là câu hỏi kỹ thuật mà bạn không có đủ thông tin, hỏi thêm chi tiết thay vì bịa đặt."
                f"{context_block}"
            )
            chat_model = config.VLLM_LLM_MODEL if provider == "vllm" else config.OLLAMA_LLM_MODEL
            chat_client = OllamaChat(
                base_url=llm_base_url,
                model=chat_model,
                timeout=30,
                provider=provider,
            )
            llm_fallback = _strip_thinking(chat_client.generate(
                prompt=state.get("query_text", ""),
                system_prompt=fallback_system,
            ))
            if llm_fallback:
                emit_trace(logger, "generate_final_answer", "end", state, outcome="llm_fallback", sources=0)
                return {
                    "final_answer": llm_fallback,
                    "sources": [],
                    "image_urls": [],
                }
        except Exception as exc:
            logger.warning("LLM fallback failed, using template: %s", exc)

        emit_trace(logger, "generate_final_answer", "end", state, outcome="template_fallback", error=error_reason, sources=0)
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
    issue_id = best_chunk.get("issue_id") or best_chunk.get("source_id", "unknown")
    source_id = best_chunk.get("source_id") or issue_id
    source_type = best_chunk.get("source_type", "faq")
    source_title = best_chunk.get("source_title", "")
    attachment_urls = best_chunk.get("attachment_urls", [])
    source_url = best_chunk.get("source_url", "")
    score = best_chunk.get("score", 0)
    has_doc_context = source_type == "module_doc" or any(
        str(c.get("source_type", "faq") or "faq") == "module_doc" for c in ranked_chunks[:3]
    )
    is_doc_flow = has_doc_context and (
        _is_doc_flow_query(active_query) or _is_progress_followup_query(active_query)
    )
    confidence = _confidence_label(float(score))
    logger.info(
        "TRACE|node=generate_final_answer|phase=best_chunk|source_id=%s|source_type=%s|vec_score=%.4f|confidence=%s",
        source_id, source_type, float(score), confidence,
    )

    top_chunks = _collect_module_doc_context_chunks(ranked_chunks, max_chunks=8) if is_doc_flow else ranked_chunks[:4]
    if is_doc_flow:
        top_chunks = _augment_module_doc_flow_chunks(top_chunks, active_query, max_chunks=8)
    selected_sources = top_chunks if is_doc_flow else _select_similar_sources(ranked_chunks)
    ambiguity_reason = ""

    if _is_ambiguous_low_confidence(float(score), selected_sources) or _should_force_hybrid_for_ambiguous_query(state, selected_sources):
        related_ids = [
            str(s.get("source_id", s.get("issue_id", ""))).strip()
            for s in selected_sources
            if str(s.get("source_id", s.get("issue_id", ""))).strip()
        ]
        has_docx_sources = any(str(s.get("source_type", "faq")) == "module_doc" for s in selected_sources)
        ambiguity_reason = (
            "Nhiều ticket gần nghĩa có điểm tương đương, chưa đủ chắc để chốt nguyên nhân duy nhất"
        )
        if related_ids:
            ambiguity_reason += f" (ứng viên: {', '.join('#' + rid for rid in related_ids[:3])})"

        # For module-doc flow questions, keep going with richer context instead of early short fallback.
        if is_doc_flow:
            logger.info("Module-doc flow query detected: bypass ambiguous fallback and continue with full context")
        else:
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
            if has_docx_sources:
                ambiguous_answer = _build_docx_hybrid_answer(selected_sources, active_query)
            else:
                # Let the LLM reply naturally using the candidate chunks as soft context.
                try:
                    provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
                    llm_base_url = config.VLLM_LLM_URL if provider == "vllm" else config.OLLAMA_BASE_URL
                    conversation_context = (state.get("conversation_context") or "").strip()
                    context_block = f"\n\nLịch sử hội thoại:\n{conversation_context}" if conversation_context else ""
                    ambiguous_system = (
                        "Bạn là trợ lý helpdesk EHC. Trả lời tự nhiên, thân thiện bằng tiếng Việt."
                        " Nếu câu hỏi chưa rõ, hỏi thêm để hiểu vấn đề. Không bịa thông tin kỹ thuật."
                        f"{context_block}"
                    )
                    chat_model = config.VLLM_LLM_MODEL if provider == "vllm" else config.OLLAMA_LLM_MODEL
                    chat_client = OllamaChat(
                        base_url=llm_base_url, model=chat_model, timeout=30, provider=provider,
                    )
                    ambiguous_answer = _strip_thinking(chat_client.generate(
                        prompt=active_query, system_prompt=ambiguous_system,
                    )) or _build_guided_fallback_answer(state, ambiguity_reason)
                except Exception as exc:
                    logger.warning("Ambiguous LLM fallback failed: %s", exc)
                    ambiguous_answer = _build_guided_fallback_answer(state, ambiguity_reason)

            return {
                "final_answer": ambiguous_answer,
                "sources": [
                    {
                        "issue_id": str(chunk.get("issue_id", "unknown")),
                        "source_id": str(chunk.get("source_id", chunk.get("issue_id", "unknown"))),
                        "source_type": str(chunk.get("source_type", "faq")),
                        "source_title": str(chunk.get("source_title", "") or ""),
                        "snippet": str(chunk.get("content_brief", "") or "")[:150],
                        "url": str(chunk.get("source_url", "") or ""),
                        "score": float(chunk.get("score", 0.0) or 0.0),
                    }
                    for chunk in selected_sources
                ],
                "image_urls": attachment_urls,
            }

    # ⭐ NEW: Load images from retrieved chunks
    MAX_IMAGES_PER_QUERY = 15
    image_storage = ImageStorage(base_dir=str(_REPO_ROOT / "data" / "docx_images"))
    all_images = []
    
    for chunk in top_chunks:
        chunk_image_ids = chunk.get("image_ids", [])
        if not chunk_image_ids:
            continue
        
        for img_id in chunk_image_ids:
            if len(all_images) >= MAX_IMAGES_PER_QUERY:
                break
            
            try:
                img_bytes = image_storage.get_image_bytes(img_id)
                if img_bytes:
                    img_metadata = image_storage.get_image_metadata(img_id)
                    all_images.append({
                        "id": img_id,
                        "data": base64.b64encode(img_bytes).decode('utf-8'),
                        "alt_text": img_metadata.get("alt_text", "") if img_metadata else "",
                        "source_file": img_metadata.get("source_file", "") if img_metadata else "",
                    })
                    logger.info(f"Loaded image {img_id} from chunk (alt_text: {len(img_metadata.get('alt_text', '') if img_metadata else '')} chars)")
            except Exception as e:
                logger.warning(f"Failed to load image {img_id}: {e}")
        
        if len(all_images) >= MAX_IMAGES_PER_QUERY:
            break
    
    logger.info(f"Total images loaded for LLM: {len(all_images)}")
    
    context_blocks = []
    for idx, chunk in enumerate(top_chunks, 1):
        chunk_title = str(chunk.get("source_title", "") or "").strip()
        section_path = str(chunk.get("section_path", "") or "").strip()
        chunk_full = str(chunk.get("content_full", "") or "").strip()
        chunk_brief = str(chunk.get("content_brief", "") or "").strip()
        source_id_label = str(chunk.get("source_id", chunk.get("issue_id", "")) or "").strip()
        source_type_label = str(chunk.get("source_type", "faq") or "faq").strip()

        if not chunk_full:
            chunk_full = chunk_brief

        # Keep a detailed but bounded block for the LLM.
        block_lines = [f"Chunk {idx}:"]
        if source_type_label:
            block_lines.append(f"source_type: {source_type_label}")
        if source_id_label:
            block_lines.append(f"source_id: {source_id_label}")
        if chunk_title:
            block_lines.append(f"title: {chunk_title}")
        if section_path:
            block_lines.append(f"section: {section_path}")
        block_lines.append("content:")
        block_lines.append(chunk_full[:1800])
        if chunk_brief and chunk_brief != chunk_full[: len(chunk_brief)]:
            block_lines.append("brief:")
            block_lines.append(chunk_brief[:400])

        context_blocks.append("\n".join(block_lines))

    context_text = "\n\n---\n\n".join(context_blocks)
    user_query = state.get("active_query") or state.get("query_text", "")
    detail_followup = _is_detail_followup(user_query)
    progress_followup = _is_progress_followup_query(user_query)
    ocr_text = state.get("ocr_text")
    grading_reason = state.get("grading_reason") or ""
    conversation_context = (state.get("conversation_context") or "").strip()
    last_source_hint = (state.get("last_source_hint") or "").strip()

    retrieval_assessment = grading_reason or "Không có"
    if ambiguity_reason:
        retrieval_assessment = f"{retrieval_assessment} | {ambiguity_reason}"

    prompt_parts = [f"Câu hỏi: {user_query}"]
    if conversation_context:
        prompt_parts.append(f"\nLịch sử hội thoại:\n{conversation_context}")
    if last_source_hint:
        prompt_parts.append(f"\nChủ đề trước: {last_source_hint}")
    if ocr_text:
        prompt_parts.append(f"\nNội dung ảnh màn hình:\n{ocr_text}")
    prompt_parts.append(f"\nThông tin tra cứu:\n{context_text}")
    if retrieval_assessment and retrieval_assessment != "Không có":
        prompt_parts.append(f"\nĐánh giá tìm kiếm: {retrieval_assessment}")
    prompt_parts.append("\nDựa vào thông tin trên, trả lời câu hỏi của người dùng.")
    prompt = "\n".join(prompt_parts)

    final_answer = ""
    has_prefilled_deterministic_answer = False
    if is_doc_flow and progress_followup:
        progress_answer = _build_progress_followup_answer(top_chunks, user_query)
        if progress_answer:
            final_answer = progress_answer
            has_prefilled_deterministic_answer = True
            logger.info("Generated progress follow-up answer from doc flow context")
    elif is_doc_flow:
        deterministic_doc_flow = _build_doc_flow_answer_by_sections(top_chunks, max_sections=5, max_points_per_section=3)
        if deterministic_doc_flow:
            final_answer = deterministic_doc_flow
            has_prefilled_deterministic_answer = True
            logger.info("Generated section-based deterministic doc-flow answer")

    try:
        provider = (state.get("model_provider") or config.MODEL_PROVIDER or "ollama").strip().lower()
        llm_base_url = config.VLLM_LLM_URL if provider == "vllm" else config.OLLAMA_BASE_URL
        
        natural_system_prompt = (
            "/no_think\n"
            "Bạn là trợ lý helpdesk nội bộ của hệ thống y tế EHC, hỗ trợ nhân viên tra cứu quy trình và xử lý sự cố phần mềm HIS.\n\n"
            "Nguyên tắc trả lời:\n"
            "- Trả lời tự nhiên, thân thiện như đồng nghiệp — không cứng nhắc, không mẫu hóa.\n"
            "- Dùng thông tin từ tài liệu/ticket được cung cấp. Không bịa đặt thông tin kỹ thuật.\n"
            "- Nếu câu hỏi hỏi về các bước, liệt kê rõ từng bước theo thứ tự.\n"
            "- Nếu câu hỏi hỏi nguyên nhân lỗi, nêu nguyên nhân cụ thể trước rồi mới hướng dẫn.\n"
            "- Không dùng Markdown (***, ##, ---). Không thêm 'Nguồn:' hay citation ở cuối.\n"
            "- Ngắn gọn, đủ thông tin — không lan man."
        )
        
        # If progress follow-up already has a deterministic high-quality answer, skip LLM.
        if final_answer:
            llm_answer = final_answer
        # ⭐ Use vision model if we have images from DOCX chunks
        elif all_images:
            logger.info(f"Using vision model with {len(all_images)} images from DOCX chunks")
            vision_llm_model = config.VLLM_VISION_MODEL if provider == "vllm" else config.OLLAMA_LLM_MODEL
            vision_client = OllamaVision(
                base_url=llm_base_url,
                model=vision_llm_model,
                timeout=180,
                provider=provider,
            )
            
            # Build enhanced prompt with image context
            image_context_lines = []
            for idx, img in enumerate(all_images, 1):
                alt_text = img.get("alt_text", "").strip()
                source_file = img.get("source_file", "").strip()
                if alt_text:
                    image_context_lines.append(f"Image {idx} (from {source_file}): {alt_text[:200]}")
                else:
                    image_context_lines.append(f"Image {idx} (from {source_file}): [No OCR text available]")
            
            enhanced_prompt_payload = dict(prompt_payload)
            enhanced_prompt_payload["request"]["docx_images_context"] = "\n".join(image_context_lines)
            enhanced_prompt = json.dumps(enhanced_prompt_payload, ensure_ascii=False, indent=2)
            
            # Extract base64 image data
            image_data_list = [img["data"] for img in all_images]
            
            llm_answer = vision_client.generate_with_images(
                prompt=enhanced_prompt,
                images=image_data_list,
                system_prompt=natural_system_prompt,
            )
            logger.info(f"Vision model generated answer with {len(all_images)} images")
        else:
            # Text-only mode (faster)
            logger.info("Using text-only model (no DOCX images)")
            chat_llm_model = config.VLLM_LLM_MODEL if provider == "vllm" else config.OLLAMA_LLM_MODEL
            chat_client = OllamaChat(
                base_url=llm_base_url,
                model=chat_llm_model,
                timeout=120,
                provider=provider,
            )
            llm_answer = chat_client.generate(
                prompt=prompt,
                system_prompt=natural_system_prompt,
            )
        llm_answer = _strip_thinking(llm_answer)
        if is_doc_flow and (not has_prefilled_deterministic_answer) and len([ln for ln in llm_answer.splitlines() if ln.strip()]) < 6:
            deterministic = _build_stepwise_answer_from_chunks(top_chunks, max_steps=10)
            if deterministic:
                llm_answer = deterministic
        elif len(llm_answer.strip()) < 120:
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
                "issue_id": str(chunk.get("issue_id", chunk.get("source_id", "unknown"))),
                "source_id": str(chunk.get("source_id", chunk.get("issue_id", "unknown"))),
                "source_type": str(chunk.get("source_type", "faq")),
                "source_title": str(chunk.get("source_title", "") or ""),
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
