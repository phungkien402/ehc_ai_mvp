"""Session memory query enrichment behavior tests."""

import sys

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp/apps/api_gateway")

from app.core.session_memory import SessionMemoryStore


def _build_seeded_store() -> SessionMemoryStore:
    store = SessionMemoryStore()
    store.remember_turn(
        session_key="telegram:user-1",
        user_text="hướng dẫn module PACS từ đầu đến cuối",
        answer_text="Các bước cấu hình PACS...",
        sources=[{"issue_id": "docx_cb9b82fbee04e66c", "snippet": "HDSD kết nối PACS server"}],
    )
    return store


def test_enrich_query_keeps_standalone_task_query_clean() -> None:
    store = _build_seeded_store()

    query = "cách gộp mã bệnh nhân"
    enriched = store.enrich_query("telegram:user-1", query)

    assert enriched == query


def test_enrich_query_adds_context_for_explicit_followup() -> None:
    store = _build_seeded_store()

    enriched = store.enrich_query("telegram:user-1", "chi tiết hơn")

    assert "Ngữ cảnh hội thoại gần nhất" in enriched
    assert "ticket #docx_cb9b82fbee04e66c" in enriched
