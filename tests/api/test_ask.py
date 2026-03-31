"""API ask endpoint tests."""

import sys

from fastapi.testclient import TestClient

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp/apps/api_gateway")

from app import app


class _FakeAgent:
    async def ainvoke(self, state):
        return {
            **state,
            "final_answer": "Tra loi mau",
            "sources": [
                {
                    "issue_id": "123",
                    "snippet": "Snippet",
                    "url": "http://example.com",
                    "score": 0.91,
                }
            ],
            "image_urls": ["http://example.com/image.png"],
            "is_relevant": "yes",
            "rewrite_attempts": 1,
            "rewritten_query": "query viet lai",
            "grading_reason": "Tai lieu phu hop",
        }


def test_ask_simple_query(monkeypatch):
    """Ask endpoint should preserve legacy fields and expose new metadata."""

    monkeypatch.setattr("app.api.routes.get_agent", lambda: _FakeAgent())
    client = TestClient(app)

    response = client.post("/api/v1/ask", json={"query": "Loi dang nhap"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["answer"] == "Tra loi mau"
    assert payload["sources"][0]["issue_id"] == "123"
    assert payload["image_urls"] == ["http://example.com/image.png"]
    assert payload["is_relevant"] == "yes"
    assert payload["rewrite_attempts"] == 1
    assert payload["rewritten_query"] == "query viet lai"
    assert payload["grading_reason"] == "Tai lieu phu hop"


def test_ask_with_image(monkeypatch):
    """Ask endpoint should accept image input without changing response contract."""

    monkeypatch.setattr("app.api.routes.get_agent", lambda: _FakeAgent())
    client = TestClient(app)

    response = client.post(
        "/api/v1/ask",
        json={"query": "Loi toa thuoc", "image_base64": "aGVsbG8="},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Tra loi mau"
