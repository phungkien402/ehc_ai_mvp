"""Workflow routing tests for the Self-RAG graph."""

from langchain_core.messages import AIMessage

from apps.agent_runtime.app.graph.router import route_after_grading, should_continue
from apps.agent_runtime.app.graph.workflow import build_workflow


def test_workflow_initialization():
    """Workflow should compile with the new Self-RAG nodes."""

    agent = build_workflow()
    assert agent is not None


def test_should_continue_routes_to_tools_when_agent_requests_tool():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_faq_tool", "args": {"query": "test"}, "id": "tool-1", "type": "tool_call"}],
            )
        ]
    }

    assert should_continue(state) == "continue"


def test_route_after_grading_rewrites_before_budget_exhausted():
    state = {
        "is_relevant": "no",
        "rewrite_attempts": 0,
        "max_rewrite_attempts": 2,
    }

    assert route_after_grading(state) == "rewrite"


def test_route_after_grading_generates_when_relevant():
    state = {
        "is_relevant": "yes",
        "rewrite_attempts": 1,
        "max_rewrite_attempts": 2,
    }

    assert route_after_grading(state) == "generate"


def test_route_after_grading_generates_when_budget_exhausted():
    state = {
        "is_relevant": "no",
        "rewrite_attempts": 2,
        "max_rewrite_attempts": 2,
    }

    assert route_after_grading(state) == "generate"
