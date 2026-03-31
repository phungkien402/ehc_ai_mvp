"""LangGraph workflow compilation."""

import logging
import sys

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp/apps/agent_runtime")

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from apps.agent_runtime.app.graph.tools import available_tools
from apps.agent_runtime.app.graph.state import WorkflowState
from apps.agent_runtime.app.graph.router import route_after_grading, should_continue
from apps.agent_runtime.app.graph.nodes import call_agent, extract_ocr_if_image, generate_final_answer, grade_documents, rewrite_query

logger = logging.getLogger(__name__)


def build_workflow():
    """Build and compile LangGraph workflow."""
    
    # Create graph
    graph = StateGraph(WorkflowState)
    
    # Add nodes
    graph.add_node("extract_ocr_if_image", extract_ocr_if_image)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", ToolNode(available_tools))
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_final_answer", generate_final_answer)
    
    # Add edges
    graph.add_edge(START, "extract_ocr_if_image")
    graph.add_edge("extract_ocr_if_image", "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": "generate_final_answer",
        }
    )

    graph.add_edge("tools", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate_final_answer",
            "rewrite": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "agent")
    graph.add_edge("generate_final_answer", END)
    
    # Compile
    agent = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    
    return agent


# Global agent instance
_agent = None

def get_agent():
    """Get or initialize agent (lazy load)."""
    global _agent
    if _agent is None:
        _agent = build_workflow()
    return _agent
