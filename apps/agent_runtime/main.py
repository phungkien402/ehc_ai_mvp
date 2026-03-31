"""Agent runtime for EHC AI."""

import asyncio
import logging
import sys
import uuid

sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp/apps/agent_runtime")

from apps.agent_runtime.app.core.config import config
from apps.agent_runtime.app.graph.workflow import get_agent
from shared.py.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO", log_file="logs/runtime.log")
logger = logging.getLogger(__name__)


async def run_query(query_text: str, image_bytes=None):
    """Run a single query through the workflow."""
    
    agent = get_agent()
    
    state = {
        "messages": [],
        "trace_id": uuid.uuid4().hex[:8],
        "query_text": query_text,
        "image_bytes": image_bytes,
        "cleaned_query": "",
        "ocr_text": None,
        "merged_query": "",
        "active_query": "",
        "rewritten_query": None,
        "rewrite_attempts": 0,
        "max_rewrite_attempts": config.MAX_REWRITE_ATTEMPTS,
        "retrieved_chunks": [],
        "is_relevant": "no",
        "grading_reason": None,
        "retrieval_debug": [],
        "last_tool_name": None,
        "final_answer": "",
        "sources": [],
        "image_urls": [],
        "error": None,
        "duration_seconds": 0.0
    }
    
    logger.info(f"Invoking workflow: {query_text[:100]}")
    result = await agent.ainvoke(state)
    
    return result


if __name__ == "__main__":
    # Example: run a test query
    result = asyncio.run(run_query("Làm sao để tìm tồn kho thuốc?"))
    print("\n=== Result ===")
    print(f"Answer: {result.get('final_answer')}")
    print(f"Sources: {result.get('sources')}")
