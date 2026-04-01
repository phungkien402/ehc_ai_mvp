"""LangGraph workflow state definition."""

import operator
from typing import Annotated, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class WorkflowState(TypedDict, total=False):
    """LangGraph state definition."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Input
    trace_id: str
    model_provider: str
    rollout_bucket: int
    query_text: str
    image_bytes: Optional[bytes]
    conversation_context: str
    last_source_hint: str
    
    # Intermediate
    cleaned_query: str
    ocr_text: Optional[str]
    merged_query: str
    active_query: str
    rewritten_query: Optional[str]
    rewrite_attempts: int
    max_rewrite_attempts: int
    retrieved_chunks: List[dict]
    is_relevant: str
    grading_reason: Optional[str]
    retrieval_debug: List[dict]
    last_tool_name: Optional[str]
    
    # Output
    final_answer: str
    sources: List[dict]
    image_urls: List[str]
    
    # Metadata
    error: Optional[str]
    duration_seconds: float
