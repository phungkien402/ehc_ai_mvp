"""API Gateway Pydantic schemas."""

from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    """User question request."""
    query: str
    image_base64: Optional[str] = None  # Base64-encoded image bytes
    user_id: Optional[str] = None
    channel: Optional[str] = None
    session_id: Optional[str] = None


class OCRRequest(BaseModel):
    """Standalone OCR test request."""
    image_base64: str
    model: Optional[str] = None


class SourceInfo(BaseModel):
    """Source chunk info in response."""
    issue_id: str
    snippet: str
    url: str
    score: float


class AskResponse(BaseModel):
    """Answer response."""
    answer: str
    sources: List[SourceInfo]
    image_urls: List[str]
    error: Optional[str] = None
    execution_time_ms: float
    is_relevant: Optional[str] = None
    rewrite_attempts: int = 0
    rewritten_query: Optional[str] = None
    grading_reason: Optional[str] = None
    ocr_text: Optional[str] = None


class OCRResponse(BaseModel):
    """Standalone OCR response."""
    ocr_text: str
    model: str
    execution_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    qdrant_ok: bool
    ollama_ok: bool
    message: str


class DashboardOverviewResponse(BaseModel):
    """System overview for dashboard."""

    api_status: str
    uptime_seconds: int
    qdrant_ok: bool
    ollama_ok: bool
    active_models: List[str]
    recent_latency_ms: Optional[float] = None


class DashboardLogEntry(BaseModel):
    """Single parsed log line."""

    cursor: int
    timestamp: str
    logger: str
    level: str
    message: str
    raw: str


class DashboardLogsResponse(BaseModel):
    """Paged dashboard logs response."""

    entries: List[DashboardLogEntry]
    next_cursor: int


class WorkflowDiagramResponse(BaseModel):
    """Workflow diagram payload."""

    content: str
    last_modified_epoch_ms: int
