from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class FAQSource:
    """Original FAQ record from Redmine."""
    issue_id: str                    # Redmine issue ID (e.g., "12345")
    subject: str                     # Issue subject (title)
    description: str                 # Issue full description
    custom_fields: dict              # Any extra Redmine custom fields
    attachment_urls: List[str]       # URLs to attached images
    url: str                          # Link to issue in Redmine
    created_at: datetime             # Created date
    updated_at: datetime             # Last updated date


@dataclass
class FAQChunk:
    """Processed text chunk ready for embedding."""
    chunk_id: str                    # UUID-based unique ID per chunk
    issue_id: str                    # Source FAQ issue ID
    content: str                     # Plain text to embed (subject + desc + fields merged)
    content_brief: str               # First 200 chars (for display)
    chunk_index: int                 # 0-based chunk number within an issue (if split)
    metadata: dict = field(default_factory=dict)  # Extra metadata: {"source_url": "...", "attachment_urls": [...]}


@dataclass
class QdrantPayload:
    """Exact structure stored in Qdrant point."""
    chunk_id: str
    issue_id: str
    content_brief: str
    attachment_urls: List[str]
    source_url: str
    embedding_model: str            # e.g., "bge-m3"
    created_at: str                 # ISO format
    
    def to_dict(self) -> dict:
        """Convert to Qdrant payload dict."""
        return {
            "chunk_id": self.chunk_id,
            "issue_id": self.issue_id,
            "content_brief": self.content_brief,
            "attachment_urls": self.attachment_urls,
            "source_url": self.source_url,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
        }


@dataclass
class WorkflowState:
    """LangGraph state: input + intermediate + output."""
    # Input
    query_text: str
    image_bytes: Optional[bytes] = None
    
    # Intermediate
    cleaned_query: str = ""
    ocr_text: Optional[str] = None
    merged_query: str = ""
    retrieved_chunks: List[dict] = field(default_factory=list)  # [{"text": str, "metadata": {...}}, ...]
    
    # Output
    final_answer: str = ""
    sources: List[dict] = field(default_factory=list)          # [{"issue_id": "...", "url": "..."}, ...]
    image_urls: List[str] = field(default_factory=list)
    
    # Metadata
    error: Optional[str] = None
    duration_seconds: float = 0.0
