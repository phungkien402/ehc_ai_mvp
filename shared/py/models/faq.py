from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib


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
class ModuleDocSection:
    """Normalized DOCX section extracted from module user guides."""
    source_id: str
    module_name: str
    section_title: str
    content: str
    source_file: str
    source_url: str
    created_at: datetime
    updated_at: datetime
    attachment_urls: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)  # Image file IDs stored separately
    metadata: dict = field(default_factory=dict)


@dataclass
class FAQChunk:
    """Processed text chunk ready for embedding."""
    chunk_id: str                    # UUID-based unique ID per chunk
    issue_id: str                    # Source FAQ issue ID
    content: str                     # Plain text to embed (subject + desc + fields merged)
    content_brief: str               # First 200 chars (for display)
    chunk_index: int                 # 0-based chunk number within an issue (if split)
    content_full: str = ""          # Longer content for LLM grounding / citations
    source_id: str = ""              # Generic source ID (FAQ issue, DOCX section)
    source_type: str = "faq"         # faq | module_doc
    source_title: str = ""           # Section or ticket title
    section_path: str = ""           # Path-like section marker for docs
    image_ids: List[str] = field(default_factory=list)  # Related image IDs from storage
    metadata: dict = field(default_factory=dict)  # Extra metadata: {"source_url": "...", "attachment_urls": [...]}

    def __post_init__(self):
        if not self.source_id:
            self.source_id = self.issue_id
        if not self.issue_id:
            self.issue_id = self.source_id


@dataclass
class QdrantPayload:
    """Exact structure stored in Qdrant point."""
    chunk_id: str
    issue_id: str
    source_id: str
    source_type: str
    source_title: str
    section_path: str
    content_full: str
    content_brief: str
    attachment_urls: List[str]
    source_url: str
    embedding_model: str            # e.g., "bge-m3"
    created_at: str                 # ISO format
    image_ids: List[str] = field(default_factory=list)  # Related image IDs
    
    def to_dict(self) -> dict:
        """Convert to Qdrant payload dict."""
        return {
            "chunk_id": self.chunk_id,
            "issue_id": self.issue_id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "section_path": self.section_path,
            "content_full": self.content_full,
            "content_brief": self.content_brief,
            "attachment_urls": self.attachment_urls,
            "source_url": self.source_url,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
            "image_ids": self.image_ids,
        }


def stable_source_id(*parts: str, prefix: str = "src") -> str:
    """Create deterministic IDs so re-ingestion updates the same logical source."""
    base = "|".join([(p or "").strip() for p in parts])
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


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
