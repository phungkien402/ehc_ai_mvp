"""Text processing utilities."""

import logging
import re
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def normalize_vietnamese(text: str) -> str:
    """
    Normalize Vietnamese query: remove common diacritics, lowercase.
    
    Simple version: just lowercase + strip whitespace.
    (Full accent removal would use `unidecode` library if needed)
    
    Args:
        text: Input Vietnamese or English text
    
    Returns:
        Normalized text
    """
    text = text.strip()
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: Full text (e.g., concatenated subject + description)
        chunk_size: Max chars per chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunk strings
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.debug(f"Split text into {len(chunks)} chunks")
    return chunks


def merge_query_with_ocr(text_query: str, ocr_text: Optional[str]) -> str:
    """
    Merge normalized text query with OCR extracted text.
    
    Args:
        text_query: Normalized user query
        ocr_text: Extracted text from image (if any)
    
    Returns:
        Merged query string
    """
    if not ocr_text:
        return text_query

    # Ignore OCR payloads that are mostly punctuation/noise (e.g., "!!!!").
    alpha_count = len(re.findall(r"[A-Za-zÀ-ỹà-ỹ0-9]", ocr_text))
    if alpha_count < 6:
        return text_query

    # Keep OCR as supporting signal only; cap length to avoid drowning user query.
    ocr_compact = re.sub(r"\s+", " ", ocr_text).strip()[:220]
    merged = f"{text_query} {ocr_compact}" if text_query else ocr_compact
    merged = normalize_vietnamese(merged)
    return merged


def compose_faq_content(subject: str, description: str, custom_fields: dict) -> str:
    """
    Compose full text for FAQ chunk from Redmine fields.
    
    Args:
        subject: Issue subject
        description: Issue description
        custom_fields: Dict of field name → value
    
    Returns:
        Combined text for embedding
    """
    parts = [subject, description]
    
    for key, value in custom_fields.items():
        if value:
            parts.append(f"{key}: {value}")
    
    return "\n\n".join(parts)
