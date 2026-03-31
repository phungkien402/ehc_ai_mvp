"""FAQ chunking processor."""

import logging
import uuid
from typing import List

import sys
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from shared.py.models.faq import FAQSource, FAQChunk
from shared.py.utils.text import compose_faq_content, chunk_text, normalize_vietnamese

logger = logging.getLogger(__name__)


class FAQChunker:
    """Convert FAQ issues to embedable chunks."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_faq(self, faq: FAQSource) -> List[FAQChunk]:
        """
        Convert single FAQ issue into list of chunks.
        
        Args:
            faq: FAQSource from Redmine
        
        Returns:
            List of FAQChunk objects ready for embedding
        """
        
        # Step 1: Compose full text
        full_text = compose_faq_content(
            subject=faq.subject,
            description=faq.description,
            custom_fields=faq.custom_fields
        )
        
        # Step 2: Normalize
        normalized = normalize_vietnamese(full_text)
        
        # Step 3: Split into chunks
        chunk_texts = chunk_text(normalized, self.chunk_size, self.chunk_overlap)
        
        # Step 4: Convert to FAQChunk objects
        chunks = []
        for idx, chunk_text_str in enumerate(chunk_texts):
            chunk = FAQChunk(
                chunk_id=str(uuid.uuid4()),
                issue_id=faq.issue_id,
                content=chunk_text_str,
                content_brief=chunk_text_str[:200] if len(chunk_text_str) > 200 else chunk_text_str,
                chunk_index=idx,
                metadata={
                    "source_url": faq.url,
                    "attachment_urls": faq.attachment_urls,
                    "created_at": faq.created_at.isoformat(),
                    "updated_at": faq.updated_at.isoformat()
                }
            )
            chunks.append(chunk)
        
        logger.debug(f"Issue #{faq.issue_id} split into {len(chunks)} chunks")
        return chunks
    
    def process_faq_batch(self, faqs: List[FAQSource]) -> List[FAQChunk]:
        """Process multiple FAQs and flatten result."""
        all_chunks = []
        for faq in faqs:
            all_chunks.extend(self.process_faq(faq))
        logger.info(f"Processed {len(faqs)} FAQs into {len(all_chunks)} chunks")
        return all_chunks
