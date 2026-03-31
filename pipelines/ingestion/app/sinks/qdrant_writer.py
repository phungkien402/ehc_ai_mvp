"""Qdrant writer for ingestion pipeline."""

import logging
import uuid
from typing import List

import sys
sys.path.insert(0, "/home/phungkien/ehc_ai_mvp")

from shared.py.models.faq import FAQChunk, QdrantPayload
from shared.py.clients.qdrant_client import QdrantWrapper
from shared.py.clients.ollama_client import OllamaEmbeddings

logger = logging.getLogger(__name__)


class QdrantWriter:
    """Embed chunks and write to Qdrant."""
    
    def __init__(self, qdrant_url: str, ollama_url: str, collection_name: str = "ehc_faq", 
                 embedding_model: str = "bge-m3", batch_size: int = 10):
        self.qdrant = QdrantWrapper(url=qdrant_url)
        self.embeddings = OllamaEmbeddings(base_url=ollama_url, model=embedding_model)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size
    
    def write_chunks(self, chunks: List[FAQChunk]) -> int:
        """
        Embed chunks and upsert to Qdrant.
        
        Args:
            chunks: List of FAQChunk objects
        
        Returns:
            Number of chunks successfully written
        """
        
        # Ensure collection exists
        self.qdrant.ensure_collection_exists(self.collection_name, vector_size=1024)
        
        # Batch embed + upsert
        written_count = 0
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            
            try:
                # Extract texts
                texts = [chunk.content for chunk in batch]
                
                # Embed batch
                vectors = self.embeddings.embed_batch(texts)
                
                if len(vectors) != len(batch):
                    logger.error(f"Embedding count mismatch: got {len(vectors)}, expected {len(batch)}")
                    continue
                
                # Upsert to Qdrant
                for chunk, vector in zip(batch, vectors):
                    point_id = int(uuid.uuid4().int % (2 ** 31))  # Hash UUID to int32
                    
                    payload = QdrantPayload(
                        chunk_id=chunk.chunk_id,
                        issue_id=chunk.issue_id,
                        content_brief=chunk.content_brief,
                        attachment_urls=chunk.metadata.get("attachment_urls", []),
                        source_url=chunk.metadata.get("source_url", ""),
                        embedding_model=self.embedding_model,
                        created_at=chunk.metadata.get("created_at", "")
                    )
                    
                    self.qdrant.upsert_chunk(
                        collection_name=self.collection_name,
                        point_id=point_id,
                        vector=vector,
                        payload=payload.to_dict()
                    )
                    written_count += 1
                
                logger.info(f"Upserted batch {i // self.batch_size + 1}: {len(batch)} chunks")
            
            except Exception as e:
                logger.error(f"Failed to process batch {i // self.batch_size + 1}: {e}")
                continue
        
        # Log final collection stats
        info = self.qdrant.collection_info(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' now has {info.get('point_count', 0)} points")
        
        return written_count
