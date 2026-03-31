"""Qdrant vector database wrapper."""

import logging
import os
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


class QdrantWrapper:
    """Qdrant connection pool + search helpers."""
    
    def __init__(self, url: str = None, api_key: str = None):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL (default from QDRANT_URL env var)
            api_key: API key if needed (default None for local)
        """
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30
        )
        logger.info(f"Initialized Qdrant client at {self.url}")
    
    def ensure_collection_exists(self, collection_name: str, vector_size: int = 1024):
        """
        Create collection if not exists.
        
        Args:
            collection_name: e.g., "ehc_faq"
            vector_size: Dimension of embedding vectors (bge-m3 = 1024)
        """
        try:
            self.client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' exists.")
        except Exception:
            logger.info(f"Creating collection '{collection_name}'...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection '{collection_name}' created with {vector_size} dimensions")
    
    def upsert_chunk(self, collection_name: str, point_id: int, vector: List[float], payload: dict):
        """
        Insert or update a single chunk.
        
        Args:
            collection_name: e.g., "ehc_faq"
            point_id: Unique integer ID
            vector: 1024-dim embedding
            payload: QdrantPayload.to_dict()
        """
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        logger.debug(f"Upserted point {point_id} to '{collection_name}'")
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = 0.5,
    ) -> List[dict]:
        """
        Search Qdrant and return matching chunks.
        
        Returns:
            [{"content_brief": str, "issue_id": str, "attachment_urls": [...], "score": float}, ...]
        """
        try:
            query_kwargs = {
                "collection_name": collection_name,
                "query": query_vector,
                "limit": limit,
            }
            if score_threshold is not None:
                query_kwargs["score_threshold"] = score_threshold

            results = self.client.query_points(**query_kwargs).points
            
            chunks = []
            for result in results:
                chunks.append({
                    "content_brief": result.payload.get("content_brief", ""),
                    "issue_id": result.payload.get("issue_id", "unknown"),
                    "attachment_urls": result.payload.get("attachment_urls", []),
                    "source_url": result.payload.get("source_url", ""),
                    "score": result.score
                })
            
            logger.info(f"Search returned {len(chunks)} chunks from '{collection_name}'")
            return chunks
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def collection_info(self, collection_name: str) -> dict:
        """Get collection stats (point count, etc.)."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "point_count": info.points_count,
                "vector_size": info.config.params.vectors.size
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
