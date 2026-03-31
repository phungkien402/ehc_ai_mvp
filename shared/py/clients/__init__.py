"""Shared clients package."""

from .qdrant_client import QdrantWrapper
from .ollama_client import OllamaEmbeddings, OllamaVision

__all__ = ["QdrantWrapper", "OllamaEmbeddings", "OllamaVision"]
