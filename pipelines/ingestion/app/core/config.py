"""Ingestion pipeline configuration."""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    """Configuration for ingestion pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )
    
    # Redmine
    REDMINE_URL: str = os.getenv("REDMINE_URL", "http://localhost:3000")
    REDMINE_API_KEY: str = os.getenv("REDMINE_API_KEY", "")
    
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION: str = "ehc_faq"
    QDRANT_VECTOR_SIZE: int = 1024
    
    # Model provider
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "ollama").lower()

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3"

    # vLLM (OpenAI-compatible API)
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    VLLM_EMBEDDING_URL: str = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8001")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")
    
    # Ingestion params
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
config = IngestionConfig()
