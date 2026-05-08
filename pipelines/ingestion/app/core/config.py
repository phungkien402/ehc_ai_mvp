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
    QDRANT_DOCS_COLLECTION: str = os.getenv("QDRANT_DOCS_COLLECTION", "ehc_module_docs")
    QDRANT_VECTOR_SIZE: int = 1024
    
    # Model provider
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "ollama").lower()

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL: str = "bge-m3"
    OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")

    # vLLM (OpenAI-compatible API)
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    VLLM_EMBEDDING_URL: str = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8001")
    VLLM_VISION_URL: str = os.getenv("VLLM_VISION_URL", "http://localhost:8001")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")

    # DOCX source
    DOCX_INPUT_DIR: str = os.getenv("DOCX_INPUT_DIR", "data/module_docs_raw")
    DOCX_OCR_ENABLED: bool = os.getenv("DOCX_OCR_ENABLED", "true").lower() == "true"
    DOCX_OCR_MAX_IMAGES: int = int(os.getenv("DOCX_OCR_MAX_IMAGES", "6"))
    DOCX_OCR_BACKEND: str = os.getenv("DOCX_OCR_BACKEND", "auto").lower()
    
    # Ingestion params
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
config = IngestionConfig()
