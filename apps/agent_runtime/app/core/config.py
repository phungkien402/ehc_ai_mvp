"""Agent runtime configuration."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeConfig(BaseSettings):
    """Configuration for agent runtime."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )
    
    # Model provider
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "ollama").lower()

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
    OLLAMA_VISION_MODEL: str = "qwen3.5:9b"
    OLLAMA_LLM_MODEL: str = "qwen2.5:14b"
    OLLAMA_GRADER_MODEL: str = "qwen2.5:14b"
    OLLAMA_REWRITE_MODEL: str = "qwen2.5:14b"

    # vLLM (OpenAI-compatible API)
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    VLLM_LLM_URL: str = os.getenv("VLLM_LLM_URL", "http://localhost:8000")
    VLLM_VISION_URL: str = os.getenv("VLLM_VISION_URL", "http://localhost:8001")
    VLLM_EMBEDDING_URL: str = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8001")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "ehc_faq"
    
    # Runtime params
    RETRIEVAL_TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.5
    GRADE_MIN_RELEVANCE: float = 0.5
    GRADE_MIN_LEXICAL_OVERLAP: float = 0.2
    MAX_REWRITE_ATTEMPTS: int = 2
    
config = RuntimeConfig()
