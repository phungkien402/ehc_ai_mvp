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
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8080")
    VLLM_LLM_URL: str = os.getenv("VLLM_LLM_URL", "http://localhost:8080")
    VLLM_VISION_URL: str = os.getenv("VLLM_VISION_URL", "http://localhost:8080")
    VLLM_EMBEDDING_URL: str = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8081")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")
    VLLM_LLM_MODEL: str = os.getenv("VLLM_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    VLLM_VISION_MODEL: str = os.getenv("VLLM_VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    VLLM_EMBEDDING_MODEL: str = os.getenv("VLLM_EMBEDDING_MODEL", "BAAI/bge-m3")
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "ehc_faq"
    QDRANT_DOCS_COLLECTION: str = os.getenv("QDRANT_DOCS_COLLECTION", "ehc_module_docs")
    DOCS_RETRIEVAL_ENABLED: bool = os.getenv("DOCS_RETRIEVAL_ENABLED", "false").lower() == "true"
    
    # Runtime params
    RETRIEVAL_TOP_K: int = 5
    DOCS_RETRIEVAL_TOP_K: int = int(os.getenv("DOCS_RETRIEVAL_TOP_K", "4"))
    SCORE_THRESHOLD: float = 0.5
    GRADE_MIN_RELEVANCE: float = 0.5
    GRADE_MIN_LEXICAL_OVERLAP: float = 0.2
    MAX_REWRITE_ATTEMPTS: int = 2
    
config = RuntimeConfig()
