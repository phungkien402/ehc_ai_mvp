"""API Gateway configuration."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """Configuration for API gateway."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )
    
    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "ehc_faq"
    
    # Model provider
    MODEL_PROVIDER: str = os.getenv("MODEL_PROVIDER", "ollama").lower()

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_VISION_MODEL: str = "qwen3.5:9b"

    # vLLM (OpenAI-compatible API)
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    VLLM_LLM_URL: str = os.getenv("VLLM_LLM_URL", "http://localhost:8000")
    VLLM_VISION_URL: str = os.getenv("VLLM_VISION_URL", "http://localhost:8001")
    VLLM_EMBEDDING_URL: str = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8001")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")

    # Rollout controls (used when MODEL_PROVIDER=ollama)
    ROLLOUT_ENABLED: bool = os.getenv("ROLLOUT_ENABLED", "false").lower() == "true"
    ROLLOUT_PERCENT_VLLM: float = float(os.getenv("ROLLOUT_PERCENT_VLLM", "0"))
    ROLLOUT_STICKY_KEY: str = os.getenv("ROLLOUT_STICKY_KEY", "session_id")
    MAX_REWRITE_ATTEMPTS: int = 2
    
config = APIConfig()
