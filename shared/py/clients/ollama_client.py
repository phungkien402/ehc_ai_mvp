"""Ollama embedding and vision client wrappers."""

import base64
import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = (
    "You are analyzing a Vietnamese hospital software screenshot. "
    "Your ONLY job is to describe what you see — do NOT suggest causes or solutions. "
    "Respond in this exact 3-section format:\n"
    "[TEXT] <all visible text in the image, preserving line structure>\n"
    "[ISSUE] <1-2 sentences: what screen/UI is shown, what error or problem is visible, any error codes or status messages>\n"
    "[UI] <1-2 sentences: describe the visual layout and state — e.g. modal popup, disabled buttons, highlighted fields, broken layout, warning icons, progress bars>\n"
    "If any text is blurry, approximate the closest readable characters. "
    "Do NOT include [CAUSE] or [ACTION] — those are determined from the support ticket database."
)


class OllamaEmbeddings:
    """Wrapper for Ollama embedding model."""
    
    def __init__(self, base_url: str = None, model: str = "bge-m3", timeout: int = 60):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.timeout = timeout
        logger.info(f"Initialized OllamaEmbeddings: {self.base_url} (model: {model})")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            text: Vietnamese or any text string
        
        Returns:
            1024-dim float vector
        
        Raises:
            RuntimeError if request fails or timeout
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.timeout
            )
            if response.status_code == 404:
                # Backward compatibility for Ollama versions that expose /api/embeddings.
                legacy = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                legacy.raise_for_status()
                data = legacy.json()
                embedding = data["embedding"]
            else:
                response.raise_for_status()
                data = response.json()
                embedding = data["embeddings"][0]
            logger.debug(f"Embedded text: {len(text)} chars -> {len(embedding)} dims")
            return embedding
        except requests.Timeout:
            logger.error(f"Embedding timeout (>{self.timeout}s)")
            raise RuntimeError("Embedding timeout")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise RuntimeError(f"Embedding error: {e}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of strings
        
        Returns:
            List of 1024-dim vectors
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout
            )
            if response.status_code == 404:
                # Legacy endpoint only supports one prompt per request.
                embeddings: List[List[float]] = []
                for text in texts:
                    legacy = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                        timeout=self.timeout,
                    )
                    legacy.raise_for_status()
                    embeddings.append(legacy.json()["embedding"])
            else:
                response.raise_for_status()
                data = response.json()
                embeddings = data["embeddings"]
            logger.info(f"Batch embedded {len(texts)} texts -> {len(embeddings)} vectors")
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return []


class OllamaVision:
    """Wrapper for Ollama vision model (OCR on images)."""
    
    def __init__(self, base_url: str = None, model: str = "qwen3.5:9b", timeout: int = 40):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.timeout = timeout
        logger.info(f"Initialized OllamaVision: {self.base_url} (model: {model})")
    
    def extract_text_from_image(self, image_bytes: bytes, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        """
        Call vision model to extract text from image.
        
        Args:
            image_bytes: Raw image bytes (PNG, JPG, etc.)
            prompt: OCR instruction in Vietnamese/English
        
        Returns:
            Extracted text
        
        Raises:
            TimeoutError if vision call exceeds timeout
            RuntimeError on other errors
        """
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Call vision endpoint
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 512,
                    },
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            extracted_text = data.get("response", "").strip()
            logger.info(f"OCR extracted {len(extracted_text)} chars from image")
            return extracted_text
        
        except requests.Timeout:
            logger.warning(f"Vision model timeout (>{self.timeout}s); skipping OCR")
            raise TimeoutError("Vision model timeout")
        except Exception as e:
            logger.error(f"Vision model failed: {e}")
            raise RuntimeError(f"Vision error: {e}")


class OllamaChat:
    """Wrapper for Ollama chat/text generation model."""

    def __init__(self, base_url: str = None, model: str = "qwen3.5:9b", timeout: int = 40):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.timeout = timeout
        logger.info(f"Initialized OllamaChat: {self.base_url} (model: {model})")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text response from a chat model.

        Args:
            prompt: User prompt for the model
            system_prompt: Optional system instruction

        Returns:
            Model response text

        Raises:
            RuntimeError on request/model errors
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "num_predict": 220,
                        "temperature": 0.2,
                    },
                },
                timeout=self.timeout
            )
            if response.status_code == 404:
                full_prompt = prompt if not system_prompt else f"{system_prompt}\n\nUser: {prompt}"
                fallback = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "think": False,
                        "options": {
                            "num_predict": 220,
                            "temperature": 0.2,
                        },
                    },
                    timeout=self.timeout,
                )
                fallback.raise_for_status()
                data = fallback.json()
                content = data.get("response", "").strip()
            else:
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
            if not content:
                raise RuntimeError("Empty response from Ollama chat model")

            return content
        except requests.Timeout:
            logger.error(f"Chat model timeout (>{self.timeout}s)")
            raise RuntimeError("Chat model timeout")
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise RuntimeError(f"Chat generation error: {e}")
