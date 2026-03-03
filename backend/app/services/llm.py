# backend/app/services/llm.py
# -------------------------------------------------
# Tiny wrapper around Ollama's HTTP `/api/generate` endpoint.
# No LangChain dependency is needed for the MVP.
# -------------------------------------------------
import os
import httpx
from typing import Optional

class LLMService:
    """
    Simple client for an Ollama model running locally.
    The constructor reads the URL and model name from environment
    variables (same names used throughout the project):
        OLLAMA_URL   – e.g. http://localhost:11434
        OLLAMA_MODEL – e.g. phi-2
    """
    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Allow explicit args or fall back to env variables
        self.base_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "phi-2")
        # Normalise URL (remove trailing slash)
        self.base_url = self.base_url.rstrip("/")

        # Re‑use a single httpx client – keeps connections alive
        self.client = httpx.Client(timeout=60)

    def chat(self, prompt: str) -> str:
        """
        Sends *prompt* to Ollama and returns the generated completion text.
        Returns an empty string on any error (the caller can handle it).
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,   # we want the whole answer at once
        }

        try:
            resp = self.client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            # Ollama returns JSON { "response": "...", "model":"...", ... }
            return resp.json().get("response", "")
        except Exception as exc:
            # In production you’d log the exception; for the MVP we just
            # return an empty answer so the API can send a clean JSON error.
            print(f"[LLMService] error: {exc}")
            return ""
