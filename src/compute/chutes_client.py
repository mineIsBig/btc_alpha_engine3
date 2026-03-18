"""Chutes.ai serverless AI inference client.

Chutes (chutes.ai) is a serverless GPU compute platform supporting:
- OpenAI-compatible LLM inference (via vLLM template)
- Custom model deployments
- Batch processing
- TEE (Trusted Execution Environments)

We use Chutes for:
1. Autonomous agent inference (high-throughput LLM calls for reasoning loop)
2. Foundation model inference for time-series (when available)
3. Embeddings for regime similarity search
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)


class ChutesClient:
    """Client for Chutes.ai serverless inference.

    Supports OpenAI-compatible chat completions via deployed chutes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "chutes_api_key", "")
        self.base_url = (base_url or getattr(settings, "chutes_base_url", "https://chutes.ai/api/v1")).rstrip("/")
        self.model = model or getattr(settings, "chutes_model", "deepseek-ai/DeepSeek-V3-0324")

        self._client = httpx.Client(
            timeout=180.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

    def close(self) -> None:
        self._client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 8192,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """Call Chutes OpenAI-compatible chat completion."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    def generate(
        self,
        prompt: str,
        system: str = "You are an expert quantitative researcher and trading systems architect.",
        **kwargs: Any,
    ) -> str:
        """Simple text generation."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(messages, **kwargs)
        return result["choices"][0]["message"]["content"]

    def generate_json(
        self,
        prompt: str,
        system: str = "You are an expert quantitative researcher. Respond only with valid JSON, no markdown.",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate structured JSON output."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(messages, **kwargs)
        text = result["choices"][0]["message"]["content"]
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())

    def call_custom_chute(
        self,
        chute_url: str,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a custom deployed chute endpoint.

        For custom model deployments (e.g., time-series foundation models).
        """
        url = f"{chute_url.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        """Check Chutes endpoint health."""
        try:
            resp = self._client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except Exception as e:
            logger.warning("chutes_health_check_failed", error=str(e))
            return False
