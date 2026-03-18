"""Targon decentralized compute client.

Targon (targon.com) is a decentralized AI inference platform on Bittensor SN4.
It provides:
- OpenAI-compatible LLM inference endpoints
- CPU compute rentals (via Targon VM / serverless SDK)
- Confidential compute with NVIDIA PPCIE / Intel TDX

We use Targon for:
1. LLM inference (agent reasoning, code generation) via OpenAI-compat API
2. CPU-bound model training tasks (sklearn, lightgbm, feature engineering)
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TARGON_URL = "https://api.targon.com/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"


class TargonClient:
    """Client for Targon decentralized inference and compute.

    Uses OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "targon_api_key", "")
        self.base_url = (base_url or getattr(settings, "targon_base_url", DEFAULT_TARGON_URL)).rstrip("/")
        self.model = model or getattr(settings, "targon_model", DEFAULT_MODEL)

        self._client = httpx.Client(
            timeout=120.0,
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
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """Call Targon's OpenAI-compatible chat completion endpoint.

        Args:
            messages: list of {role, content} message dicts
            model: override model name
            temperature: sampling temperature
            max_tokens: max response tokens
            response_format: e.g. {"type": "json_object"} for JSON mode

        Returns:
            Full API response dict
        """
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
        system: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> str:
        """Simple text generation helper."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(messages, **kwargs)
        return result["choices"][0]["message"]["content"]

    def generate_json(
        self,
        prompt: str,
        system: str = "You are a helpful AI assistant. Respond only with valid JSON.",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate structured JSON output."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        result = self.chat_completion(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        text = result["choices"][0]["message"]["content"]
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return json.loads(text.strip())

    def health_check(self) -> bool:
        """Check if Targon endpoint is reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/models")
            return resp.status_code == 200
        except Exception as e:
            logger.warning("targon_health_check_failed", error=str(e))
            return False
