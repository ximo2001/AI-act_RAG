"""Any OpenAI-compatible chat API (Groq, Gemini's OpenAI endpoint, vLLM, ...).

Uses plain HTTP instead of the ``openai`` SDK: one POST is all we need.
The API key is read from the environment by the caller, never hardcoded.
"""

from __future__ import annotations

from typing import Any

import httpx

from ai_act_rag.domain.exceptions import ProviderUnavailableError


class OpenAICompatLLM:
    """Client for ``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Configure the client.

        Args:
            model: Model name, e.g. ``"llama-3.1-8b-instant"`` on Groq.
            base_url: API base URL, e.g. ``"https://api.groq.com/openai/v1"``.
            api_key: Bearer token (from the environment).
            temperature: Sampling temperature.
            seed: Sampling seed (ignored by some providers).
            timeout: HTTP timeout in seconds.
            client: Optional injected HTTP client (tests).
        """
        self._model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._temperature = temperature
        self._seed = seed
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client if this adapter created it (an injected one is left open)."""
        if self._owns_client:
            self._client.close()

    @property
    def name(self) -> str:
        """Provider/model id.

        Returns:
            ``"openai_compat:<model>"``.
        """
        return f"openai_compat:{self._model}"

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Generate a chat completion.

        ``json_object`` mode is used instead of strict ``json_schema`` because it
        is the lowest common denominator across free providers; the schema is
        still described in the prompt and validated with Pydantic afterwards.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: If given, request JSON output.
            max_tokens: Generation budget.

        Returns:
            Generated text.

        Raises:
            ProviderUnavailableError: On HTTP errors.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "seed": self._seed,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.post(self._url, json=payload, headers=self._headers)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"OpenAI-compatible request failed: {exc}") from exc
        return str(resp.json()["choices"][0]["message"]["content"])
