"""Ollama provider (handy on a laptop: ``ollama pull qwen2.5:3b``)."""

from __future__ import annotations

from typing import Any

import httpx

from ai_act_rag.domain.exceptions import ProviderUnavailableError


class OllamaLLM:
    """Calls a local Ollama server through its REST API (``/api/chat``).

    Ollama's ``format`` field accepts a JSON schema and enforces it with a
    grammar, like llama.cpp.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Configure the client.

        Args:
            model: Ollama model tag, e.g. ``"qwen2.5:3b"``.
            base_url: Ollama server URL.
            temperature: Sampling temperature.
            seed: Sampling seed.
            timeout: HTTP timeout in seconds.
            client: Optional injected HTTP client (tests).
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._options = {"temperature": temperature, "seed": seed}
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
            ``"ollama:<model>"``.
        """
        return f"ollama:{self._model}"

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Generate a chat completion.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: Optional JSON schema passed as Ollama ``format``.
            max_tokens: Generation budget (``num_predict``).

        Returns:
            Generated text.

        Raises:
            ProviderUnavailableError: If the server cannot be reached.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {**self._options, "num_predict": max_tokens},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        try:
            resp = self._client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Ollama request failed: {exc}") from exc
        return str(resp.json()["message"]["content"])
