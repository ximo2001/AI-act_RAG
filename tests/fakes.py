"""Test doubles: a deterministic embedder and a scripted LLM (no model downloads)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import numpy.typing as npt

from ai_act_rag.application.bm25 import tokenize


class HashingEmbedder:
    """Deterministic bag-of-words embedder (feature hashing), for tests only."""

    def __init__(self, dim: int = 256) -> None:
        """Set the vector size.

        Args:
            dim: Embedding dimension.
        """
        self.dim = dim

    @property
    def name(self) -> str:
        """Model id.

        Returns:
            A fixed fake name.
        """
        return "fake-hashing"

    def _vec(self, text: str) -> npt.NDArray[np.float32]:
        """Hash tokens into a normalised vector.

        Args:
            text: Input text.

        Returns:
            Vector (dim,).
        """
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed passages.

        Args:
            texts: Passages.

        Returns:
            Array (n, dim).
        """
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a query.

        Args:
            text: Query.

        Returns:
            Vector (dim,).
        """
        return self._vec(text)


class FakeLLM:
    """Scripted LLM: returns queued responses and records the prompts it saw."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Queue responses.

        Args:
            responses: Raw strings returned in order; when exhausted, a default
                JSON citing the first unit id present in the prompt is returned.
        """
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Model id.

        Returns:
            ``"fake"``.
        """
        return "fake"

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Return the next scripted response.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: Ignored.
            max_tokens: Ignored.

        Returns:
            Scripted raw output.
        """
        self.calls.append({"system": system, "user": user, "schema": json_schema})
        if self.responses:
            return self.responses.pop(0)
        return json.dumps(
            {"answer": "Fake answer.", "cited_articles": ["Article 1"], "confidence": "high"}
        )
