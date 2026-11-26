"""Ports (interfaces) the application layer depends on.

They are ``typing.Protocol`` classes: any object with the right methods fits,
so infrastructure adapters and test fakes need no inheritance.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from ai_act_rag.domain.models import RetrievedChunk


class Embedder(Protocol):
    """Turns texts into L2-normalised dense vectors."""

    @property
    def name(self) -> str:
        """Model identifier, stored with the index to detect mismatches."""
        ...

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed passages to index.

        Args:
            texts: Passages.

        Returns:
            Array of shape (len(texts), dim), L2-normalised.
        """
        ...

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a search query (models like e5/bge use a query prefix).

        Args:
            text: The query.

        Returns:
            Array of shape (dim,), L2-normalised.
        """
        ...


class Retriever(Protocol):
    """Anything that returns ranked chunks for a query."""

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Return the top-k chunks for a query.

        Args:
            query: Natural-language query.
            k: Number of results.

        Returns:
            Chunks sorted by decreasing relevance, ranks starting at 1.
        """
        ...


class Reranker(Protocol):
    """Re-scores (query, chunk) pairs, typically with a cross-encoder."""

    def rerank(self, query: str, hits: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Reorder hits by relevance to the query.

        Args:
            query: The query.
            hits: Candidate chunks.
            k: Number of results to keep.

        Returns:
            The top-k hits, re-ranked.
        """
        ...


class LLM(Protocol):
    """A chat LLM that can be asked for JSON matching a schema."""

    @property
    def name(self) -> str:
        """Provider/model identifier for logs and eval reports."""
        ...

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Generate a completion.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: If given, providers that support it constrain the
                output to this JSON schema (llama.cpp grammar, Ollama ``format``).
            max_tokens: Generation budget.

        Returns:
            Raw generated text (expected to be JSON when a schema is given).
        """
        ...
