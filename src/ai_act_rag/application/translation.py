"""Cross-lingual retrieval: translate non-English queries before searching.

The corpus is the English text and the default embedder is English-only, so a
Spanish question retrieves poorly. ``TranslatingRetriever`` is a decorator over
any :class:`Retriever`: it asks the LLM for an English version of the query,
searches with it, and leaves everything else (the prompt, the answer language)
unchanged.
"""

from __future__ import annotations

from ai_act_rag.application.answering import detect_language
from ai_act_rag.application.prompts import PromptTemplate
from ai_act_rag.domain.models import RetrievedChunk
from ai_act_rag.domain.ports import LLM, Retriever


class TranslatingRetriever:
    """Retriever decorator that translates Spanish queries to English first."""

    def __init__(self, inner: Retriever, llm: LLM, prompt: PromptTemplate) -> None:
        """Wrap a retriever.

        Args:
            inner: Retriever that expects English queries.
            llm: LLM used for translation.
            prompt: Translation prompt (``prompts/translate_v1.txt``).
        """
        self.inner = inner
        self.llm = llm
        self.prompt = prompt

    def translate(self, query: str) -> str:
        """Return an English version of the query (unchanged if already English).

        Args:
            query: User query.

        Returns:
            English query.
        """
        if detect_language(query) == "en":
            return query
        out = self.llm.generate(
            system=self.prompt.system,
            user=self.prompt.render_user(query, context=""),
            max_tokens=96,
        )
        # Keep the first non-empty line; small models sometimes add notes after it.
        lines = [line.strip().strip('"') for line in out.splitlines() if line.strip()]
        return lines[0] if lines else query

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Translate if needed, then delegate to the wrapped retriever.

        Args:
            query: User query (any supported language).
            k: Number of results.

        Returns:
            Ranked chunks.
        """
        return self.inner.search(self.translate(query), k)
