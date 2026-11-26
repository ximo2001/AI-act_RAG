"""MCP server (official Python SDK) exposing ``search_ai_act`` and ``ask_ai_act``.

Run over stdio (the transport desktop MCP clients use)::

    uv run python -m ai_act_rag.infrastructure.mcp_server
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from mcp.server.mcpserver import MCPServer

from ai_act_rag.application.answering import AnswerService
from ai_act_rag.infrastructure.config import Settings
from ai_act_rag.infrastructure.container import build_answer_service

mcp = MCPServer("ai-act-rag")


@lru_cache(maxsize=1)
def _service() -> AnswerService:
    """Build the pipeline once, on first tool call.

    Returns:
        The answer service.
    """
    return build_answer_service(Settings.from_env())


@mcp.tool()
def search_ai_act(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Search the EU AI Act (Regulation 2024/1689) and return the most relevant passages.

    Args:
        query: What to look for, in English or Spanish.
        k: Number of passages (1-20).

    Returns:
        Passages with their article/annex/recital id and text.
    """
    hits = _service().retriever.search(query, k=min(max(k, 1), 20))
    return [
        {"unit_id": h.chunk.unit_id, "chunk_id": h.chunk.chunk_id, "text": h.chunk.text}
        for h in hits
    ]


@mcp.tool()
def ask_ai_act(question: str) -> dict[str, Any]:
    """Answer a question about the EU AI Act with citations to its articles.

    Args:
        question: The question, in English or Spanish.

    Returns:
        Answer, cited articles, confidence, whether the text is silent on it, and
        whether the model output could not be parsed (``parse_error``; the
        answer is then the "not stated" text but it is a failure, not a finding).
    """
    ans = _service().ask(question)
    return {
        "answer": ans.answer,
        "cited_articles": ans.cited_articles,
        "confidence": ans.confidence.value,
        "abstained": ans.abstained,
        "parse_error": ans.parse_error,
    }


def main() -> None:
    """Run the server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
