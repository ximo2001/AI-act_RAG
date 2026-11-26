"""MCP tool tests: the tool functions are called directly with a fake pipeline."""

import asyncio
import json

import pytest
from tests.fakes import FakeLLM

from ai_act_rag.application.answering import AnswerService
from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.prompts import load_prompt
from ai_act_rag.domain.models import Chunk
from ai_act_rag.infrastructure import mcp_server


@pytest.fixture(autouse=True)
def fake_service(monkeypatch: pytest.MonkeyPatch, mini_chunks: list[Chunk]) -> None:
    llm = FakeLLM(
        [json.dumps({"answer": "NOT_IN_TEXT", "cited_articles": [], "confidence": "low"})]
    )
    service = AnswerService(BM25Retriever(mini_chunks), llm, load_prompt("v2"), top_k=3)
    monkeypatch.setattr(mcp_server, "_service", lambda: service)


def test_search_tool() -> None:
    hits = mcp_server.search_ai_act("prohibitions of certain AI practices", k=1)
    assert hits[0]["unit_id"] == "Article 1"


def test_ask_tool_abstains() -> None:
    out = mcp_server.ask_ai_act("What is the corporate tax rate?")
    assert out["abstained"] is True
    assert out["cited_articles"] == []
    assert out["parse_error"] is False


def test_tools_are_registered() -> None:
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"search_ai_act", "ask_ai_act"} <= names
