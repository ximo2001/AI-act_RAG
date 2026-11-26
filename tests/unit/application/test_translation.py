"""Tests for the query-translating retriever decorator."""

from tests.fakes import FakeLLM

from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.prompts import load_prompt
from ai_act_rag.application.translation import TranslatingRetriever
from ai_act_rag.domain.models import Chunk, RetrievedChunk


class SpyRetriever:
    """Records the queries it receives and delegates to BM25."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.inner = BM25Retriever(chunks)
        self.queries: list[str] = []

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.inner.search(query, k)


def test_spanish_query_is_translated_before_search(mini_chunks: list[Chunk]) -> None:
    llm = FakeLLM(["Which remote biometric identification systems are high-risk?\nNote: ..."])
    spy = SpyRetriever(mini_chunks)
    retriever = TranslatingRetriever(spy, llm, load_prompt("v1", kind="translate"))
    question = "¿Qué sistemas de identificación biométrica remota son de alto riesgo?"
    hits = retriever.search(question, k=1)
    assert spy.queries == ["Which remote biometric identification systems are high-risk?"]
    assert hits[0].chunk.unit_id == "Annex III"
    assert "Translate" in llm.calls[0]["system"]
    assert question in llm.calls[0]["user"]


def test_english_query_skips_the_llm(mini_chunks: list[Chunk]) -> None:
    llm = FakeLLM()
    spy = SpyRetriever(mini_chunks)
    retriever = TranslatingRetriever(spy, llm, load_prompt("v1", kind="translate"))
    retriever.search("What are the high-risk AI systems?", k=1)
    assert llm.calls == []
    assert spy.queries == ["What are the high-risk AI systems?"]
