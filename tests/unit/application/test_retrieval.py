"""Tests for dense retrieval, RRF and hybrid retrieval."""

from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.retrieval import (
    DenseRetriever,
    HybridRetriever,
    NumpyIndex,
    reciprocal_rank_fusion,
    unique_units,
)
from ai_act_rag.domain.models import Chunk, RetrievedChunk, UnitType


def _hit(cid: str, rank: int) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=cid, unit_id=cid.split("#")[0], unit_type=UnitType.ARTICLE, header="", body=cid
    )
    return RetrievedChunk(chunk=chunk, score=1.0, rank=rank)


def test_rrf_rewards_documents_ranked_well_by_both() -> None:
    a = [_hit("A#1", 1), _hit("B#1", 2), _hit("C#1", 3)]
    b = [_hit("B#1", 1), _hit("C#1", 2), _hit("A#1", 3)]
    fused = reciprocal_rank_fusion([a, b], k=3, rrf_k=60)
    assert fused[0].chunk.chunk_id == "B#1"  # ranks 2 and 1
    assert fused[0].score == 1 / 62 + 1 / 61
    assert [h.rank for h in fused] == [1, 2, 3]


def test_rrf_includes_documents_seen_by_one_retriever_only() -> None:
    fused = reciprocal_rank_fusion([[_hit("A#1", 1)], [_hit("Z#1", 1)]], k=5)
    assert {h.chunk.chunk_id for h in fused} == {"A#1", "Z#1"}


def test_unique_units_keeps_first_occurrence() -> None:
    hits = [_hit("Article 5#2", 1), _hit("Article 5#1", 2), _hit("Annex III#1", 3)]
    assert unique_units(hits) == ["Article 5", "Annex III"]


def test_dense_and_hybrid_retrieve_relevant_unit(mini_chunks, embedder) -> None:  # type: ignore[no-untyped-def]
    index = NumpyIndex(embedder.embed_documents([c.text for c in mini_chunks]))
    dense = DenseRetriever(mini_chunks, embedder, index)
    hybrid = HybridRetriever([dense, BM25Retriever(mini_chunks)], candidates=10)
    query = "remote biometric identification systems"
    assert dense.search(query, k=1)[0].chunk.unit_id == "Annex III"
    assert hybrid.search(query, k=1)[0].chunk.unit_id == "Annex III"


class _ReverseReranker:
    def rerank(self, query: str, hits: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        return [h.model_copy(update={"rank": i}) for i, h in enumerate(reversed(hits[:k]), 1)]


def test_hybrid_applies_reranker(mini_chunks, embedder) -> None:  # type: ignore[no-untyped-def]
    index = NumpyIndex(embedder.embed_documents([c.text for c in mini_chunks]))
    dense = DenseRetriever(mini_chunks, embedder, index)
    plain = HybridRetriever([dense], candidates=3).search("high-risk", k=3)
    reranked = HybridRetriever([dense], candidates=3, reranker=_ReverseReranker()).search(
        "high-risk", k=3
    )
    assert [h.chunk.chunk_id for h in reranked] == [h.chunk.chunk_id for h in reversed(plain)]
