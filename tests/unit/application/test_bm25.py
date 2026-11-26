"""Tests for the from-scratch BM25 retriever."""

import math

from ai_act_rag.application.bm25 import BM25Retriever, tokenize
from ai_act_rag.domain.models import Chunk, UnitType


def _chunk(cid: str, body: str) -> Chunk:
    return Chunk(
        chunk_id=cid, unit_id=cid.split("#")[0], unit_type=UnitType.ARTICLE, header="", body=body
    )


def test_tokenize_lowercases_drops_stopwords_and_plurals() -> None:
    assert tokenize("The Prohibited AI practices of systems") == [
        "prohibited",
        "ai",
        "practice",
        "system",
    ]


def test_exact_term_match_ranks_first() -> None:
    chunks = [
        _chunk("Article 1#1", "general provisions subject matter"),
        _chunk("Article 5#1", "prohibited practices subliminal techniques"),
        _chunk("Article 9#1", "risk management system"),
    ]
    hits = BM25Retriever(chunks).search("subliminal techniques", k=3)
    assert hits[0].chunk.chunk_id == "Article 5#1"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.score > 0 for h in hits)


def test_idf_formula() -> None:
    chunks = [_chunk("Article 1#1", "alpha beta"), _chunk("Article 2#1", "alpha gamma")]
    bm = BM25Retriever(chunks, k1=1.5, b=0.0)
    # "beta" appears in 1 of 2 docs: idf = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln 2
    expected = math.log(2) * 1 * 2.5 / (1 + 1.5)
    assert math.isclose(bm.scores("beta")[0], expected)
    assert bm.scores("beta")[1] == 0.0


def test_unknown_query_returns_nothing() -> None:
    assert BM25Retriever([_chunk("Article 1#1", "alpha")]).search("zzz") == []
