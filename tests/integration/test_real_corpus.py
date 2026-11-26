"""Checks on the real downloaded AI Act (skipped when data/raw is not present, e.g. in CI)."""

from pathlib import Path

import pytest

from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.chunking import ChunkingConfig, chunk_units
from ai_act_rag.infrastructure.eurlex import parse_ai_act

SOURCE = Path("data/raw/ai_act_32024R1689_en.xhtml")
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="AI Act not downloaded")


@pytest.fixture(scope="module")
def units():  # type: ignore[no-untyped-def]
    return {u.unit_id: u for u in parse_ai_act(SOURCE.read_bytes())}


def test_structure_and_metadata(units) -> None:  # type: ignore[no-untyped-def]
    assert units["Article 5"].title == "Prohibited AI practices"
    assert units["Article 6"].section == "Section 1 - Classification of AI systems as high-risk"
    assert units["Article 113"].chapter == "Chapter XIII - Final provisions"
    assert "10^25" in units["Article 51"].text


def test_bm25_finds_penalties_article(units) -> None:  # type: ignore[no-untyped-def]
    chunks = chunk_units(list(units.values()), ChunkingConfig(max_words=150))
    hits = BM25Retriever(chunks).search("administrative fines 35 000 000 prohibited practices", 3)
    assert hits[0].chunk.unit_id == "Article 99"
