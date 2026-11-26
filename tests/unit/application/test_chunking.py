"""Tests for structure-aware chunking."""

from ai_act_rag.application.chunking import (
    ChunkingConfig,
    chunk_unit,
    chunk_units,
    make_header,
)
from ai_act_rag.domain.models import LegalUnit, UnitType


def _article(paragraphs: list[str]) -> LegalUnit:
    return LegalUnit(
        unit_id="Article 9",
        unit_type=UnitType.ARTICLE,
        number="9",
        title="Risk management system",
        chapter="Chapter III - High-risk AI systems",
        paragraphs=paragraphs,
    )


def test_chunks_never_cross_units(mini_units: list[LegalUnit]) -> None:
    chunks = chunk_units(mini_units, ChunkingConfig(max_words=5))
    unit_ids = {u.unit_id for u in mini_units}
    assert {c.unit_id for c in chunks} == unit_ids
    for c in chunks:
        assert c.chunk_id.startswith(c.unit_id + "#")


def test_small_unit_is_one_chunk_with_header() -> None:
    chunks = chunk_unit(_article(["1. Short paragraph."]), ChunkingConfig(max_words=50))
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "Article 9#1"
    assert chunks[0].text.startswith(
        "Article 9 - Risk management system | Chapter III - High-risk AI systems\n"
    )


def test_paragraphs_are_packed_up_to_budget() -> None:
    paras = ["one two three", "four five six", "seven eight nine"]
    chunks = chunk_unit(_article(paras), ChunkingConfig(max_words=6))
    assert [c.body for c in chunks] == ["one two three\n\nfour five six", "seven eight nine"]


def test_long_paragraph_split_on_list_points() -> None:
    para = "1. Intro words here:\n(a) alpha beta gamma;\n(b) delta epsilon zeta."
    chunks = chunk_unit(_article([para]), ChunkingConfig(max_words=5))
    bodies = [c.body for c in chunks]
    assert bodies == ["1. Intro words here:", "(a) alpha beta gamma;", "(b) delta epsilon zeta."]


def test_recitals_can_be_excluded(mini_units: list[LegalUnit]) -> None:
    chunks = chunk_units(mini_units, ChunkingConfig(include_recitals=False))
    assert all(c.unit_type != UnitType.RECITAL for c in chunks)


def test_header_for_recital_has_no_title() -> None:
    unit = LegalUnit(unit_id="Recital 3", unit_type=UnitType.RECITAL, number="3", paragraphs=["x"])
    assert make_header(unit) == "Recital 3"
