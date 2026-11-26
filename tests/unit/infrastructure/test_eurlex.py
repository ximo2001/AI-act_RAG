"""Tests for the OJ/EUR-Lex XHTML parser."""

from pathlib import Path

import pytest

from ai_act_rag.domain.exceptions import ParsingError
from ai_act_rag.domain.models import LegalUnit, UnitType
from ai_act_rag.infrastructure.eurlex import parse_ai_act

FIXTURES = Path(__file__).parents[2] / "fixtures"


def _by_id(units: list[LegalUnit]) -> dict[str, LegalUnit]:
    return {u.unit_id: u for u in units}


def test_parses_all_unit_types_in_order(mini_units: list[LegalUnit]) -> None:
    assert [u.unit_id for u in mini_units] == [
        "Recital 1",
        "Recital 2",
        "Article 1",
        "Article 6",
        "Annex III",
    ]


def test_article_metadata(mini_units: list[LegalUnit]) -> None:
    art1 = _by_id(mini_units)["Article 1"]
    assert art1.unit_type == UnitType.ARTICLE
    assert art1.title == "Subject matter"  # stray backtick from the OJ removed
    assert art1.chapter == "Chapter I - General provisions"
    assert art1.section is None
    art6 = _by_id(mini_units)["Article 6"]
    assert art6.chapter == "Chapter III - High-risk AI systems"
    assert art6.section == "Section 1 - Classification of AI systems as high-risk"


def test_list_points_keep_their_labels(mini_units: list[LegalUnit]) -> None:
    art1 = _by_id(mini_units)["Article 1"]
    assert art1.paragraphs[1].splitlines() == [
        "2. This Regulation lays down:",
        "(a) harmonised rules;",
        "(b) prohibitions of certain AI practices.",
    ]


def test_recital_strips_number_and_footnote_marker(mini_units: list[LegalUnit]) -> None:
    rec = _by_id(mini_units)["Recital 1"]
    assert (
        rec.text
        == "The purpose of this Regulation is to improve the internal market, in particular."
    )


def test_annex_title_and_nested_points(mini_units: list[LegalUnit]) -> None:
    annex = _by_id(mini_units)["Annex III"]
    assert annex.title == "High-risk AI systems referred to in Article 6(2)"
    assert "1. Biometrics:\n(a) remote biometric identification systems." in annex.text


def test_structure_check_rejects_unexpected_documents() -> None:
    with pytest.raises(ParsingError):
        parse_ai_act((FIXTURES / "mini_act.xhtml").read_bytes())  # default expects 113/180/13
