"""Tests for evaluation metrics."""

import pytest

from ai_act_rag.evaluation.metrics import (
    _numbers,
    citation_scores,
    faithfulness,
    hit_at_k,
    key_facts_recall,
    numbers_grounded,
    polarity,
    reciprocal_rank,
    reference_facts,
    split_sentences,
    yes_no_correct,
)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["A", "B", "C"], ["C"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["A"], ["Z"]) == 0.0


def test_citation_scores() -> None:
    s = citation_scores(["Article 5", "Article 6"], ["Article 5"])
    assert s == {"correct": 1.0, "precision": 0.5, "recall": 1.0}
    assert citation_scores([], ["Article 5"])["correct"] == 0.0


def test_split_sentences() -> None:
    assert split_sentences("One. Two? Three") == ["One.", "Two?", "Three"]


def test_faithfulness_supported_vs_unsupported() -> None:
    evidence = "Providers shall keep the logs for at least six months."
    assert faithfulness("Providers must keep logs for at least six months.", evidence) == 1.0
    assert faithfulness("The penalty is imprisonment for directors.", evidence) == 0.0
    assert faithfulness("Logs kept six months. Directors go to prison.", evidence) == 0.5


def test_numbers_grounded_normalises_separators() -> None:
    evidence = "fines of up to EUR 35 000 000 or 7 % of turnover"
    assert numbers_grounded("Up to EUR 35 000 000 or 7 %.", evidence) == 1.0
    assert numbers_grounded("Up to EUR 20 000 000.", evidence) == 0.0
    assert numbers_grounded("No numbers here.", evidence) == 1.0


def test_hit_at_k_counts_any_acceptable_unit() -> None:
    retrieved = ["Recital 179", "Article 5"]
    assert hit_at_k(retrieved, ["Article 113", "Recital 179"], 1) == 1.0
    assert hit_at_k(retrieved, ["Article 99"], 2) == 0.0


def test_numbers_keep_lists_apart_and_join_thousands() -> None:
    assert _numbers("Articles 5, 6 and 7") == {"5", "6", "7"}
    assert _numbers("EUR 35 000 000 or 7 %") == {"35000000", "7"}
    assert _numbers("1,000 or 1.000.000 or 2.5") == {"1000", "1000000", "25"}
    assert numbers_grounded("Articles 5, 6.", "See Article 5 and Article 6.") == 1.0


def test_polarity_and_yes_no_correct() -> None:
    assert polarity("No, it is prohibited (Art. 5(1)(e)).") == "no"
    assert polarity("Sí, salvo excepciones.") == "yes"
    assert polarity("Fines of up to EUR 35 000 000.") is None
    ref = "No. Such systems are prohibited (Art. 5(1)(e))."
    assert yes_no_correct("Yes, an AI system can be used for that.", ref) == 0.0
    assert yes_no_correct("No, it is prohibited.", ref) == 1.0
    assert yes_no_correct("Anything", "EUR 35 000 000 (Art. 99(3)).") is None


def test_key_facts_ignore_citation_parentheses() -> None:
    ref = "Up to EUR 35 000 000 or 7 % of turnover (Art. 99(3))."
    assert reference_facts(ref) == {"35000000", "7"}
    assert key_facts_recall("EUR 35.000.000 o el 7 %", ref) == 1.0
    assert key_facts_recall("EUR 750 000", ref) == 0.0
    assert key_facts_recall("whatever", "No numbers here (Art. 5).") is None
