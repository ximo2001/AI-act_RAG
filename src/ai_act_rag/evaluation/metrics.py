"""Evaluation metrics: retrieval, citations, abstention and faithfulness.

Everything here is a pure function so each metric is unit-tested and easy
to explain in a code review.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ai_act_rag.application.bm25 import tokenize


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1 if any gold unit is in the top-k retrieved units, else 0.

    ``gold`` lists every unit that fully answers the question (e.g. an article
    and the recital restating it), so finding any one of them is a success.
    Averaged over questions this is Hit@k (a.k.a. success@k).

    Args:
        retrieved: Retrieved unit ids, ranked, without duplicates.
        gold: Acceptable unit ids.
        k: Cut-off.

    Returns:
        1.0 or 0.0.
    """
    return float(bool(set(retrieved[:k]) & set(gold)))


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """1 / rank of the first relevant unit (0 if none is retrieved).

    Averaged over questions this is the MRR.

    Args:
        retrieved: Retrieved unit ids, ranked.
        gold: Relevant unit ids.

    Returns:
        Reciprocal rank in [0, 1].
    """
    gold_set = set(gold)
    for i, unit in enumerate(retrieved, start=1):
        if unit in gold_set:
            return 1.0 / i
    return 0.0


def citation_scores(cited: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    """Compare the units cited in an answer with the gold units.

    Args:
        cited: Unit ids cited by the system.
        gold: Gold unit ids.

    Returns:
        ``correct`` (1 if at least one gold unit is cited), ``precision`` and
        ``recall`` of the cited set.
    """
    c, g = set(cited), set(gold)
    hit = len(c & g)
    return {
        "correct": float(hit > 0),
        "precision": hit / len(c) if c else 0.0,
        "recall": hit / len(g) if g else 0.0,
    }


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
# A number is either digit groups joined by single thousands separators
# ("35 000 000", "1,000", "1.000.000") or a plain integer/decimal ("7", "2.5").
# A list such as "Articles 5, 6" stays two numbers: the separator must be
# followed by exactly three digits to join groups.
_NUMBER = re.compile(r"\d{1,3}(?:[ .,]\d{3})+|\d+(?:[.,]\d+)?")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences (good enough for short answers).

    Args:
        text: Answer text.

    Returns:
        Non-empty sentences.
    """
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def lexical_support(sentence: str, evidence: str) -> float:
    """Share of the sentence's content words that also appear in the evidence.

    Args:
        sentence: One answer sentence.
        evidence: Concatenated text of the cited (or retrieved) chunks.

    Returns:
        Support in [0, 1]; 1.0 for sentences without content words.
    """
    words = set(tokenize(sentence))
    if not words:
        return 1.0
    return len(words & set(tokenize(evidence))) / len(words)


def faithfulness(answer: str, evidence: str, threshold: float = 0.6) -> float:
    """Fraction of answer sentences supported by the evidence.

    A sentence counts as supported when at least ``threshold`` of its content
    words occur in the evidence. It is a deliberately simple, explainable
    proxy (no LLM judge): it catches answers drifting away from the cited text,
    but it cannot detect a contradiction that reuses the same words, and it only
    makes sense when answer and evidence share a language (English here).

    Args:
        answer: Answer text.
        evidence: Text of the passages the answer relies on.
        threshold: Minimum lexical support per sentence.

    Returns:
        Faithfulness in [0, 1].
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    return sum(lexical_support(s, evidence) >= threshold for s in sentences) / len(sentences)


def _numbers(text: str) -> set[str]:
    """Extract numbers, normalised by removing spaces/separators (``35 000 000`` -> ``35000000``).

    Args:
        text: Any text.

    Returns:
        Set of digit strings.
    """
    return {re.sub(r"\D", "", m) for m in _NUMBER.findall(text)}


def numbers_grounded(answer: str, evidence: str) -> float:
    """Fraction of numbers in the answer that also appear in the evidence.

    Numbers (amounts, percentages, days, dates) are where a legal answer is
    most dangerous to hallucinate, and the check works across languages.

    Args:
        answer: Answer text.
        evidence: Context text.

    Returns:
        Share in [0, 1]; 1.0 if the answer contains no numbers.
    """
    nums = _numbers(answer)
    if not nums:
        return 1.0
    return len(nums & _numbers(evidence)) / len(nums)


_YES = {"yes", "sí", "si"}
_NO = {"no"}
# Parenthesised references such as "(Art. 99(3))" or "(anexo III, punto 4)" in the
# reference answers: their numbers are citations, not facts to reproduce.
_CITATION_PAREN = re.compile(
    r"\((?:[^()]|\([^()]*\))*(?:art|annex|anexo|recital|considerando)(?:[^()]|\([^()]*\))*\)",
    re.IGNORECASE,
)


def polarity(text: str) -> str | None:
    """Return "yes"/"no" if the text starts with a yes/no word, else None.

    Args:
        text: Answer or reference answer.

    Returns:
        ``"yes"``, ``"no"`` or None.
    """
    words = re.findall(r"[a-záéíóúñ]+", text.lower()[:20])
    if not words:
        return None
    if words[0] in _YES:
        return "yes"
    if words[0] in _NO:
        return "no"
    return None


def yes_no_correct(answer: str, reference: str) -> float | None:
    """Check that a yes/no answer has the same polarity as the reference answer.

    Catches the most damaging error of a small model on legal questions:
    citing the right article but saying "Yes" to something it prohibits.

    Args:
        answer: System answer.
        reference: Reference answer written from the text.

    Returns:
        1.0 / 0.0, or None when the reference is not a yes/no answer.
    """
    expected = polarity(reference)
    if expected is None:
        return None
    return float(polarity(answer) == expected)


def reference_facts(reference: str) -> set[str]:
    """Numbers stated in a reference answer, ignoring parenthesised citations.

    Args:
        reference: Reference answer.

    Returns:
        Normalised digit strings (amounts, percentages, dates, days...).
    """
    return _numbers(_CITATION_PAREN.sub(" ", reference))


def key_facts_recall(answer: str, reference: str) -> float | None:
    """Share of the reference's numeric facts that the answer reproduces.

    Args:
        answer: System answer.
        reference: Reference answer.

    Returns:
        Recall in [0, 1], or None when the reference states no numbers.
    """
    facts = reference_facts(reference)
    if not facts:
        return None
    return len(facts & _numbers(answer)) / len(facts)
