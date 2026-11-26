"""Tests for the answer use case with a fake LLM (no model needed)."""

import json

import pytest
from tests.fakes import FakeLLM

from ai_act_rag.application.answering import (
    ABSTAIN_MESSAGES,
    AnswerService,
    detect_language,
    normalize_citation,
    parse_llm_output,
)
from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.prompts import PromptTemplate, load_prompt
from ai_act_rag.domain.exceptions import LLMOutputError
from ai_act_rag.domain.models import MAX_CITATIONS, Chunk, Confidence, LLMAnswer

PROMPT = PromptTemplate(version="test", system="sys", user="Context:\n{context}\nQ: {question}")


def _service(chunks: list[Chunk], llm: FakeLLM) -> AnswerService:
    return AnswerService(BM25Retriever(chunks), llm, PROMPT, top_k=3)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Article 5", "Article 5"),
        ("Art. 5(1)(f)", "Article 5"),
        ("artículo 99", "Article 99"),
        ("Annex III, point 4", "Annex III"),
        ("anexo iv", "Annex IV"),
        ("Recital (29)", "Recital 29"),
        ("GDPR", None),
    ],
)
def test_normalize_citation(raw: str, expected: str | None) -> None:
    assert normalize_citation(raw) == expected


def test_detect_language() -> None:
    assert detect_language("¿Qué multas se aplican según el Reglamento?") == "es"
    assert detect_language("What fines apply under the Regulation?") == "en"


def test_parse_llm_output_extracts_json_from_noise() -> None:
    raw = 'Sure! {"answer": "Yes.", "cited_articles": ["Article 5"], "confidence": "HIGH"} done'
    parsed = parse_llm_output(raw)
    assert parsed.answer == "Yes."
    assert parsed.confidence == Confidence.HIGH


def test_parse_llm_output_rejects_garbage() -> None:
    with pytest.raises(LLMOutputError):
        parse_llm_output("no json at all")


def test_answer_cites_only_retrieved_units(mini_chunks: list[Chunk]) -> None:
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "answer": "Biometric systems are high-risk.",
                    "cited_articles": ["Annex III", "Article 42"],
                    "confidence": "high",
                }
            )
        ]
    )
    answer = _service(mini_chunks, llm).ask("remote biometric identification")
    assert not answer.abstained
    assert answer.cited_articles == ["Annex III"]  # Article 42 was never retrieved
    assert set(answer.latency_ms) == {"retrieval", "generation", "total"}
    # The context sent to the LLM carries citable ids in brackets.
    assert "[Annex III]" in llm.calls[0]["user"]
    assert llm.calls[0]["schema"]["required"] == ["answer", "cited_articles", "confidence"]


def test_sentinel_triggers_localised_abstention(mini_chunks: list[Chunk]) -> None:
    llm = FakeLLM(
        [json.dumps({"answer": "NOT_IN_TEXT", "cited_articles": [], "confidence": "low"})]
    )
    answer = _service(mini_chunks, llm).ask("¿Cuál es el tipo del impuesto de sociedades?")
    assert answer.abstained
    assert answer.answer == ABSTAIN_MESSAGES["es"]
    assert answer.cited_articles == []


def test_invalid_output_falls_back_to_abstention_flagged_as_parse_error(
    mini_chunks: list[Chunk],
) -> None:
    answer = _service(mini_chunks, FakeLLM(["I think yes"])).ask("What is high-risk?")
    assert answer.abstained
    assert answer.parse_error
    assert answer.raw_output == "I think yes"


def test_deliberate_abstention_is_not_a_parse_error(mini_chunks: list[Chunk]) -> None:
    raw = json.dumps({"answer": "NOT_IN_TEXT", "cited_articles": [], "confidence": "low"})
    answer = _service(mini_chunks, FakeLLM([raw])).ask("What is the tax rate?")
    assert answer.abstained and not answer.parse_error


def test_citations_are_capped_in_schema_and_validation() -> None:
    schema = LLMAnswer.model_json_schema()
    assert schema["properties"]["cited_articles"]["maxItems"] == MAX_CITATIONS
    many = [f"Article {i}" for i in range(1, 9)]
    parsed = LLMAnswer.model_validate({"answer": "x", "cited_articles": [" Article 1", *many]})
    assert parsed.cited_articles == [f"Article {i}" for i in range(1, 6)]


@pytest.mark.parametrize(
    ("answer", "abstained"),
    [
        ("Not specified", True),
        ("No se proporcionó información sobre el salario mínimo.", True),
        ("The context does not mention the number of employees.", True),
        ("No, it is prohibited except for medical or safety reasons.", False),
        ("No, the system is prohibited.", False),
        ("Fines of up to EUR 35 000 000.", False),
        # The limitation is in a later clause/sentence, not in the answer itself.
        (
            "Providers must ensure human oversight. The exact measures are not specified "
            "by the deployer.",
            False,
        ),
        ("Providers must keep the logs; the retention period is not stated.", False),
        ("Providers must ensure that the purpose is not specified by the deployer.", False),
        ("Not specified in the text.", True),
        ("The budget of each sandbox is not specified in the given context.", True),
        ("The number of employees is not mentioned.", True),
        # A comma continues the clause with an actual answer.
        ("The deadline is not specified, but Article 73 requires 15 days.", False),
    ],
)
def test_abstention_phrasings(mini_chunks: list[Chunk], answer: str, abstained: bool) -> None:
    raw = json.dumps({"answer": answer, "cited_articles": ["Article 1"], "confidence": "low"})
    assert _service(mini_chunks, FakeLLM([raw])).ask("question?").abstained is abstained


def test_prompt_states_the_answer_language(mini_chunks: list[Chunk]) -> None:
    llm = FakeLLM()
    AnswerService(BM25Retriever(mini_chunks), llm, load_prompt("v2"), top_k=2).ask(
        "¿Qué sistemas de IA son de alto riesgo según el anexo III?"
    )
    assert 'Write the "answer" in Spanish.' in llm.calls[0]["user"]


def test_limitation_after_first_sentence_is_not_abstention(mini_chunks: list[Chunk]) -> None:
    answer = "Biometric systems are high-risk. The text does not specify their number."
    raw = json.dumps({"answer": answer, "cited_articles": ["Annex III"], "confidence": "high"})
    assert not _service(mini_chunks, FakeLLM([raw])).ask("biometric?").abstained
