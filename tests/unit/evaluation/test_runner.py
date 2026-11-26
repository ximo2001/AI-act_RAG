"""Evaluation runner: answer scoring and retrieval metrics with fakes (no models)."""

import json
from pathlib import Path

from tests.fakes import FakeLLM

from ai_act_rag.application.answering import AnswerService
from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.prompts import PromptTemplate
from ai_act_rag.domain.models import Chunk
from ai_act_rag.evaluation.runner import (
    EvalQuestion,
    evaluate_answers,
    evaluate_retriever,
    rescore,
    summarize_answers,
)

PROMPT = PromptTemplate(version="test", system="sys", user="Context:\n{context}\nQ: {question}")


def _q(qid: str, answerable: bool, gold: list[str] | None = None) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        lang="en",
        question="Which remote biometric identification systems are high-risk?",
        gold_units=gold or [],
        answerable=answerable,
    )


def _answer(text: str, cited: list[str]) -> str:
    return json.dumps({"answer": text, "cited_articles": cited, "confidence": "high"})


def test_parse_errors_are_failures_not_correct_abstentions(mini_chunks: list[Chunk]) -> None:
    llm = FakeLLM(["garbage", _answer("NOT_IN_TEXT", [])])
    service = AnswerService(BM25Retriever(mini_chunks), llm, PROMPT, top_k=3)
    res = evaluate_answers(service, [_q("u1", False), _q("u2", False)])
    assert [r["parse_error"] for r in res["records"]] == [True, False]
    assert res["summary"]["abstain_correct_unanswerable"] == 0.5
    assert res["summary"]["parse_error_rate"] == 0.5


def test_uncited_answer_has_zero_faithfulness(mini_chunks: list[Chunk]) -> None:
    text = "Remote biometric identification systems are high-risk."
    llm = FakeLLM([_answer(text, []), _answer(text, ["Annex III"])])
    service = AnswerService(BM25Retriever(mini_chunks), llm, PROMPT, top_k=3)
    res = evaluate_answers(service, [_q("a1", True, ["Annex III"])] * 2)
    uncited, cited = res["records"]
    assert uncited["faithfulness"] == 0.0
    assert cited["faithfulness"] == 1.0


def test_gold_in_context_uses_the_first_top_k_chunks(mini_chunks: list[Chunk]) -> None:
    retriever = BM25Retriever(mini_chunks)
    q = _q("r1", True, ["Annex III"])
    first_unit = retriever.search(q.question, k=1)[0].chunk.unit_id
    assert first_unit == "Annex III"
    res = evaluate_retriever(retriever, [q], ks=(1, 5), context_k=1)
    assert res["summary"]["gold_in_context@1"] == 1.0
    assert res["summary"]["context_k"] == 1
    miss = evaluate_retriever(retriever, [_q("r2", True, ["Article 99"])], ks=(1, 5), context_k=1)
    assert miss["summary"]["gold_in_context@1"] == 0.0


def test_per_language_hit_uses_largest_cutoff_up_to_5(mini_chunks: list[Chunk]) -> None:
    res = evaluate_retriever(BM25Retriever(mini_chunks), [_q("r1", True, ["Annex III"])], ks=(1, 3))
    assert res["summary"]["hit@3_en"] == 1.0
    assert "hit@5_en" not in res["summary"]


def test_rescore_applies_new_gold_without_touching_answers(tmp_path: Path) -> None:
    lat = {"total": 10.0, "generation": 8.0, "retrieval": 2.0}
    common = {"lang": "en", "confidence": "high", "latency_ms": lat, "parse_error": False}
    records = [
        {
            **common,
            "id": "a1",
            "answerable": True,
            "abstained": False,
            "gold": ["Article 5"],
            "answer": "Prohibited.",
            "cited": ["Article 6"],
            "correct": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "faithfulness": 0.9,
            "numbers_grounded": 1.0,
        },
        {
            **common,
            "id": "u1",
            "answerable": False,
            "abstained": True,
            "gold": [],
            "answer": "Not stated.",
            "cited": [],
        },
    ]
    path = tmp_path / "answers_test_v2.json"
    path.write_text(json.dumps({"config": {"llm": "fake"}, "records": records}), "utf-8")
    questions = [
        EvalQuestion(id="a1", lang="en", question="q", gold_units=["Article 6"], answerable=True),
        EvalQuestion(id="u1", lang="en", question="q", gold_units=[], answerable=False),
    ]
    payload = rescore(path, questions)
    a1 = payload["records"][0]
    assert a1["gold"] == ["Article 6"]
    assert (a1["correct"], a1["precision"], a1["recall"]) == (1.0, 1.0, 1.0)
    assert a1["answer"] == "Prohibited." and a1["faithfulness"] == 0.9
    assert payload["summary"]["citation_correct"] == 1.0
    assert payload["summary"]["abstain_correct_unanswerable"] == 1.0
    assert json.loads(path.read_text("utf-8")) == payload


def _scored(qid: str, answer: str, reference: str, abstained: bool = False) -> dict:
    """Minimal answerable record as stored in ``answers_*.json``."""
    lat = {"total": 1.0, "generation": 1.0, "retrieval": 0.0}
    rec = {
        "id": qid,
        "lang": "en",
        "answerable": True,
        "abstained": abstained,
        "parse_error": False,
        "answer": answer,
        "reference": reference,
        "latency_ms": lat,
    }
    if not abstained:
        rec.update(correct=1.0, precision=1.0, recall=1.0, faithfulness=1.0, numbers_grounded=1.0)
    return rec


def test_correctness_metrics_skip_inapplicable_and_penalise_abstentions() -> None:
    records = [
        _scored("y1", "No, it is prohibited.", "No. Prohibited (Art. 5)."),
        _scored("y2", "Yes, it is allowed.", "No. Prohibited (Art. 5)."),
        _scored("y3", "Not stated.", "No. Prohibited (Art. 5).", abstained=True),
        _scored("f1", "Up to EUR 35 000 000 or 7 %.", "EUR 35 000 000 or 7 % (Art. 99(3))."),
        _scored("f2", "Fines apply.", "EUR 15 000 000 or 3 % (Art. 99(4))."),
        _scored("n1", "Article 50 applies.", "Transparency duties apply (Art. 50)."),
    ]
    summary = summarize_answers(records)
    # yes/no: y1 right, y2 wrong polarity, y3 abstained -> 1/3; f*/n1 not yes/no.
    assert summary["yes_no_correct"] == 1 / 3
    # key facts: f1 reproduces both numbers, f2 none; y*/n1 state no facts.
    assert summary["key_facts_recall"] == 0.5


def test_rescore_recomputes_correctness_from_new_reference(tmp_path: Path) -> None:
    record = _scored("a1", "No, it is prohibited.", "Yes, it is allowed.")
    record.update(gold=["Article 5"], cited=["Article 5"])
    path = tmp_path / "answers_test_v3.json"
    path.write_text(json.dumps({"config": {}, "records": [record]}), "utf-8")
    question = EvalQuestion(
        id="a1",
        lang="en",
        question="q",
        gold_units=["Article 5"],
        answerable=True,
        reference_answer="No. Prohibited (Art. 5).",
    )
    payload = rescore(path, [question])
    assert payload["records"][0]["reference"] == "No. Prohibited (Art. 5)."
    assert payload["summary"]["yes_no_correct"] == 1.0
