"""Report generation from result files."""

import json
from pathlib import Path

from ai_act_rag.evaluation.report import END, START, classify_errors, retrieval_label, write_report

SUMMARY = {
    "n_questions": 3,
    "citation_correct": 0.5,
    "citation_precision": 0.75,
    "citation_recall": 0.6,
    "abstain_correct_unanswerable": 1.0,
    "false_abstention_answerable": 0.125,
    "parse_error_rate": 0.025,
    "faithfulness_en": 0.9,
    "numbers_grounded": 1.0,
    "latency_ms_mean": 12000.0,
    "latency_ms_p50": 11000.0,
}
CONFIG = {
    "llm": "llama_cpp:model.gguf",
    "prompt_version": "v2",
    "retrieval_mode": "hybrid",
    "reranker_model": "x",
    "top_k": 5,
    "chunk_words": 150,
}
ENV = {"date_utc": "2026-11-25T00:00:00", "cpu_count": 4, "python": "3.12"}


def _record(qid: str, gold: list[str]) -> dict:
    """A correct answer (or a correct abstention when there is no gold unit)."""
    return {
        "id": qid,
        "answerable": bool(gold),
        "gold": gold,
        "retrieved_units": gold,
        "cited": gold,
        "abstained": not gold,
    }


RECORDS = [_record("a", ["Article 5"]), _record("b", ["Article 6"]), _record("c", [])]
RETRIEVAL_SUMMARY = {
    "hit@1": 0.5,
    "hit@3": 0.6,
    "hit@5": 0.7,
    "hit@10": 0.8,
    "mrr@10": 0.55,
    "gold_in_context@5": 0.65,
    "context_k": 5,
    "latency_ms_mean": 10.0,
    "n_questions": 7,
}


def test_report_regenerates_readme_block(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "answers_v2.json").write_text(
        json.dumps({"env": ENV, "config": CONFIG, "summary": SUMMARY, "records": RECORDS})
    )
    (results / "retrieval.json").write_text(
        json.dumps(
            {
                "env": ENV,
                "results": [
                    {
                        "embedding_model": "bge",
                        "chunk_words": 150,
                        "mode": "hybrid",
                        "summary": RETRIEVAL_SUMMARY,
                    }
                ],
            }
        )
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold numbers\n{END}\noutro\n")
    md = write_report(results, readme)
    text = readme.read_text()
    assert "old numbers" not in text
    assert (
        "| model.gguf | v2 | hybrid+rerank | 50.0% | 75.0% | 60.0% | 100.0% | 12.5% | 2.5% |" in md
    )
    # Counts come from the result files, not from hand-typed text.
    assert "(2 answerable questions, 1 not answerable from the text)" in md
    assert "### Retrieval (7 answerable questions" in md
    assert "Gold in LLM context (top-5 chunks)" in md and "| 65.0% |" in md
    assert "_llama_cpp, temperature 0, top-k = 5 chunks of 150 words" in md
    assert text.startswith("intro\n") and text.endswith("outro\n")
    assert (results / "summary.md").exists()


def test_rerank_label_only_for_hybrid() -> None:
    assert retrieval_label({"retrieval_mode": "dense", "reranker_model": "x"}) == "dense"
    assert retrieval_label({"retrieval_mode": "hybrid", "reranker_model": ""}) == "hybrid"
    assert (
        retrieval_label(
            {"retrieval_mode": "hybrid", "reranker_model": "x", "translate_queries": True}
        )
        == "hybrid+rerank+translate"
    )


def test_classify_errors_attributes_failures_to_stages() -> None:
    def rec(qid: str, gold: list[str], retrieved: list[str], cited: list[str], abst: bool) -> dict:
        return {
            "id": qid,
            "gold": gold,
            "answerable": bool(gold),
            "retrieved_units": retrieved,
            "cited": cited,
            "abstained": abst,
        }

    errs = classify_errors(
        [
            rec("ok", ["Article 5"], ["Article 5"], ["Article 5"], False),
            rec("ret", ["Article 5"], ["Article 6"], ["Article 6"], False),
            rec("gen", ["Article 5"], ["Article 5", "Article 6"], ["Article 6"], False),
            rec("abs", ["Article 5"], ["Article 5"], [], True),
            rec("hal", [], ["Article 99"], ["Article 99"], False),
            rec("good_abs", [], ["Article 99"], [], True),
            {**rec("parse", [], ["Article 99"], [], True), "parse_error": True},
        ]
    )
    assert errs == {
        "retrieval_miss": ["ret"],
        "generation_miss": ["gen"],
        "false_abstention": ["abs"],
        "hallucination": ["hal"],
        "parse_error": ["parse"],
    }
