"""Evaluation runners: retrieval comparison and end-to-end answer evaluation.

Outputs (JSON) go to ``eval/results/``; :mod:`ai_act_rag.evaluation.report`
turns them into the Markdown tables of the README. No number in the README is
typed by hand.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ai_act_rag.application.answering import AnswerService
from ai_act_rag.application.prompts import load_prompt
from ai_act_rag.application.retrieval import unique_units
from ai_act_rag.application.translation import TranslatingRetriever
from ai_act_rag.domain.ports import LLM, Embedder, Reranker, Retriever
from ai_act_rag.evaluation.metrics import (
    citation_scores,
    faithfulness,
    hit_at_k,
    key_facts_recall,
    numbers_grounded,
    reciprocal_rank,
    yes_no_correct,
)
from ai_act_rag.infrastructure.config import Settings
from ai_act_rag.infrastructure.container import build_index, build_retriever, set_seeds
from ai_act_rag.infrastructure.index_store import load_meta

logger = logging.getLogger(__name__)

KS = (1, 3, 5, 10)


class EvalQuestion(BaseModel):
    """One line of ``eval/questions.jsonl``."""

    id: str
    lang: str
    question: str
    gold_units: list[str]
    answerable: bool
    reference_answer: str | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    """Load the evaluation set.

    Args:
        path: JSONL file.

    Returns:
        Questions.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EvalQuestion.model_validate_json(line) for line in lines if line.strip()]


def _env_info() -> dict[str, Any]:
    """Describe the machine the eval ran on (latency depends on it).

    Returns:
        Platform info.
    """
    return {
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def evaluate_retriever(
    retriever: Retriever,
    questions: list[EvalQuestion],
    ks: tuple[int, ...] = KS,
    context_k: int = 5,
) -> dict[str, Any]:
    """Compute Hit@k and MRR at unit level over answerable questions.

    Hit@k counts the first k distinct *units*, but the LLM only sees the first
    ``context_k`` *chunks* (several of which may belong to the same unit), so
    ``gold_in_context@<context_k>`` reports whether a gold unit actually reaches
    the prompt.

    Args:
        retriever: Retriever under test.
        questions: Eval set (unanswerable questions are skipped).
        ks: Cut-offs (in units).
        context_k: Number of chunks given to the LLM (``Settings.top_k``).

    Returns:
        Aggregated metrics plus per-question unit rankings.
    """
    ctx_key = f"gold_in_context@{context_k}"
    per_q: list[dict[str, Any]] = []
    latencies: list[float] = []
    for q in questions:
        if not q.answerable:
            continue
        t0 = time.perf_counter()
        hits = retriever.search(q.question, k=max(ks) * 3)  # chunks; several per unit
        latencies.append((time.perf_counter() - t0) * 1000)
        units = unique_units(hits)
        context_units = unique_units(hits[:context_k])
        per_q.append(
            {
                "id": q.id,
                "lang": q.lang,
                "gold": q.gold_units,
                "retrieved": units[: max(ks)],
                **{f"hit@{k}": hit_at_k(units, q.gold_units, k) for k in ks},
                "rr": reciprocal_rank(units[: max(ks)], q.gold_units),
                ctx_key: hit_at_k(context_units, q.gold_units, len(context_units)),
            }
        )
    summary: dict[str, Any] = {
        f"hit@{k}": statistics.mean(p[f"hit@{k}"] for p in per_q) for k in ks
    }
    summary["mrr@10"] = statistics.mean(p["rr"] for p in per_q)
    summary[ctx_key] = statistics.mean(p[ctx_key] for p in per_q)
    summary["context_k"] = context_k
    summary["latency_ms_mean"] = statistics.mean(latencies)
    # Per-language Hit@k at the largest cut-off <= 5 (the LLM context size by default).
    lang_k = max((k for k in ks if k <= 5), default=min(ks))
    for lang in ("en", "es"):
        sub = [p for p in per_q if p["lang"] == lang]
        if sub:
            summary[f"hit@{lang_k}_{lang}"] = statistics.mean(p[f"hit@{lang_k}"] for p in sub)
    summary["n_questions"] = len(per_q)
    return {"summary": summary, "per_question": per_q}


def run_retrieval_eval(
    base: Settings,
    questions: list[EvalQuestion],
    embedders: Mapping[str, Embedder],
    chunk_sizes: list[int],
    reranker: Reranker | None,
    out_path: Path,
    translator: LLM | None = None,
) -> list[dict[str, Any]]:
    """Grid: embedding model x chunk size x retrieval mode (dense/BM25/hybrid/+rerank).

    Indexes are built once per (embedder, chunk size) under ``data/index_eval/``.

    Args:
        base: Base settings.
        questions: Eval set.
        embedders: Name -> embedder to compare.
        chunk_sizes: Max words per chunk to compare.
        reranker: Optional cross-encoder for the "hybrid+rerank" row.
        out_path: JSON output file.
        translator: Optional LLM; adds a "+translate" row where non-English
            queries are translated to English before retrieval.

    Returns:
        One result dict per configuration.
    """
    results = []
    for emb_name, embedder in embedders.items():
        for words in chunk_sizes:
            settings = replace(
                base,
                chunk_words=words,
                index_dir=Path("data/index_eval") / f"{emb_name}_w{words}",
                # The plain "hybrid" row must not pick up the default reranker from
                # settings; the reranked row passes it explicitly below.
                reranker_model="",
            )
            chunks = build_index(settings, embedder=embedder)
            modes: list[tuple[str, Retriever]] = [
                (m, build_retriever(settings, embedder=embedder, mode=m))
                for m in ("bm25", "dense", "hybrid")
            ]
            if reranker is not None:
                modes.append(
                    (
                        "hybrid+rerank",
                        build_retriever(
                            settings, embedder=embedder, mode="hybrid", reranker=reranker
                        ),
                    )
                )
            if translator is not None:
                best_name, best = modes[-1]
                modes.append(
                    (
                        f"{best_name}+translate",
                        TranslatingRetriever(
                            best, translator, load_prompt("v1", base.prompts_dir, "translate")
                        ),
                    )
                )
            for mode, retriever in modes:
                res = evaluate_retriever(retriever, questions, context_k=base.top_k)
                results.append(
                    {
                        "embedding_model": emb_name,
                        "chunk_words": words,
                        "n_chunks": len(chunks),
                        "mode": mode,
                        **res,
                    }
                )
                logger.info(
                    "%s w=%d %s: %s",
                    emb_name,
                    words,
                    mode,
                    {k: round(v, 3) for k, v in res["summary"].items()},
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"env": _env_info(), "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def evaluate_answers(service: AnswerService, questions: list[EvalQuestion]) -> dict[str, Any]:
    """Run the full pipeline on every question and score the answers.

    Args:
        service: Configured answer service (retriever + LLM + prompt).
        questions: Eval set.

    Returns:
        Summary metrics and per-question records.
    """
    records = []
    for i, q in enumerate(questions, start=1):
        ans = service.ask(q.question)
        context = "\n".join(h.chunk.text for h in ans.sources)
        cited_text = "\n".join(
            h.chunk.text for h in ans.sources if h.chunk.unit_id in ans.cited_articles
        )
        rec: dict[str, Any] = {
            "id": q.id,
            "lang": q.lang,
            "answerable": q.answerable,
            "question": q.question,
            "gold": q.gold_units,
            "answer": ans.answer,
            "cited": ans.cited_articles,
            "confidence": ans.confidence.value,
            "abstained": ans.abstained,
            "parse_error": ans.parse_error,
            "retrieved_units": unique_units(ans.sources),
            "latency_ms": ans.latency_ms,
            "raw_output": ans.raw_output,
            "reference": q.reference_answer,
        }
        if q.answerable and not ans.abstained:
            rec.update(citation_scores(ans.cited_articles, q.gold_units))
            # Faithfulness is measured against the *cited* passages only. An answer
            # that cites nothing is unsupported by construction (the user has no
            # passage to check it against), so it scores 0 instead of falling back
            # to the whole context, which would reward not citing.
            rec["faithfulness"] = faithfulness(ans.answer, cited_text) if cited_text else 0.0
            rec["numbers_grounded"] = numbers_grounded(ans.answer, context)
        records.append(rec)
        logger.info(
            "[%d/%d] %s abstained=%s cited=%s",
            i,
            len(questions),
            q.id,
            ans.abstained,
            ans.cited_articles,
        )

    return {"summary": summarize_answers(records), "records": records}


def _correctness(
    records: list[dict[str, Any]], check: Callable[[str, str], float | None]
) -> list[float]:
    """Apply a reference-based check to answerable records where it is defined.

    Args:
        records: Answerable records (with a ``reference`` field).
        check: e.g. :func:`yes_no_correct` (returns None when not applicable).

    Returns:
        One score per applicable record; abstentions and parse errors score 0.
    """
    scores = []
    for r in records:
        ref = r.get("reference") or ""
        value = check(r["answer"], ref)
        if value is None:
            continue
        scores.append(0.0 if r["abstained"] else value)
    return scores


def summarize_answers(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-question answer records into the reported metrics.

    Args:
        records: Records produced by :func:`evaluate_answers`.

    Returns:
        Summary metrics.
    """
    ans_q = [r for r in records if r["answerable"]]
    unans_q = [r for r in records if not r["answerable"]]
    answered = [r for r in ans_q if not r["abstained"]]
    en_answered = [r for r in answered if r["lang"] == "en"]
    lat = [r["latency_ms"]["total"] for r in records]

    def mean(values: list[float]) -> float:
        """Mean that is 0.0 for an empty list (e.g. a run where nothing was answered).

        Args:
            values: Values to average.

        Returns:
            The mean, or 0.0.
        """
        return statistics.mean(values) if values else 0.0

    def parse_error(r: dict[str, Any]) -> bool:
        """Whether a record's output was unparseable (absent in older result files).

        Args:
            r: One answer record.

        Returns:
            True on a parse error.
        """
        return bool(r.get("parse_error", False))

    summary = {
        "n_questions": len(records),
        # Citation accuracy over ALL answerable questions: an abstention counts as wrong.
        "citation_correct": mean([r.get("correct", 0.0) for r in ans_q]),
        "citation_precision": mean([r["precision"] for r in answered]),
        "citation_recall": mean([r["recall"] for r in answered]),
        # Unparseable output is shown as an abstention but is a failure, not a
        # correct "not in the text": only deliberate abstentions count here.
        "abstain_correct_unanswerable": mean(
            [float(r["abstained"] and not parse_error(r)) for r in unans_q]
        ),
        # Includes parse errors (the user sees "not in the text" either way);
        # they are also reported on their own in ``parse_error_rate``.
        "false_abstention_answerable": mean([float(r["abstained"]) for r in ans_q]),
        "parse_error_rate": mean([float(parse_error(r)) for r in records]),
        "faithfulness_en": mean([r["faithfulness"] for r in en_answered]),
        "numbers_grounded": mean([r["numbers_grounded"] for r in answered]),
        # Answer correctness against the reference answer (abstentions count as wrong).
        "yes_no_correct": mean(_correctness(ans_q, yes_no_correct)),
        "key_facts_recall": mean(_correctness(ans_q, key_facts_recall)),
        "latency_ms_mean": mean(lat),
        "latency_ms_p50": statistics.median(lat) if lat else 0.0,
        "latency_ms_max": max(lat) if lat else 0.0,
        "generation_ms_mean": mean([r["latency_ms"]["generation"] for r in records]),
        "retrieval_ms_mean": mean([r["latency_ms"]["retrieval"] for r in records]),
    }
    return summary


def rescore(path: Path, questions: list[EvalQuestion]) -> dict[str, Any]:
    """Re-apply (corrected) gold labels to a stored answer run without re-running the LLM.

    Only label-dependent fields change (gold, reference answer, citation scores and the
    answer-correctness checks derived from the reference);
    answers, citations, abstentions, faithfulness and latency are kept as recorded.

    Args:
        path: An ``answers_*.json`` file (rewritten in place).
        questions: Current eval set.

    Returns:
        The updated payload.
    """
    payload: dict[str, Any] = json.loads(path.read_text("utf-8"))
    gold = {q.id: q.gold_units for q in questions}
    reference = {q.id: q.reference_answer for q in questions}
    for rec in payload["records"]:
        rec["gold"] = gold[rec["id"]]
        rec["reference"] = reference[rec["id"]]
        if rec["answerable"] and not rec["abstained"]:
            rec.update(citation_scores(rec["cited"], rec["gold"]))
    payload["summary"] = summarize_answers(payload["records"])
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    return payload


def run_answer_eval(
    base: Settings,
    questions: list[EvalQuestion],
    llm: LLM,
    retriever: Retriever,
    prompt_versions: list[str],
    out_dir: Path,
    tag: str = "",
) -> list[dict[str, Any]]:
    """Evaluate the pipeline once per prompt version, sharing LLM and retriever.

    Args:
        base: Base settings (top_k, max_tokens, prompts dir, index dir...).
        questions: Eval set.
        llm: LLM provider (loaded once).
        retriever: Retriever (loaded once).
        prompt_versions: e.g. ``["v1", "v2"]``.
        out_dir: Folder for ``answers_<tag>_<version>.json``.
        tag: Run label (e.g. model name), used in file names.

    Returns:
        One result per prompt version.
    """
    # Chunk size and embedder are read from the index actually loaded, not from
    # settings, so the result file describes what was evaluated even if the
    # environment changed.
    index_meta = load_meta(base.index_dir)
    results = []
    for version in prompt_versions:
        set_seeds(base.seed)
        service = AnswerService(
            retriever=retriever,
            llm=llm,
            prompt=load_prompt(version, base.prompts_dir),
            top_k=base.top_k,
            max_tokens=base.max_tokens,
        )
        res = evaluate_answers(service, questions)
        payload = {
            "env": _env_info(),
            "config": {
                "llm": llm.name,
                "prompt_version": version,
                "retrieval_mode": base.retrieval_mode,
                "embedding_model": index_meta.get("embedding_model"),
                "reranker_model": base.reranker_model,
                "translate_queries": base.translate_queries,
                "chunk_words": index_meta.get("chunk_words"),
                "top_k": base.top_k,
                "max_tokens": base.max_tokens,
                "seed": base.seed,
            },
            **res,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"answers_{tag + '_' if tag else ''}{version}.json"
        (out_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        logger.info("%s %s: %s", tag, version, {k: round(v, 3) for k, v in res["summary"].items()})
        results.append(payload)
    return results
