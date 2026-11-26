"""Turn ``eval/results/*.json`` into Markdown tables (summary.md + README section).

The README block between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->``
is fully regenerated, so published numbers always match the result files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(x: float) -> str:
    """Format a ratio as a percentage.

    Args:
        x: Ratio in [0, 1].

    Returns:
        e.g. ``"81.2%"``.
    """
    return f"{100 * x:.1f}%"


def _pct_opt(x: float | None) -> str:
    """Format an optional ratio (metrics missing from older result files).

    Args:
        x: Ratio in [0, 1], or None.

    Returns:
        e.g. ``"81.2%"``, or ``"n/a"``.
    """
    return "n/a" if x is None else _pct(x)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table.

    Args:
        headers: Column names.
        rows: Cell values.

    Returns:
        Markdown string.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def retrieval_table(data: dict[str, Any]) -> str:
    """Retrieval comparison table.

    Args:
        data: Content of ``retrieval.json``.

    Returns:
        Markdown table.
    """
    rows = []
    context_k = data["results"][0]["summary"].get("context_k") if data["results"] else None
    for r in data["results"]:
        s = r["summary"]
        rows.append(
            [
                r["embedding_model"],
                str(r["chunk_words"]),
                r["mode"],
                _pct(s["hit@1"]),
                _pct(s["hit@3"]),
                _pct(s["hit@5"]),
                _pct(s["hit@10"]),
                f"{s['mrr@10']:.3f}",
                _pct_opt(s.get(f"gold_in_context@{s.get('context_k')}")),
                _pct(s.get("hit@5_en", 0.0)),
                _pct(s.get("hit@5_es", 0.0)),
                f"{s['latency_ms_mean']:.0f}",
            ]
        )
    return _table(
        [
            "Embedder",
            "Chunk words",
            "Retriever",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Hit@10",
            "MRR@10",
            f"Gold in LLM context (top-{context_k} chunks)" if context_k else "Gold in LLM context",
            "Hit@5 EN",
            "Hit@5 ES",
            "ms/query",
        ],
        rows,
    )


def retrieval_label(config: dict[str, Any]) -> str:
    """Describe the retrieval setup of an answer run.

    The reranker is only applied by the hybrid retriever, so ``+rerank`` is shown
    only for ``hybrid`` even if a reranker model is configured.

    Args:
        config: ``config`` block of an ``answers_*.json`` file.

    Returns:
        e.g. ``"hybrid+rerank+translate"``.
    """
    label: str = config["retrieval_mode"]
    if config["retrieval_mode"] == "hybrid" and config.get("reranker_model"):
        label += "+rerank"
    if config.get("translate_queries"):
        label += "+translate"
    return label


def answers_table(runs: list[dict[str, Any]]) -> str:
    """End-to-end answer quality table, one row per (model, prompt) run.

    Args:
        runs: Contents of ``answers_*.json`` files.

    Returns:
        Markdown table.
    """
    rows = []
    for run in runs:
        c, s = run["config"], run["summary"]
        rows.append(
            [
                c["llm"].split(":", 1)[-1],
                c["prompt_version"],
                retrieval_label(c),
                _pct(s["citation_correct"]),
                _pct(s["citation_precision"]),
                _pct_opt(s.get("citation_recall")),
                _pct(s["abstain_correct_unanswerable"]),
                _pct(s["false_abstention_answerable"]),
                _pct_opt(s.get("parse_error_rate")),
                _pct_opt(s.get("yes_no_correct")),
                _pct_opt(s.get("key_facts_recall")),
                _pct(s["faithfulness_en"]),
                _pct(s["numbers_grounded"]),
                f"{s['latency_ms_mean'] / 1000:.1f}",
                f"{s['latency_ms_p50'] / 1000:.1f}",
            ]
        )
    return _table(
        [
            "LLM",
            "Prompt",
            "Retrieval",
            "Correct citation",
            "Citation precision (answered)",
            "Citation recall (answered)",
            "Correct 'not in text'",
            "False 'not in text'",
            "Parse errors",
            "Yes/No correct",
            "Key facts",
            "Faithfulness (EN)",
            "Numbers grounded",
            "Latency mean (s)",
            "Latency p50 (s)",
        ],
        rows,
    )


def classify_errors(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Attribute each failure to a stage of the pipeline.

    * ``retrieval_miss``: answerable, no gold unit in the context -> retrieval's fault.
    * ``generation_miss``: answerable, gold unit in the context but not cited.
    * ``false_abstention``: answerable, the system said "not in the text".
    * ``hallucination``: not answerable, the system answered anyway.
    * ``parse_error``: the LLM output could not be parsed (any question).

    Args:
        records: Per-question records of one ``answers_*.json`` run.

    Returns:
        Error type -> question ids.
    """
    errors: dict[str, list[str]] = {
        "retrieval_miss": [],
        "generation_miss": [],
        "false_abstention": [],
        "hallucination": [],
        "parse_error": [],
    }
    for r in records:
        gold = set(r["gold"])
        if r.get("parse_error"):
            errors["parse_error"].append(r["id"])
        elif not r["answerable"]:
            if not r["abstained"]:
                errors["hallucination"].append(r["id"])
        elif not gold & set(r["retrieved_units"]):
            errors["retrieval_miss"].append(r["id"])
        elif r["abstained"]:
            errors["false_abstention"].append(r["id"])
        elif not gold & set(r["cited"]):
            errors["generation_miss"].append(r["id"])
    return errors


def errors_table(runs: list[dict[str, Any]]) -> str:
    """Error attribution table, one row per run.

    Args:
        runs: Contents of ``answers_*.json`` files.

    Returns:
        Markdown table (counts, with question ids).
    """
    rows = []
    for run in runs:
        c = run["config"]
        errs = classify_errors(run["records"])
        name = c["llm"].split(":", 1)[-1] + " / " + c["prompt_version"]
        if c.get("translate_queries"):
            name += " / translate"
        rows.append([name] + [f"{len(v)} ({', '.join(v)})" if v else "0" for v in errs.values()])
    return _table(
        [
            "Run",
            "Retrieval miss",
            "Generation miss (gold retrieved, not cited)",
            "False 'not in text'",
            "Hallucination on unanswerable",
            "Unparseable output",
        ],
        rows,
    )


def build_markdown(results_dir: Path, with_errors: bool = False) -> str:
    """Build the full results section.

    Args:
        results_dir: Folder with result JSON files.
        with_errors: Also include the error-attribution table (summary.md only).

    Returns:
        Markdown text.
    """
    parts: list[str] = []
    retrieval = results_dir / "retrieval.json"
    if retrieval.exists():
        data = json.loads(retrieval.read_text("utf-8"))
        env = data["env"]
        n_retrieval = data["results"][0]["summary"]["n_questions"] if data["results"] else 0
        parts += [
            f"### Retrieval ({n_retrieval} answerable questions, unit-level Hit@k)",
            "",
            retrieval_table(data),
            "",
            f"_Run {env['date_utc']} on {env['cpu_count']} CPU cores, Python {env['python']}._",
            "",
        ]
    runs = [json.loads(p.read_text("utf-8")) for p in sorted(results_dir.glob("answers_*.json"))]
    if runs:
        first = runs[0]
        records = first["records"]
        n_unanswerable = sum(not r["answerable"] for r in records)
        n_answerable = len(records) - n_unanswerable
        config, env = first["config"], first["env"]
        provider = config["llm"].split(":", 1)[0]
        parts += [
            f"### End-to-end answers ({n_answerable} answerable questions, "
            f"{n_unanswerable} not answerable from the text)",
            "",
            answers_table(runs),
            "",
            f"_{provider}, temperature 0, top-k = {config['top_k']} chunks of "
            f"{config['chunk_words']} words; run on {env['cpu_count']} CPU cores._",
            "",
        ]
        if with_errors:
            parts += [
                "### Where the errors come from",
                "",
                errors_table(runs),
                "",
            ]
    return "\n".join(parts).strip() + "\n"


def write_report(results_dir: Path, readme: Path | None = None) -> str:
    """Write ``summary.md`` and refresh the README results block.

    Args:
        results_dir: Folder with result JSON files.
        readme: README to update (skipped if None or without markers).

    Returns:
        The generated Markdown.
    """
    md = build_markdown(results_dir)
    full = build_markdown(results_dir, with_errors=True)
    (results_dir / "summary.md").write_text("# Evaluation results\n\n" + full, encoding="utf-8")
    if readme is not None and readme.exists():
        text = readme.read_text("utf-8")
        if START in text and END in text:
            head, rest = text.split(START, 1)
            _, tail = rest.split(END, 1)
            readme.write_text(f"{head}{START}\n{md}{END}{tail}", encoding="utf-8")
    return md
