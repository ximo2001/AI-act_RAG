"""Command-line entry point: ``ai-act-rag <command>``.

Commands:
    build-index      parse + chunk + embed the downloaded AI Act
    ask              answer one question
    eval-retrieval   dense vs BM25 vs hybrid (x chunk size x embedder)
    eval-answers     end-to-end eval for one or more prompt versions
    report           regenerate eval/results/summary.md and the README table
    rescore          re-apply the current labels to stored answer runs (no LLM)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ai_act_rag.domain.exceptions import AIActRAGError, IndexNotBuiltError
from ai_act_rag.evaluation.report import write_report
from ai_act_rag.evaluation.runner import (
    load_questions,
    rescore,
    run_answer_eval,
    run_retrieval_eval,
)
from ai_act_rag.infrastructure.config import Settings
from ai_act_rag.infrastructure.container import (
    build_answer_service,
    build_index,
    build_llm,
    build_retriever,
    set_seeds,
    wrap_with_translation,
)
from ai_act_rag.infrastructure.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder

QUESTIONS = Path("eval/questions.jsonl")
RESULTS = Path("eval/results")


def _cmd_build_index(args: argparse.Namespace) -> None:
    """Build the index with the current settings.

    Args:
        args: Parsed CLI args.
    """
    settings = Settings.from_env(chunk_words=args.chunk_words)
    if not settings.source_path.is_file():
        raise IndexNotBuiltError(
            f"AI Act source not found at {settings.source_path}. "
            "Run: uv run python scripts/download_ai_act.py (or ./scripts/bootstrap.sh)"
        )
    chunks = build_index(settings)
    print(f"Indexed {len(chunks)} chunks")


def _cmd_ask(args: argparse.Namespace) -> None:
    """Answer one question and print JSON.

    Args:
        args: Parsed CLI args.
    """
    service = build_answer_service(Settings.from_env(prompt_version=args.prompt))
    answer = service.ask(args.question)
    print(
        json.dumps(
            answer.model_dump(exclude={"sources", "raw_output"}), indent=2, ensure_ascii=False
        )
    )
    print("Sources:", ", ".join(h.chunk.chunk_id for h in answer.sources))


def _cmd_eval_retrieval(args: argparse.Namespace) -> None:
    """Run the retrieval comparison grid.

    Args:
        args: Parsed CLI args.
    """
    settings = Settings.from_env()
    set_seeds(settings.seed)
    embedders = {Path(m).name: SentenceTransformerEmbedder(m) for m in args.embedding_models}
    reranker = CrossEncoderReranker(args.reranker) if args.reranker else None
    translator = None
    if args.translate:
        translator = build_llm(settings)
    run_retrieval_eval(
        settings,
        load_questions(QUESTIONS),
        embedders,
        args.chunk_words,
        reranker,
        RESULTS / "retrieval.json",
        translator=translator,
    )


def _cmd_eval_answers(args: argparse.Namespace) -> None:
    """Run the end-to-end answer evaluation.

    Args:
        args: Parsed CLI args.
    """
    settings = Settings.from_env(max_tokens=args.max_tokens)
    questions = load_questions(QUESTIONS)
    if args.limit:
        questions = questions[: args.limit]
    llm = build_llm(settings)
    retriever = wrap_with_translation(build_retriever(settings), llm, settings)
    run_answer_eval(
        settings,
        questions,
        llm=llm,
        retriever=retriever,
        prompt_versions=args.prompts,
        out_dir=RESULTS,
        tag=args.tag,
    )


def _cmd_report(args: argparse.Namespace) -> None:
    """Regenerate the Markdown report and README table from results.

    Args:
        args: Parsed CLI args.
    """
    write_report(RESULTS, Path("README.md"))


def _cmd_rescore(args: argparse.Namespace) -> None:
    """Re-apply the current labels/reference answers to stored runs (no LLM calls).

    Args:
        args: Parsed CLI args.
    """
    questions = load_questions(QUESTIONS)
    for path in sorted(RESULTS.glob("answers_*.json")):
        rescore(path, questions)
        print(f"rescored {path}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="ai-act-rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-index", help="parse, chunk and embed the AI Act")
    p.add_argument("--chunk-words", type=int, default=None)
    p.set_defaults(func=_cmd_build_index)

    p = sub.add_parser("ask", help="answer a question")
    p.add_argument("question")
    p.add_argument("--prompt", default=None, help="prompt version (default: settings)")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("eval-retrieval", help="compare retrieval configurations")
    p.add_argument("--embedding-models", nargs="+", default=["models/bge-small-en-v1.5"])
    p.add_argument("--chunk-words", nargs="+", type=int, default=[150, 400])
    p.add_argument("--reranker", default="", help="cross-encoder for a hybrid+rerank row")
    p.add_argument(
        "--translate", action="store_true", help="add a row translating ES queries with the LLM"
    )
    p.set_defaults(func=_cmd_eval_retrieval)

    p = sub.add_parser("eval-answers", help="end-to-end answer evaluation")
    p.add_argument("--prompts", nargs="+", default=["v1", "v2"])
    p.add_argument("--tag", default="", help="label for the result files")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions")
    p.add_argument(
        "--max-tokens", type=int, default=None, help="generation budget (default: settings)"
    )
    p.set_defaults(func=_cmd_eval_answers)

    p = sub.add_parser("report", help="regenerate result tables")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("rescore", help="re-apply labels to stored answer runs (no LLM)")
    p.set_defaults(func=_cmd_rescore)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        args.func(args)
    except AIActRAGError as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
