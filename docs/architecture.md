# Architecture

AI Act RAG is a **pragmatic hexagonal** (ports and adapters) application in a
single Python package, `src/ai_act_rag/`. Design decisions and their rationale
are logged in [`DECISIONS.md`](../DECISIONS.md) (see D04 for this layout).

## Layout

```
src/ai_act_rag/
  domain/            pure types and interfaces, no I/O
    models.py        LegalUnit, Chunk, RetrievedChunk, LLMAnswer, Answer (Pydantic)
    ports.py         Embedder, Retriever, Reranker, LLM (typing.Protocol)
    exceptions.py    AIActRAGError and subclasses (ParsingError, IndexNotBuiltError, ...)
  application/       pure-Python logic, depends only on domain
    chunking.py      structural chunking: one unit -> chunks of <= N words, with a header
    bm25.py          Okapi BM25 implemented from scratch (tokenizer, light stemming)
    retrieval.py     DenseRetriever, reciprocal_rank_fusion, HybridRetriever
    translation.py   TranslatingRetriever (decorator over any Retriever)
    prompts.py       versioned prompt templates loaded from prompts/*.txt
    answering.py     AnswerService: the ask() use case
  infrastructure/    adapters and wiring
    eurlex.py        download (EUR-Lex / Cellar, PyPI mirror fallback) + OJ XHTML parser
    embeddings.py    sentence-transformers embedder and cross-encoder reranker
    index_store.py   chunks.jsonl + FAISS IndexFlatIP + meta.json
    llm/             llama.cpp, Ollama and OpenAI-compatible providers
    config.py        Settings: frozen dataclass read from AIACT_* env vars
    container.py     composition root: builds adapters and the AnswerService
    api.py           FastAPI: POST /ask, GET /search, GET /health
    mcp_server.py    MCP server: search_ai_act, ask_ai_act tools
    demo.py          Gradio demo
  evaluation/        offline evaluation
    metrics.py       Hit@k, MRR, citation P/R, lexical faithfulness, number grounding,
                     yes/no polarity, key facts
    runner.py        retrieval grid, end-to-end answer evaluation -> eval/results/*.json,
                     rescore (re-apply labels and metrics to stored runs, no LLM)
    report.py        Markdown tables from eval/results/ (summary.md + README block)
  cli.py             ai-act-rag build-index | ask | eval-retrieval | eval-answers | report | rescore
```

Outside the package:

```
prompts/             answer_v1..v3.txt, translate_v1.txt (versioned, evaluated; v3 default)
eval/                questions.jsonl (40 labelled questions) and results/ (JSON + summary.md)
scripts/             download_ai_act.py, download_models.py, bootstrap.sh, run_all_evals.sh
data/, models/       downloaded text, built index and model weights (not committed)
tests/               unit/ (fakes, no models), integration/ (real corpus, skipped if absent)
```

## Layers

**Domain.** Pydantic models for the corpus (`LegalUnit` = article, recital or
annex; `Chunk`; `RetrievedChunk`) and for answers (`LLMAnswer` is the schema the
LLM must fill; `Answer` is what the system returns, with sources and latency).
`ports.py` declares the four capabilities the application needs as
`typing.Protocol`s. Any object with matching methods satisfies a port, so
adapters and test fakes (`tests/fakes.py`) need no inheritance.

**Application.** Everything that is the "RAG logic" and can be unit-tested
without models or network:

- `chunking.py` splits each legal unit along its paragraphs and prefixes every
  chunk with a header (`Article 5 - Prohibited AI practices | Chapter II - ...`),
  so a chunk is self-describing for BM25, the embedder and the LLM.
- `bm25.py` and `DenseRetriever` both implement `Retriever`. The dense
  retriever depends on a small `VectorIndex` protocol (FAISS in production,
  `NumpyIndex` in tests).
- `HybridRetriever` takes the top candidates from each retriever, fuses them
  with Reciprocal Rank Fusion (`rrf_k=60`; rank-based, so BM25 and cosine
  scores need no calibration) and optionally reranks with a `Reranker`.
- `TranslatingRetriever` wraps any retriever: non-English queries are
  translated to English by the LLM before searching. The answer prompt still
  receives the original question, so the answer stays in the user's language.
- `AnswerService.ask()` is the use case (see the flow below).

**Infrastructure.** Concrete technology: the EUR-Lex parser (relies on the
stable `art_N` / `rct_N` / `anx_X` ids of the Official Journal XHTML and checks
113 articles, 180 recitals and 13 annexes), sentence-transformers, FAISS, LLM
providers, settings, and the three entry points (API, MCP, Gradio) plus the
CLI. Heavy libraries are imported lazily inside `container.py` builders, so
CI runs without the `ml`/`llm` extras.

**Evaluation.** Pure metric functions, runners that write JSON to
`eval/results/`, and a report generator. The README result tables are not
typed by hand: `ai-act-rag report` regenerates them from the result files (the
block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`, plus
`eval/results/summary.md` with the per-question error attribution).

## Request flow: `AnswerService.ask(question)`

1. An entry point (`api.py`, `mcp_server.py`, `demo.py` or `cli.py`) gets the
   service from `container.build_answer_service(Settings.from_env())`, built
   once per process (the API builds it at startup and serialises calls with a
   lock, because llama.cpp is not thread-safe).
2. **Retrieve.** `retriever.search(question, k=top_k)`. With the defaults the
   retriever chain is `TranslatingRetriever` (if `AIACT_TRANSLATE_QUERIES` is
   on) -> `HybridRetriever(dense + BM25, RRF)` -> cross-encoder reranker.
3. **Prompt.** `format_context(hits)` renders the passages with their unit
   ids; the versioned template (`prompts/answer_<version>.txt`) is filled with
   the context, the question and the answer language detected from the
   question (English or Spanish).
4. **Generate.** `llm.generate(..., json_schema=AnswerService.json_schema())`.
   llama.cpp turns the schema into a grammar (Ollama uses `format`), so the
   output is constrained JSON.
5. **Post-process.** The output is validated with Pydantic (`LLMAnswer`);
   unparseable output becomes an abstention, never a crash. Citations are
   normalised (`Art. 5(1)` -> `Article 5`) and **kept only if the unit was in
   the retrieved context**. An abstention sentinel (`NOT_IN_TEXT` or an
   equivalent phrase in the first sentence) becomes a localised "not stated in
   the text" answer.
6. The `Answer` is returned with its sources and per-stage latency
   (retrieval, generation, total; "retrieval" includes query translation and
   reranking); adapters map it to their own response schema (e.g.
   `AskResponse` in the API).

## Dependency rule

Imports point inwards only: `infrastructure -> application -> domain`, and
`evaluation` uses application and infrastructure as a client. `domain` and
`application` never import FastAPI, FAISS, sentence-transformers, llama.cpp or
httpx; they talk to those through the ports. `container.py` is the only module
that knows which implementation backs each port.

## Why "pragmatic" hexagonal

- **Single package, three layer folders** (plus `evaluation/`). No separate
  `ports/in`, `use_cases/` or DTO layers: the only use case is answering a
  question, and one service class says that more clearly than a tree of
  near-empty modules.
- **Protocols instead of abstract base classes.** Structural typing keeps
  adapters and test fakes decoupled from the interfaces and is checked by
  `mypy --strict`.
- **No DI framework.** `container.py` is a handful of plain functions that
  read `Settings` and build objects; tests and the evaluation runner pass
  pre-built fakes or reused components through optional arguments.
- **Pydantic in the domain.** It is a validation library rather than an
  infrastructure detail, and it gives the LLM output schema for free.
