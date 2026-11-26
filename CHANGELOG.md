# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Prompt v3 (yes/no rule, neutral worked example, keep exact figures) and two
  answer-correctness metrics, Yes/No correct and Key facts, applied to every
  stored run with `ai-act-rag rescore` (DECISIONS D13).
- `scripts/download_models.py` downloads both GGUFs (3B and 1.5B); `--llm`
  selects one.
- README: "Key results" table near the top (default 3B v3 vs the 1.5B with v3
  and v2), three recorded answers from the evaluation, a "Future work" section
  and a Spanish summary with the headline numbers.
- Final evaluation on the audited code: retrieval grid, 1.5B (prompt v1/v2/v3
  with query translation, v2 without) and 3B (v1/v2/v3), with per-question
  error attribution in `eval/results/summary.md`.

### Changed

- Default configuration: Qwen2.5-3B-Instruct Q4_K_M + prompt v3 (was 1.5B +
  v2); the 1.5B is the fast option (DECISIONS D12).
- Documentation states that "correct citation" and the lexical faithfulness
  metric do not check answer correctness (example: q07 in the default run).
- DECISIONS: model provenance (D02, D08) made consistent with the README; the
  1.5B GGUF is quantised locally and not byte-identical to the HF file.

### Fixed

- API: calls into the pipeline are serialised with a lock (503 "busy" after
  120 s), `/search` parameters are validated and `parse_error` is returned by
  `/ask` and the MCP `ask_ai_act` tool.
- Docker: llama.cpp is compiled without `-march=native`, so the image runs on
  other CPUs.
- `scripts/download_models.py --skip-llm` is honoured for the Hugging Face
  source.

## [0.1.0] - 2026-11-25

### Added

- Ingestion of Regulation (EU) 2024/1689 from the Official Journal XHTML
  (EUR-Lex / Cellar download, hash-verified PyPI mirror fallback): 113 articles,
  180 recitals and 13 annexes with titles, chapters and sections.
- Structure-aware chunking: chunks never cross a legal unit and carry a
  context header.
- Retrieval: BM25 implemented from scratch, dense retrieval with
  sentence-transformers and FAISS, hybrid fusion with Reciprocal Rank Fusion,
  optional cross-encoder reranker.
- Optional query translation (`TranslatingRetriever`) so Spanish questions
  retrieve well over the English corpus.
- Generation with versioned prompts (`prompts/answer_v1.txt`, `answer_v2.txt`),
  JSON-schema constrained decoding, Pydantic validation, citations grounded in
  the retrieved context and an explicit "not stated in the text" abstention.
- LLM providers: llama.cpp (local GGUF on CPU, default Qwen2.5-1.5B-Instruct
  Q4_K_M), Ollama and any OpenAI-compatible API.
- Reproducible evaluation on 40 questions: Hit@k and MRR@10 for a retrieval
  grid; citation precision/recall, abstention, lexical faithfulness, number
  grounding and latency end to end. Result tables are generated from
  `eval/results/` by `ai-act-rag report`.
- Interfaces: `ai-act-rag` CLI, FastAPI (`/ask`, `/search`, `/health`), MCP
  server (`search_ai_act`, `ask_ai_act`) and a Gradio demo.
- Dockerfile and GitHub Actions CI (ruff, mypy --strict, pytest without
  models).
