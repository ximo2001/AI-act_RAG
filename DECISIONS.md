# Decisions

Architecture and design decision log.
Each entry: context → decision → why / consequences.

## D01 — Data source behind a restricted network: official OJ XHTML via a PyPI mirror
- **Context:** the environment this was built in had no network access to
  `eur-lex.europa.eu` / `publications.europa.eu` (blocked by its proxy).
- **Decision:** `scripts/download_ai_act.py` downloads the official text from
  EUR-Lex/Cellar (this is the documented path). When EUR-Lex is unreachable it falls back
  to the **byte-identical Official Journal XHTML** (`32024R1689.en.xhtml`, CONVEX
  converter output, ELI `http://data.europa.eu/eli/reg/2024/1689/oj`) that ships
  inside the `eu-ai-act-classifier==0.4.0` sdist on PyPI.
- **Integrity:** the PyPI mirror path verifies the file against a **pinned SHA-256**
  (and the sdist is pinned by version). The EUR-Lex path cannot pin a hash (the
  HTML view may change byte-wise), so it **validates the structure** (113 articles,
  180 recitals, 13 annexes) before saving, and `build-index` **records the source
  SHA-256 in the index metadata** (`meta.json`).
- **Consequence:** the parser targets the OJ XHTML format (same `id="art_N"`,
  `rct_N`, `anx_X` structure that EUR-Lex's HTML view embeds). Re-run the download
  locally to confirm the structure against EUR-Lex.

## D02 — Models behind a restricted network: Docker Hub OCI artifacts instead of Hugging Face
- **Context:** `huggingface.co` blocked; Docker Hub registry reachable.
- **Decision:** the GGUF chat models come from Docker Hub's `ai/qwen2.5`
  artifacts: the 3B (default LLM) is the ready Q4_K_M artifact
  `ai/qwen2.5:3B-Q4_K_M`, whose GGUF layer is downloaded as is; the 1.5B is the F16
  artifact, quantised to Q4_K_M locally (see D08).
  Sentence-transformers weights come from Weaviate's `semitechnologies/*`
  inference images (they bake the HF model folders). `scripts/download_models.py`
  uses Hugging Face by default (`--source dockerhub` is the fallback path) and
  downloads both GGUFs (`--llm 3b` / `--llm 1.5b` for one, `--skip-llm` for none).

## D03 — Packaging: uv + pyproject.toml with `==` pins (replaces requirements.txt)
- **Context:** the repository template prescribes `requirements.txt` with `==`
  pins; uv adds a lockfile for transitive dependencies and fast, reproducible
  installs.
- **Decision:** direct deps pinned with `==` in `pyproject.toml`, transitive deps
  locked in `uv.lock`. Heavy deps are optional extras (`ml`, `llm`, `serve`) so CI
  and unit tests never install torch or download models.

## D04 — Architecture: pragmatic hexagonal in a single package
- **Context:** the repository template prescribes a full hexagonal tree
  (`ports/in`, `use_cases/`, `dto/`, ...); with a single use case most of those
  folders would be near-empty.
- **Decision:** one package `src/ai_act_rag/` with three layer folders (plus
  `evaluation/`, an offline client of the other three).
  `domain/` = pydantic models + `Protocol` ports (Embedder, LLM, ...), no deps.
  `application/` = pure-Python logic (chunking, BM25, RRF, answering use case).
  `infrastructure/` = EUR-Lex parser, sentence-transformers, FAISS, LLM providers,
  FastAPI, MCP, Gradio, config/wiring. Ports are `typing.Protocol`s, so test fakes
  need no inheritance. No DI framework: one `container.py` builds everything.

## D05 — BM25 implemented from scratch
- ~60 lines, unit-tested against the Okapi formula. Avoids a dependency and makes
  the scoring (and the light stemming/stopwords) explicit in an interview.

## D06 — Retrieval defaults chosen from the eval, not a priori
- Grid: {bge-small-en-v1.5, paraphrase-multilingual-MiniLM-L12-v2} x {150, 400 words}
  x {BM25, dense, hybrid RRF, hybrid + cross-encoder, hybrid + cross-encoder + query
  translation}. Default:
  **bge-small-en-v1.5, 150-word chunks, hybrid + ms-marco-MiniLM-L-6-v2 reranker**,
  chosen from the retrieval grid in `eval/results/retrieval.json` (see the README
  table): Hit@5 90.6% and MRR@10 0.719 with the reranker vs 78.1% and 0.672 for
  hybrid alone; 150-word chunks beat 400 on Hit@5 and MRR@10 in every bge-small row
  (hybrid + reranker: Hit@5 90.6% vs 81.2%). The reranker costs 515 ms/query at 150
  words (1270 at 400), small next to LLM generation.
- The multilingual paraphrase model is clearly worse **dense-only**: it is trained
  for sentence similarity, not retrieval, and truncates long chunks (128-token
  training length); at 400 words its dense Hit@5 is 15.6%. Once fused with BM25 and
  reranked at 150 words, it is comparable to bge-small (Hit@5 90.6% for both, MRR@10
  0.748 vs 0.719), about one question apart.
- `intfloat/multilingual-e5-small`, a retrieval-trained multilingual embedder and the
  natural next candidate, was not reachable from the build environment (no HF; not in the Docker
  Hub images found) - swap it in with `AIACT_EMBEDDING_MODEL` (README "Future work").

## D07 — Structured output: JSON schema constrained decoding + Pydantic
- llama.cpp turns the Pydantic JSON schema into a grammar, so the 1.5B model cannot
  emit malformed JSON *as long as generation is not cut by `max_tokens`*; a truncated
  object is still invalid. Pydantic validates every output (this case, and providers
  without grammar support). Unparseable output => abstain, never crash, and the
  answer is flagged `parse_error` so the evaluation counts it as a failure (in the
  results: q17 in the 1.5B v1 run, cut by the token budget, and q40 in the 1.5B v3 run,
  a repetition loop that ran into it).
- Citations are normalised and filtered to units present in the context: the
  system never displays a citation it did not retrieve.

## D08 — LLM: Qwen2.5-Instruct Q4_K_M on CPU (3B default, 1.5B fast option)
- Both sizes are evaluated with every prompt on 4 CPU cores (translation on). With the
  default prompt v3 the 3B has p50 latency 19.4 s and the 1.5B 13.9 s; the choice of
  the 3B as default is argued in D12.
- The 1.5B GGUF was produced by quantising the official F16 GGUF to Q4_K_M with
  llama.cpp's own quantiser. It is not byte-identical to the Q4_K_M file published
  on HF, so a run with that file may differ slightly. The 3B used for the results is
  Docker Hub's ready Q4_K_M artifact (D02); it was not compared byte-wise with the HF
  file either.
- "transformers" provider not implemented: llama.cpp covers the local CPU case
  with less RAM and faster inference; Ollama covers the laptop case.

## D09 — Faithfulness metric: lexical support + number grounding (no LLM judge)
- An LLM-as-judge with a 1.5B model would be noisier than the thing it judges.
  Instead: a sentence is "supported" if >= 60% of its content words appear in the
  cited passages; plus "numbers grounded": every number in the answer must appear in
  the context. Both are explainable and deterministic; limitations are documented
  (paraphrases lower the score; contradictions reusing the same words pass;
  lexical faithfulness is only computed for English answers).
- **Known gap:** neither metric, nor "correct citation", checks that the answer
  agrees with the cited text. In the 1.5B v2 run q07 cites Article 5 with
  faithfulness 1.0 but answers "Yes" to a practice Article 5 prohibits. D13 adds two
  narrow correctness checks against the reference answers (yes/no polarity, key
  figures); full answer correctness (a hand-labelled subset, then an LLM judge
  calibrated on it) is still the first item of the README "Future work".
- A one-word answer ("Yes", "No") has no content words and scores faithfulness 0,
  however right it is; this is most of the faithfulness drop of prompt v3 (D13).

## D10 — Prompt iteration on the eval set (and its caveat)
- The first end-to-end run (1.5B, v1 vs v2) was used as a diagnostic: the 1.5B model
  copied the content of v2's worked example (a logs-retention example answered a
  human-oversight question) and answered Spanish questions in English.
- Fixes were generic, not question-specific: the example moved to a topic absent from
  the eval set (CE marking) and the answer language is now stated explicitly in the
  user message ("Write the answer in Spanish"), computed from the question.
- Caveat: with only 40 questions there is no separate dev/test split, so the reported
  v2 numbers are slightly optimistic. Documented in the README methodology caveats.
  Prompt v3 is a second iteration on the same set, with the same caveat (D13).

## D11 — Query translation (on by default), evaluated rather than assumed
- `TranslatingRetriever` wraps any retriever and translates non-English queries to
  English with the same LLM before searching (one short extra LLM call; 1.5B v2 p50
  latency 13.7 s with translation vs 12.0 s without).
  Generation still sees the original question, so the answer stays in Spanish.
- Chosen over indexing the Spanish text (doubles the index, citations must be mapped)
  and over a multilingual embedder (none suitable was reachable from the build environment).

## D12 — Default configuration from the final eval
- **Decision:** Qwen2.5-3B-Instruct Q4_K_M + prompt v3 + bge-small-en-v1.5 with
  150-word chunks + hybrid RRF + cross-encoder reranker + query translation. Numbers
  from `eval/results/` (README "Results"). With 32 answerable questions one question is
  ≈ 3.1 points; with 8 unanswerable questions 12.5 points; with the 7 yes/no questions
  of "Yes/No correct" ≈ 14.3 points.
- **Why 3B + v3:** it is the only run that is best or tied-best on every
  correctness column that matters for a legal assistant: correct citation 90.6% (best),
  correct 'not in text' 100.0% and false 'not in text' 0.0% (it neither answers
  out-of-scope questions nor refuses valid ones), Yes/No correct 85.7% (tied with
  3B v1), parse errors 0.0%, numbers grounded 100.0%. Its errors: 3 generation misses
  (q12, q22, q25: a recital or neighbouring article cited instead of the gold one) and
  0 answered unanswerable questions. p50 latency is 19.4 s (mean 21.9 s) on 4 CPU
  cores, faster than the 3B with v2 (26.9 s) because its answers are shorter.
- **What it does not win:** key facts 69.4% vs 77.8% for 3B v1 (which abstains on only
  25.0% of the unanswerable questions), and faithfulness (EN) 77.8% vs 95.7% for 3B v2,
  mostly because v3 answers yes/no questions with a bare "Yes"/"No" (D13). Its one
  yes/no error is q07: a bare "Yes" to a practice Article 5 prohibits.
- **The 1.5B is the fast option**
  (`AIACT_LLM_MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf`, p50 13.9 s with v3,
  13.7 s with v2). It gets only 4 of the 8 unanswerable questions right with either prompt
  (correct 'not in text' 50.0%: with v2 it answers 4 of them; with v3 it answers 3 and
  fails q40 with a parse error). v3 vs v2 on the 1.5B: correct citation 84.4% vs
  78.1% and citation precision 62.6% vs 59.2%, but key facts 44.4% vs 52.8%,
  faithfulness (EN) 74.1% vs 96.3%, numbers grounded 96.1% vs 100.0% and one parse
  error (q40, 2.5%); Yes/No correct is 71.4% with both (it answers "Yes" to q07 and q11).
  Use `AIACT_PROMPT_VERSION=v2` with it to keep more figures and no parse errors.
- **Caveat:** v3 was written after the v2 error analysis on the same 40 questions and
  the correctness metrics were added at the same time (D13); there is no held-out
  split, so the v3 numbers, and its Yes/No gain above all, are optimistic.

## D13 — Prompt v3 and answer-correctness metrics
- **Context:** "correct citation" rewarded answers with the wrong polarity: in the
  1.5B v2 run q07 and q11 cite the right article and answer "Yes" where the text says
  "No" (scraping facial images is prohibited; military uses are excluded). The 3B with
  v2 had the opposite failure: it answered "not in the text" on 5 answerable questions
  (q06, q07, q11, q12, q30), all yes/no questions whose passage prohibits or excludes
  the thing asked about. Neither error showed in the existing metrics.
- **Metrics:** two checks against the `reference_answer` of each answerable question
  (`evaluation/metrics.py`), where abstentions and parse errors score 0:
  - **Yes/No correct** - the 7 answerable questions whose reference starts with
    "Yes"/"No" ("Sí"/"No" in Spanish); the answer must start with the same word.
  - **Key facts** - over the 12 answerable questions whose reference states numbers,
    the share of those numbers (parenthesised citations such as "(Art. 99(3))"
    excluded) that appear in the answer.
  They are computed offline from the stored answers with `ai-act-rag rescore`, which
  re-applies the labels without calling the LLM; earlier metrics are unchanged.
- **Prompt v3** (`prompts/answer_v3.txt`) = v2 plus three generic fixes from the v2
  error analysis: a yes/no rule ("start with Yes or No; a passage that prohibits,
  excludes or exempts the thing asked about IS the answer, not NOT_IN_TEXT"), the
  worked example moved to the AI Board's composition (the 1.5B had copied the
  CE-marking example into q31), and an instruction to keep exact figures.
- **Effect:** on the 3B, Yes/No correct 14.3% -> 85.7% and false 'not in text'
  15.6% -> 0.0%, with correct 'not in text' still 100.0%. On the 1.5B, correct citation
  78.1% -> 84.4% but key facts 52.8% -> 44.4%; Yes/No correct stays 71.4%. The q31
  copy of the example is gone.
- **Side effect:** both models now answer all 7 yes/no questions with a single word
  ("Yes"/"No") instead of a sentence, despite rule 1 asking for complete sentences, and
  the 3B also answers three figure questions with the bare figure (q04, q08, q09; with
  v2 only q04 was a bare figure).
  The lexical faithfulness metric scores a one-word "Yes"/"No" 0, which explains most
  of the drop (3B 95.7% -> 77.8%, 1.5B 96.3% -> 74.1%): without the 6 English bare
  yes/no answers, the 3B v3's remaining English answers score 100% and the 1.5B's
  95.2% (computed from the `answers_*.json` records). The 3B's answers are also
  shorter overall (36.5 vs 61.9 words per answered English question); the 1.5B's are
  not (52.1 vs 47.0). For users a bare "No" with its citation is correct but curt;
  requiring a justifying sentence is the next prompt change.
- **Overfitting caveat:** v3 was written after reading the failures of these same 40
  questions, and the yes/no rule targets exactly the questions the new metric scores.
  There is no held-out split, so v3's gains, the Yes/No gain above all, are optimistic
  until confirmed on new questions (README "Future work").
