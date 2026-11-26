#!/usr/bin/env bash
# Reproduce every number in the README (CPU: ~1h15 on 4 cores).
# Requires ./scripts/bootstrap.sh (data + models, both GGUFs: the 3B default and the 1.5B).
set -euo pipefail
SMALL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
LARGE=models/qwen2.5-3b-instruct-q4_k_m.gguf

# 1. Retrieval grid (+ a row translating Spanish queries with the 1.5B model).
AIACT_LLM_MODEL=$SMALL uv run ai-act-rag -v eval-retrieval \
  --embedding-models models/bge-small-en-v1.5 models/paraphrase-multilingual-MiniLM-L12-v2 \
  --chunk-words 150 400 --reranker models/ms-marco-MiniLM-L-6-v2 --translate

# 2. End-to-end answers (default retrieval: bge-small, 150 words, hybrid + reranker,
#    Spanish questions translated to English for retrieval).
uv run ai-act-rag build-index
AIACT_LLM_MODEL=$SMALL AIACT_TRANSLATE_QUERIES=1 \
  uv run ai-act-rag -v eval-answers --prompts v1 v2 v3 --tag qwen2.5-1.5b
# Ablation: same model and prompt without query translation.
AIACT_LLM_MODEL=$SMALL AIACT_TRANSLATE_QUERIES=0 \
  uv run ai-act-rag -v eval-answers --prompts v2 --tag qwen2.5-1.5b-notranslate
if [ -f "$LARGE" ]; then
  AIACT_LLM_MODEL=$LARGE AIACT_TRANSLATE_QUERIES=1 \
    uv run ai-act-rag -v eval-answers --prompts v1 v2 v3 --tag qwen2.5-3b
fi

# 3. Tables (eval/results/summary.md + README block).
uv run ai-act-rag report
