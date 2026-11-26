#!/usr/bin/env bash
# Download the AI Act + models and build the index (run once, from the repo root).
#   MODEL_SOURCE=dockerhub ./scripts/bootstrap.sh   # if huggingface.co is not reachable
set -euo pipefail
uv run python scripts/download_ai_act.py
uv run python scripts/download_models.py --source "${MODEL_SOURCE:-huggingface}"
uv run ai-act-rag build-index
