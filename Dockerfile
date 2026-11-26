# CPU image for the REST API. Data and models are NOT baked in: mount them.
#   docker build -t ai-act-rag .
#   docker run -p 8000:8000 -v $PWD/data:/app/data -v $PWD/models:/app/models ai-act-rag

# --- build stage: compile llama-cpp-python and resolve the locked environment ---
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
# GGML_NATIVE=OFF: compile llama.cpp for a portable x86-64 baseline (AVX2/FMA) instead of
# -march=native, so an image built on one machine does not crash with SIGILL on another.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    CMAKE_ARGS="-DGGML_NATIVE=OFF"
WORKDIR /app
# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra ml --extra llm --extra serve
COPY src ./src
RUN uv sync --frozen --no-dev --extra ml --extra llm --extra serve

# --- runtime stage ---
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
COPY prompts ./prompts
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "ai_act_rag.infrastructure.api:app", "--host", "0.0.0.0", "--port", "8000"]
