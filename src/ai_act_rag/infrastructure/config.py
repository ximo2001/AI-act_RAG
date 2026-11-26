"""Settings loaded from environment variables (prefix ``AIACT_``).

A frozen dataclass instead of a settings framework: every knob is visible in
one place and defaults run the full local pipeline. See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _env(name: str, default: Any) -> Any:
    """Read ``AIACT_<NAME>`` and cast it to the type of the default.

    Args:
        name: Field name (upper-cased and prefixed).
        default: Default value (its type drives the cast).

    Returns:
        The typed value.
    """
    raw = os.environ.get(f"AIACT_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        return Path(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    """All runtime configuration.

    Attributes:
        source_path: Downloaded AI Act XHTML.
        index_dir: Where chunks + FAISS index are stored.
        embedding_model: sentence-transformers model id or local folder.
        reranker_model: Cross-encoder id/folder; empty string disables reranking.
        chunk_words: Max words per chunk.
        retrieval_mode: ``dense``, ``bm25`` or ``hybrid``.
        top_k: Number of chunks given to the LLM.
        llm_provider: ``llama_cpp``, ``ollama`` or ``openai``.
        llm_model: GGUF path (llama_cpp) or model name (ollama/openai). Default
            Qwen2.5-3B-Instruct Q4_K_M; the 1.5B GGUF is the faster option
            (``DECISIONS.md`` D12).
        llm_ctx: Context window for llama.cpp.
        llm_threads: CPU threads for llama.cpp (0 = auto).
        max_tokens: Generation budget per answer (the JSON answer plus citations).
        ollama_url: Ollama server.
        openai_base_url: OpenAI-compatible endpoint (e.g. Groq).
        openai_api_key: API key, only from the environment (``AIACT_OPENAI_API_KEY``).
        prompt_version: Prompt file to use (``prompts/answer_<version>.txt``).
        translate_queries: Translate non-English questions to English (with the
            LLM) before retrieval; the answer is still in the question's language.
            On by default: the corpus and default embedder are English-only, and
            translation lifts Spanish Hit@5 substantially in ``eval/results``.
        prompts_dir: Folder with versioned prompts.
        seed: Global seed.
    """

    source_path: Path = Path("data/raw/ai_act_32024R1689_en.xhtml")
    index_dir: Path = Path("data/index")
    embedding_model: str = "models/bge-small-en-v1.5"
    reranker_model: str = "models/ms-marco-MiniLM-L-6-v2"
    chunk_words: int = 150
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    llm_provider: str = "llama_cpp"
    llm_model: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_ctx: int = 4096
    llm_threads: int = 0
    max_tokens: int = 384
    ollama_url: str = "http://localhost:11434"
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: str = field(default="", repr=False)
    prompt_version: str = "v3"
    translate_queries: bool = True
    prompts_dir: Path = Path("prompts")
    seed: int = 42

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from ``AIACT_*`` variables, then apply explicit overrides.

        Args:
            **overrides: Values that win over the environment (e.g. CLI flags).

        Returns:
            Settings instance.
        """
        values = {f.name: _env(f.name, f.default) for f in fields(cls)}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
