"""Composition root: builds concrete adapters from :class:`Settings`.

This is the only module that knows which implementation backs each port.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from ai_act_rag.application.answering import AnswerService
from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.chunking import ChunkingConfig, chunk_units
from ai_act_rag.application.prompts import load_prompt
from ai_act_rag.application.retrieval import DenseRetriever, HybridRetriever
from ai_act_rag.application.translation import TranslatingRetriever
from ai_act_rag.domain.models import Chunk
from ai_act_rag.domain.ports import LLM, Embedder, Reranker, Retriever
from ai_act_rag.infrastructure.config import Settings
from ai_act_rag.infrastructure.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from ai_act_rag.infrastructure.eurlex import parse_ai_act, sha256
from ai_act_rag.infrastructure.index_store import FaissIndex, load_index, save_index
from ai_act_rag.infrastructure.llm.llama_cpp_llm import LlamaCppLLM
from ai_act_rag.infrastructure.llm.ollama_llm import OllamaLLM
from ai_act_rag.infrastructure.llm.openai_compat_llm import OpenAICompatLLM

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Fix Python/NumPy (and torch, if installed) seeds for reproducibility.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def build_embedder(settings: Settings) -> Embedder:
    """Create the sentence-transformers embedder.

    Args:
        settings: Settings.

    Returns:
        Embedder.
    """
    return SentenceTransformerEmbedder(settings.embedding_model)


def build_reranker(settings: Settings) -> Reranker | None:
    """Create the optional cross-encoder reranker.

    Args:
        settings: Settings.

    Returns:
        Reranker, or None when ``reranker_model`` is empty.
    """
    if not settings.reranker_model:
        return None
    return CrossEncoderReranker(settings.reranker_model)


def build_index(settings: Settings, embedder: Embedder | None = None) -> list[Chunk]:
    """Parse, chunk, embed and persist the corpus.

    Args:
        settings: Settings (source path, chunk size, index dir...).
        embedder: Optional pre-built embedder (tests inject a fake one).

    Returns:
        The indexed chunks.
    """
    set_seeds(settings.seed)
    units = parse_ai_act(settings.source_path.read_bytes())
    chunks = chunk_units(units, ChunkingConfig(max_words=settings.chunk_words))
    embedder = embedder or build_embedder(settings)
    vectors = embedder.embed_documents([c.text for c in chunks])
    save_index(
        settings.index_dir,
        chunks,
        vectors,
        {
            "embedding_model": embedder.name,
            "chunk_words": settings.chunk_words,
            "n_chunks": len(chunks),
            "source_sha256": sha256(settings.source_path),
        },
    )
    logger.info("Indexed %d chunks into %s", len(chunks), settings.index_dir)
    return chunks


def build_retriever(
    settings: Settings,
    embedder: Embedder | None = None,
    mode: str | None = None,
    reranker: Reranker | None = None,
) -> Retriever:
    """Load the index and build the configured retriever.

    Args:
        settings: Settings.
        embedder: Optional pre-built embedder (reused across eval runs).
        mode: Overrides ``settings.retrieval_mode`` (dense | bm25 | hybrid).
        reranker: Optional pre-built reranker (otherwise from settings).

    Returns:
        A retriever.

    Raises:
        ValueError: On an unknown mode, or when the index was built with another
            chunk size or embedder than the current settings.
    """
    mode = mode or settings.retrieval_mode
    chunks, index, meta = load_index(settings.index_dir)
    if meta.get("chunk_words") != settings.chunk_words:
        raise ValueError(
            f"Index built with chunk_words={meta.get('chunk_words')!r}, settings say "
            f"{settings.chunk_words}: rebuild the index or set AIACT_CHUNK_WORDS"
        )
    bm25 = BM25Retriever(chunks)
    if mode == "bm25":
        return bm25
    embedder = embedder or build_embedder(settings)
    # Compare the final path component, so "models/bge-small-en-v1.5" and
    # "BAAI/bge-small-en-v1.5" (same weights, local copy vs hub id) match.
    if Path(meta.get("embedding_model", "")).name != Path(embedder.name).name:
        raise ValueError(
            f"Index built with {meta.get('embedding_model')!r}, query embedder is "
            f"{embedder.name!r}: rebuild the index"
        )
    dense = DenseRetriever(chunks, embedder, index)
    if mode == "dense":
        return dense
    if mode == "hybrid":
        return HybridRetriever([dense, bm25], reranker=reranker or build_reranker(settings))
    raise ValueError(f"Unknown retrieval mode {mode!r}")


def build_llm(settings: Settings) -> LLM:
    """Create the configured LLM provider.

    Args:
        settings: Settings.

    Returns:
        LLM adapter.

    Raises:
        ValueError: On an unknown provider or missing API key.
    """
    provider = settings.llm_provider
    if provider == "llama_cpp":
        return LlamaCppLLM(
            settings.llm_model,
            n_ctx=settings.llm_ctx,
            n_threads=settings.llm_threads or None,
            seed=settings.seed,
        )
    if provider == "ollama":
        return OllamaLLM(settings.llm_model, base_url=settings.ollama_url, seed=settings.seed)
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("Set AIACT_OPENAI_API_KEY to use the openai provider")
        return OpenAICompatLLM(
            settings.llm_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            seed=settings.seed,
        )
    raise ValueError(f"Unknown LLM provider {provider!r}")


def wrap_with_translation(retriever: Retriever, llm: LLM, settings: Settings) -> Retriever:
    """Wrap a retriever with query translation when ``settings.translate_queries`` is on.

    Args:
        retriever: Retriever to wrap.
        llm: LLM used to translate non-English queries.
        settings: Settings (``translate_queries``, ``prompts_dir``).

    Returns:
        A :class:`TranslatingRetriever` over ``retriever``, or ``retriever`` as is.
    """
    if not settings.translate_queries:
        return retriever
    return TranslatingRetriever(
        retriever, llm, load_prompt("v1", Path(settings.prompts_dir), "translate")
    )


def build_answer_service(
    settings: Settings, llm: LLM | None = None, retriever: Retriever | None = None
) -> AnswerService:
    """Build the full RAG pipeline.

    Args:
        settings: Settings.
        llm: Optional pre-built LLM (tests, eval reuse).
        retriever: Optional pre-built retriever.

    Returns:
        Ready-to-use :class:`AnswerService`.
    """
    set_seeds(settings.seed)
    llm = llm or build_llm(settings)
    return AnswerService(
        retriever=wrap_with_translation(retriever or build_retriever(settings), llm, settings),
        llm=llm,
        prompt=load_prompt(settings.prompt_version, Path(settings.prompts_dir)),
        top_k=settings.top_k,
        max_tokens=settings.max_tokens,
    )


__all__ = [
    "FaissIndex",
    "build_answer_service",
    "build_index",
    "build_llm",
    "build_retriever",
    "set_seeds",
    "wrap_with_translation",
]
