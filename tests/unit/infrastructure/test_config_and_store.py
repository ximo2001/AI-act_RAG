"""Tests for settings, prompt loading and the FAISS index store."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.fakes import FakeLLM

from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.prompts import format_context, load_prompt
from ai_act_rag.application.translation import TranslatingRetriever
from ai_act_rag.domain.exceptions import IndexNotBuiltError
from ai_act_rag.domain.models import Chunk, RetrievedChunk
from ai_act_rag.infrastructure.config import Settings
from ai_act_rag.infrastructure.container import build_retriever, wrap_with_translation
from ai_act_rag.infrastructure.embeddings import _prefixes_for
from ai_act_rag.infrastructure.index_store import load_chunks, load_index, save_index


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIACT_TOP_K", "7")
    monkeypatch.setenv("AIACT_LLM_PROVIDER", "ollama")
    s = Settings.from_env(prompt_version="v1")
    assert (s.top_k, s.llm_provider, s.prompt_version) == (7, "ollama", "v1")
    assert "openai_api_key" not in repr(s)  # secrets never logged


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_versioned_prompts_load(version: str) -> None:
    p = load_prompt(version)
    assert p.system and "{question}" in p.user and "{context}" in p.user
    assert "QQ" in p.render_user("QQ", "CC") and "CC" in p.render_user("QQ", "CC")


def test_save_and_load_index_roundtrip(tmp_path: Path, mini_chunks: list[Chunk], embedder) -> None:  # type: ignore[no-untyped-def]
    vectors = embedder.embed_documents([c.text for c in mini_chunks])
    save_index(tmp_path, mini_chunks, vectors, {"embedding_model": embedder.name})
    chunks, index, meta = load_index(tmp_path)
    assert chunks == mini_chunks
    assert meta["embedding_model"] == "fake-hashing"
    ids, scores = index.search(vectors[2], k=1)
    assert ids == [2] and scores[0] == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(IndexNotBuiltError):
        load_chunks(tmp_path)


def test_format_context_uses_citable_ids(mini_chunks: list[Chunk]) -> None:
    ctx = format_context([RetrievedChunk(chunk=mini_chunks[0], score=1.0, rank=1)])
    assert ctx.startswith("[Recital 1] Recital 1\n")


def test_build_retriever_rejects_chunk_size_mismatch(
    tmp_path: Path,
    mini_chunks: list[Chunk],
    embedder,  # type: ignore[no-untyped-def]
) -> None:
    vectors = embedder.embed_documents([c.text for c in mini_chunks])
    meta = {"embedding_model": "models/fake-hashing", "chunk_words": 150}
    save_index(tmp_path, mini_chunks, vectors, meta)
    # Same final path component as the embedder name: accepted.
    settings = Settings(index_dir=tmp_path, chunk_words=150, reranker_model="")
    assert build_retriever(settings, embedder=embedder, mode="dense")
    with pytest.raises(ValueError, match="chunk_words"):
        build_retriever(replace(settings, chunk_words=400), embedder=embedder, mode="bm25")


def test_build_retriever_rejects_embedder_mismatch(
    tmp_path: Path,
    mini_chunks: list[Chunk],
    embedder,  # type: ignore[no-untyped-def]
) -> None:
    vectors = embedder.embed_documents([c.text for c in mini_chunks])
    meta = {"embedding_model": "models/bge-small-en-v1.5", "chunk_words": 150}
    save_index(tmp_path, mini_chunks, vectors, meta)
    settings = Settings(index_dir=tmp_path, chunk_words=150, reranker_model="")
    with pytest.raises(ValueError, match=r"bge-small-en-v1\.5.*fake-hashing"):
        build_retriever(settings, embedder=embedder, mode="dense")
    # BM25 does not use the embedder, so the mismatch does not matter there.
    assert build_retriever(settings, embedder=embedder, mode="bm25")


def test_wrap_with_translation_follows_settings(mini_chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(mini_chunks)
    on = wrap_with_translation(bm25, FakeLLM(), Settings(translate_queries=True))
    assert isinstance(on, TranslatingRetriever)
    assert wrap_with_translation(bm25, FakeLLM(), Settings(translate_queries=False)) is bm25


@pytest.mark.parametrize(
    ("model", "query_prefix"),
    [
        ("models/bge-small-en-v1.5", "Represent this sentence"),
        ("BAAI/bge-base-en-v1.5", "Represent this sentence"),
        ("intfloat/multilingual-e5-small", "query: "),
        ("models/e5-base-v2", "query: "),
        ("models/e5_experiments/all-MiniLM-L6-v2", ""),
    ],
)
def test_embedding_prefixes_use_the_final_path_component(model: str, query_prefix: str) -> None:
    assert _prefixes_for(model)[0].startswith(query_prefix)
    if not query_prefix:
        assert _prefixes_for(model) == ("", "")
