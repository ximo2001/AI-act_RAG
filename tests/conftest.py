"""Shared pytest fixtures: tiny corpus, fake embedder and fake LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_act_rag.application.chunking import ChunkingConfig, chunk_units
from ai_act_rag.domain.models import Chunk, LegalUnit
from ai_act_rag.infrastructure.eurlex import parse_ai_act
from tests.fakes import HashingEmbedder

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_units() -> list[LegalUnit]:
    """Parsed units of the tiny fixture document."""
    return parse_ai_act((FIXTURES / "mini_act.xhtml").read_bytes(), expected_counts=None)


@pytest.fixture
def mini_chunks(mini_units: list[LegalUnit]) -> list[Chunk]:
    """Chunks of the tiny fixture document."""
    return chunk_units(mini_units, ChunkingConfig(max_words=250))


@pytest.fixture
def embedder() -> HashingEmbedder:
    """Fake deterministic embedder."""
    return HashingEmbedder()
