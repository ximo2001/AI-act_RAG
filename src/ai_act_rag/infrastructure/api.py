"""REST API (FastAPI): ``POST /ask``, ``GET /search``, ``GET /health``.

The pipeline (index, embedder, LLM) is built once at startup, in the FastAPI
lifespan. If that fails (index not built, model missing...) the server still
starts: ``/health`` reports ``ready: false`` and ``/ask`` / ``/search`` answer
503 with the reason instead of a 500. Tests override :func:`get_service`.

llama.cpp models are not thread-safe and FastAPI runs sync endpoints in a
thread pool, so every call into the pipeline is serialised with a lock. A
request that waits longer than :data:`LOCK_TIMEOUT_S` for it gets a 503
("busy") instead of piling up. ``/health`` is ``async`` so it runs on the event
loop and never waits for the thread pool.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ai_act_rag import __version__
from ai_act_rag.application.answering import AnswerService
from ai_act_rag.domain.exceptions import AIActRAGError
from ai_act_rag.domain.models import Confidence
from ai_act_rag.infrastructure.config import Settings
from ai_act_rag.infrastructure.container import build_answer_service

logger = logging.getLogger(__name__)

# One request at a time inside the pipeline: llama.cpp is not thread-safe.
_PIPELINE_LOCK = threading.Lock()
# Longest a request waits for the pipeline before answering 503 "busy" (room for
# a few CPU generations queued ahead of it).
LOCK_TIMEOUT_S = 120.0


@contextmanager
def _pipeline_lock() -> Iterator[None]:
    """Hold the pipeline lock, waiting at most :data:`LOCK_TIMEOUT_S`.

    Yields:
        Control while the lock is held.

    Raises:
        HTTPException: 503 if the lock is not acquired in time.
    """
    if not _PIPELINE_LOCK.acquire(timeout=LOCK_TIMEOUT_S):
        raise HTTPException(status_code=503, detail="Pipeline busy, try again later")
    try:
        yield
    finally:
        _PIPELINE_LOCK.release()


class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    question: str = Field(min_length=3, max_length=1000)


class SourceOut(BaseModel):
    """A context passage returned with the answer."""

    chunk_id: str
    unit_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    """Response of ``POST /ask`` (a DTO: the domain ``Answer`` is not exposed as is)."""

    answer: str
    cited_articles: list[str]
    confidence: Confidence
    abstained: bool
    parse_error: bool
    sources: list[SourceOut]
    latency_ms: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline at startup; keep the error if it cannot be built.

    On shutdown, the LLM client is closed if the provider exposes ``close()``
    (the HTTP providers hold an ``httpx.Client``).

    Args:
        app: The FastAPI application (its ``state`` holds the service).

    Yields:
        Control to the server while it runs.
    """
    app.state.service = None
    app.state.startup_error = None
    try:
        app.state.service = build_answer_service(Settings.from_env())
    except Exception as exc:  # any failure must become a 503, not a crash or a 500
        logger.exception("Could not build the RAG pipeline")
        app.state.startup_error = f"{type(exc).__name__}: {exc}"
    yield
    service = app.state.service
    close = getattr(service.llm, "close", None) if service is not None else None
    if callable(close):
        close()


def get_service(request: Request) -> AnswerService:
    """Return the pipeline built at startup (FastAPI dependency).

    Args:
        request: Current request (gives access to ``app.state``).

    Returns:
        The answer service.

    Raises:
        HTTPException: 503 if the pipeline could not be built.
    """
    service: AnswerService | None = getattr(request.app.state, "service", None)
    if service is None:
        reason = getattr(request.app.state, "startup_error", None) or "not initialised"
        raise HTTPException(status_code=503, detail=f"Pipeline not available: {reason}")
    return service


app = FastAPI(
    title="AI Act RAG",
    version=__version__,
    description="Questions about Regulation (EU) 2024/1689 answered with article citations.",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Liveness probe (answers even when the pipeline failed to load or is busy).

    Args:
        request: Current request (gives access to ``app.state``).

    Returns:
        Status, version and whether the pipeline is ready.
    """
    ready = getattr(request.app.state, "service", None) is not None
    return {"status": "ok", "version": __version__, "ready": ready}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, service: AnswerService = Depends(get_service)) -> AskResponse:  # noqa: B008
    """Answer a question about the AI Act.

    Args:
        req: Request body.
        service: Injected answer service.

    Returns:
        Answer, citations and the passages used.

    Raises:
        HTTPException: 503 if the pipeline is not available (index/model missing)
            or busy for longer than :data:`LOCK_TIMEOUT_S`.
    """
    try:
        with _pipeline_lock():
            ans = service.ask(req.question)
    except AIActRAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(
        answer=ans.answer,
        cited_articles=ans.cited_articles,
        confidence=ans.confidence,
        abstained=ans.abstained,
        parse_error=ans.parse_error,
        sources=[
            SourceOut(
                chunk_id=h.chunk.chunk_id, unit_id=h.chunk.unit_id, score=h.score, text=h.chunk.text
            )
            for h in ans.sources
        ],
        latency_ms=ans.latency_ms,
    )


@app.get("/search", response_model=list[SourceOut])
def search(
    q: Annotated[str, Query(min_length=3, max_length=1000)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    service: AnswerService = Depends(get_service),  # noqa: B008
) -> list[SourceOut]:
    """Retrieval only: the passages most relevant to ``q``, without generating an answer.

    No answer is generated, but the LLM may still be called once to translate a
    non-English query when ``translate_queries`` is on.

    Args:
        q: Query (3-1000 characters).
        k: Number of passages (1-20).
        service: Injected answer service (its retriever is used).

    Returns:
        Ranked passages.

    Raises:
        HTTPException: 503 if the pipeline is not available or busy.
    """
    try:
        with _pipeline_lock():
            hits = service.retriever.search(q, k=k)
    except AIActRAGError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        SourceOut(
            chunk_id=h.chunk.chunk_id, unit_id=h.chunk.unit_id, score=h.score, text=h.chunk.text
        )
        for h in hits
    ]
