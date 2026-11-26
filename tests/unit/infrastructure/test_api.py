"""API tests with a fake pipeline (BM25 + scripted LLM): no index or model needed."""

import json

import pytest
from fastapi.testclient import TestClient
from tests.fakes import FakeLLM

from ai_act_rag.application.answering import AnswerService
from ai_act_rag.application.bm25 import BM25Retriever
from ai_act_rag.application.prompts import load_prompt
from ai_act_rag.domain.models import Chunk
from ai_act_rag.infrastructure import api
from ai_act_rag.infrastructure.api import app, get_service


@pytest.fixture
def client(mini_chunks: list[Chunk]) -> TestClient:
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "answer": "Annex III lists them.",
                    "cited_articles": ["Annex III"],
                    "confidence": "high",
                }
            )
        ]
    )
    service = AnswerService(BM25Retriever(mini_chunks), llm, load_prompt("v2"), top_k=3)
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_returns_answer_with_citations(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Which biometric systems are high-risk?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cited_articles"] == ["Annex III"]
    assert body["abstained"] is False
    assert body["sources"] and {"chunk_id", "unit_id", "score", "text"} <= set(body["sources"][0])


def test_ask_reports_parse_error(mini_chunks: list[Chunk]) -> None:
    service = AnswerService(
        BM25Retriever(mini_chunks), FakeLLM(['{"answer": "trunc']), load_prompt("v2"), top_k=3
    )
    app.dependency_overrides[get_service] = lambda: service
    try:
        body = TestClient(app).post("/ask", json={"question": "What is high-risk?"}).json()
    finally:
        app.dependency_overrides.clear()
    assert body["abstained"] is True and body["parse_error"] is True


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


@pytest.mark.parametrize(
    "params",
    [{"q": "ab"}, {"q": "x" * 1001}, {"q": "risk", "k": 0}, {"q": "risk", "k": 21}],
)
def test_search_validates_input(client: TestClient, params: dict[str, object]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_busy_pipeline_gives_503_and_health_still_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api, "LOCK_TIMEOUT_S", 0.01)
    with api._PIPELINE_LOCK:  # another request is using the pipeline
        resp = client.post("/ask", json={"question": "Which biometric systems are high-risk?"})
        assert client.get("/health").status_code == 200
        assert client.get("/search", params={"q": "risk"}).status_code == 503
    assert resp.status_code == 503
    assert "busy" in resp.json()["detail"]


def test_search_is_retrieval_only(client: TestClient) -> None:
    resp = client.get("/search", params={"q": "remote biometric identification", "k": 2})
    assert resp.status_code == 200
    assert resp.json()[0]["unit_id"] == "Annex III"


def test_pipeline_failure_at_startup_gives_503() -> None:
    app.state.service = None
    app.state.startup_error = "IndexNotBuiltError: no index"
    try:
        client = TestClient(app)
        assert client.get("/health").json()["ready"] is False
        resp = client.post("/ask", json={"question": "What is high-risk?"})
        assert resp.status_code == 503
        assert "no index" in resp.json()["detail"]
        assert client.get("/search", params={"q": "risk"}).status_code == 503
    finally:
        del app.state.service, app.state.startup_error


def test_lifespan_turns_build_errors_into_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(settings: object) -> None:
        raise ValueError("model missing")

    monkeypatch.setattr("ai_act_rag.infrastructure.api.build_answer_service", fail)
    with TestClient(app) as client:
        resp = client.get("/search", params={"q": "risk"})
    assert resp.status_code == 503
    assert "ValueError: model missing" in resp.json()["detail"]


def test_lifespan_closes_the_llm_client(
    monkeypatch: pytest.MonkeyPatch, mini_chunks: list[Chunk]
) -> None:
    class ClosingLLM(FakeLLM):
        closed = False

        def close(self) -> None:
            self.closed = True

    llm = ClosingLLM()
    service = AnswerService(BM25Retriever(mini_chunks), llm, load_prompt("v2"), top_k=3)
    monkeypatch.setattr("ai_act_rag.infrastructure.api.build_answer_service", lambda s: service)
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"] is True
        assert llm.closed is False
    assert llm.closed is True
