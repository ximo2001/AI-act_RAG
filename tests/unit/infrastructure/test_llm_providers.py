"""HTTP LLM adapters tested against an in-process mock transport (no network)."""

import json

import httpx
import pytest

from ai_act_rag.domain.exceptions import ProviderUnavailableError
from ai_act_rag.infrastructure.llm.ollama_llm import OllamaLLM
from ai_act_rag.infrastructure.llm.openai_compat_llm import OpenAICompatLLM

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


def _client(handler):  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_sends_schema_as_format_and_reads_content() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"message": {"content": '{"answer": "ok"}'}})

    llm = OllamaLLM("qwen2.5:3b", client=_client(handler))
    out = llm.generate("sys", "user", json_schema=SCHEMA, max_tokens=10)
    assert out == '{"answer": "ok"}'
    assert seen["url"].endswith("/api/chat")
    assert seen["format"] == SCHEMA
    assert seen["options"]["num_predict"] == 10 and seen["options"]["temperature"] == 0.0
    assert llm.name == "ollama:qwen2.5:3b"


def test_openai_compat_uses_bearer_and_json_mode() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm = OpenAICompatLLM("m", "https://example.test/v1", "secret", client=_client(handler))
    assert llm.generate("s", "u", json_schema=SCHEMA) == "{}"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_http_errors_become_provider_errors() -> None:
    llm = OllamaLLM("m", client=_client(lambda r: httpx.Response(500)))
    with pytest.raises(ProviderUnavailableError):
        llm.generate("s", "u")


def test_http_providers_close_only_their_own_client() -> None:
    injected = _client(lambda r: httpx.Response(200))
    OllamaLLM("m", client=injected).close()
    assert not injected.is_closed
    for llm in (OllamaLLM("m"), OpenAICompatLLM("m", "https://example.test/v1", "k")):
        llm.close()
        assert llm._client.is_closed
