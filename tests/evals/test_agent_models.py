"""Matrix coverage and native Anthropic wire behavior, without provider calls."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")
pytest.importorskip("anthropic", reason="requires the Anthropic skill-eval extra")

import httpx2  # noqa: E402
from evals.agent.models import get_spec, selected_specs  # noqa: E402
from evals.agent.probe_tool_calling import (  # noqa: E402
    Candidate,
    ProbeResult,
    _MessagesDriver,
    answer_references_result,
)
from pydantic_ai import Agent  # noqa: E402


def test_default_and_matrix_use_only_enmaas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RHOAI_EVAL_AGENT_MODELS", raising=False)
    assert [spec.id for spec in selected_specs()] == ["enmaas-sol"]
    matrix = selected_specs("matrix")
    assert [spec.id for spec in matrix] == [
        "enmaas-sol",
        "enmaas-opus",
        "enmaas-glm",
        "enmaas-deepseek",
    ]
    assert all(spec.api_key_env == "ENMAAS_KEY" for spec in matrix)
    assert all(httpx2.URL(spec.base_url).host == "api.enmaas.devshift.net" for spec in matrix)
    assert all("temperature" not in spec.settings() for spec in matrix)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Use meta-llama/Llama-3.1-8B-Instruct.", True),
        ("Use Llama 3.1 8B.", True),
        ("Use mistralai/Mistral-7B-Instruct-v0.3.", True),
        ("Use Mistral 7B for its lower latency.", True),
        ("Use Llama 3.1 70B.", False),
        ("Use Qwen 3.1 8B.", False),
        ("The GPUs are available.", False),
    ],
)
def test_probe_answer_must_reference_a_returned_model(answer: str, expected: bool) -> None:
    assert answer_references_result(answer) is expected


@pytest.mark.asyncio
async def test_opus_uses_native_messages_and_replays_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENMAAS_KEY", "test-enmaas-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-direct-provider-key")
    requests: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.host == "api.enmaas.devshift.net"
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-enmaas-key"
        body = json.loads(request.content)
        assert body["model"] == "claude-opus-4-6"
        assert "temperature" not in body
        requests.append(body)
        if len(requests) == 1:
            content = [{"type": "tool_use", "id": "lookup-1", "name": "lookup", "input": {}}]
            stop_reason = "tool_use"
        else:
            result = body["messages"][-1]["content"][0]
            assert result["type"] == "tool_result"
            assert result["tool_use_id"] == "lookup-1"
            assert "A100-80" in json.dumps(result["content"])
            content = [{"type": "text", "text": "Use the A100-80."}]
            stop_reason = "end_turn"
        return httpx2.Response(
            200,
            json={
                "id": f"msg-{len(requests)}",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        monkeypatch.setattr(
            "pydantic_ai.providers.anthropic.create_async_httpx2_client", lambda: client
        )
        agent = Agent(get_spec("enmaas-opus").build())

        @agent.tool_plain
        def lookup() -> str:
            return "A100-80"

        result = await agent.run("Find the available GPU.", model_settings={"max_tokens": 1024})

    assert result.output == "Use the A100-80."
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_probe_omits_null_sdk_fields_when_replaying_messages() -> None:
    from anthropic import AsyncAnthropic

    requests: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 2:
            block = body["messages"][1]["content"][0]
            assert block == {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}}
            assert body["messages"][2]["content"][0]["tool_use_id"] == "call-1"
        return httpx2.Response(
            200,
            json={
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [{"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}}],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )

    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http,
        AsyncAnthropic(api_key="test-key", http_client=http) as client,
    ):
        candidate = Candidate("test", "claude-opus-4-6", None, "TEST_KEY", "messages")
        driver = _MessagesDriver(client, candidate, ProbeResult(candidate))
        calls, _ = await driver.send()
        driver.add_tool_result(calls[0], '{"gpu": "A100-80"}')
        await driver.send()

    assert len(requests) == 2
