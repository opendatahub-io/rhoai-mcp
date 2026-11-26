"""Matrix coverage and native Anthropic wire behavior, without provider calls."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")
pytest.importorskip("anthropic", reason="requires the Anthropic skill-eval extra")

import httpx2  # noqa: E402
from evals.agent.models import get_spec, selected_specs  # noqa: E402
from pydantic_ai import Agent  # noqa: E402


def test_default_and_matrix_use_only_enmaas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RHOAI_EVAL_AGENT_MODELS", raising=False)
    matrix = selected_specs("all")
    assert [spec.id for spec in matrix] == [
        "enmaas-sol",
        "enmaas-opus",
        "enmaas-glm",
        "enmaas-deepseek",
    ]
    assert all(spec.api_key_env == "ENMAAS_KEY" for spec in matrix)
    assert all(httpx2.URL(spec.base_url).host == "api.enmaas.devshift.net" for spec in matrix)
    assert all("temperature" not in spec.settings() for spec in matrix)
    assert [spec.id for spec in selected_specs()] == ["enmaas-deepseek"]


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
        assert body["thinking"] == {"type": "adaptive"}
        assert "output_config" not in body  # Effort remains the provider default.
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


def test_spec_without_temperature_requests_none() -> None:
    """A spec that omits temperature must not request one (RHAI-45972 follow-up).

    The matrix invariant is "every spec leaves temperature to the provider" —
    the gpt-5.6 family rejects the parameter, and a cross-model difference in
    sampling configuration would bias comparisons. The dataclass default must
    enforce that, not rely on every future spec remembering to set None.
    """
    from evals.agent.models import AgentModelSpec

    spec = AgentModelSpec(
        id="probe",
        model_name="some-model",
        base_url=None,
        api_key_env="PROBE_KEY",
    )
    assert "temperature" not in spec.settings()
    assert spec.describe()["sampling"].startswith("provider-default")


def test_describe_reports_effective_temperature() -> None:
    """``describe()`` must report the temperature ``settings()`` sends (RHAI-45972).

    ``extra_settings`` forms the base of the effective settings, so a
    ``temperature`` there is live whenever the dedicated field is ``None``.
    The result record must not claim provider-default sampling in that case,
    and an explicit ``temperature`` must still win over ``extra_settings``.
    """
    from evals.agent.models import AgentModelSpec

    via_extra = AgentModelSpec(
        id="probe",
        model_name="some-model",
        base_url=None,
        api_key_env="PROBE_KEY",
        extra_settings={"temperature": 0.5},
    )
    assert via_extra.settings()["temperature"] == 0.5
    assert via_extra.describe()["temperature"] == 0.5
    assert via_extra.describe()["sampling"] == "temperature=0.5"

    explicit = AgentModelSpec(
        id="probe",
        model_name="some-model",
        base_url=None,
        api_key_env="PROBE_KEY",
        temperature=0.2,
        extra_settings={"temperature": 0.5},
    )
    assert explicit.settings()["temperature"] == 0.2
    assert explicit.describe()["temperature"] == 0.2
    assert explicit.describe()["sampling"] == "temperature=0.2"
