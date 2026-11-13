"""Offline runner contracts using a local model and an in-memory tool session.

Covers: checkpoints, partial history, failure handling, and session persistence.

The agent is built without providers or subprocesses. Pydantic AI still runs
the conversation, captures partial messages, and enforces usage limits, so the
failure tests exercise the same history-capture path as a live evaluation.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")

from evals.agent import runner  # noqa: E402
from evals.agent.models import AgentModelSpec, get_spec  # noqa: E402
from evals.agent.transcript import (  # noqa: E402
    AgentTranscript,
    AttemptArtifact,
    from_pydantic_ai,
)
from mcp.types import Tool  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel  # noqa: E402
from pydantic_ai.toolsets import FunctionToolset  # noqa: E402

SPEC = get_spec("enmaas-glm")


def user_prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def echo(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=f"reply: {user_prompts(messages)[-1]}")])


class MemorySession(FunctionToolset[None]):
    """Stand in for an MCP session, including state lost on reconnection."""

    def __init__(self) -> None:
        self.entries = 0
        self.exits = 0
        self.active = False
        self.depth = 0
        self.counter = 0
        self.schema_reads = 0
        super().__init__(tools=[self.read_state, self.wait_for_result])

    async def __aenter__(self) -> MemorySession:
        # Like MCPToolset, nested agent runs reuse the outer connection.
        if self.depth == 0:
            self.active = True
            self.entries += 1
            self.counter = 0
        self.depth += 1
        return self

    async def __aexit__(self, *_args: Any) -> None:
        self.depth -= 1
        if self.depth == 0:
            self.active = False
            self.exits += 1

    async def list_tools(self) -> list[Tool]:
        assert self.active, "schemas must come from the open conversation session"
        self.schema_reads += 1
        return [
            Tool(name=name, inputSchema={"type": "object", "properties": {}})
            for name in ("read_state", "wait_for_result")
        ]

    def read_state(self) -> int:
        assert self.active
        self.counter += 1
        return self.counter

    async def wait_for_result(self) -> str:
        assert self.active
        await asyncio.Future()
        raise AssertionError("the pending tool must be cancelled by the turn timeout")


@pytest.fixture
def install_agent(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[MemorySession, Mock]]:
    def install(function: FunctionDef = echo) -> tuple[MemorySession, Mock]:
        session = MemorySession()
        agent = Agent(FunctionModel(function), toolsets=[session])
        build = Mock(return_value=(agent, session))
        monkeypatch.setattr(runner, "build_agent", build)
        return session, build

    return install


@pytest.mark.parametrize("spec_id", ["enmaas-sol", "enmaas-opus", "enmaas-glm", "enmaas-deepseek"])
async def test_build_agent_delivers_shared_discovery_policy_without_preloading_a_skill(
    monkeypatch: pytest.MonkeyPatch, spec_id: str
) -> None:
    instructions = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions.append(info.instructions or "")
        return echo(messages, info)

    session = MemorySession()
    monkeypatch.setattr(AgentModelSpec, "build", lambda _self: FunctionModel(respond))
    monkeypatch.setattr("pydantic_ai.mcp.MCPToolset", lambda *_args, **_kwargs: session)
    agent, toolset = runner.build_agent(get_spec(spec_id))
    async with agent:
        result = await agent.run("Help me choose an LLM.")

    assert toolset is session
    assert "before responding or calling domain tools" in instructions[0]
    assert "confirmation gates" in instructions[0]
    transcript = from_pydantic_ai(result.all_messages())
    assert "navigator:" in transcript.skill_catalog
    assert "navigator-deploy:" in transcript.skill_catalog
    assert transcript.skills_loaded == []  # The model must still choose and load a skill.


@pytest.mark.parametrize("mock_cluster", [True, False])
def test_cluster_probe_override_is_scoped_to_the_mock_server(
    monkeypatch: pytest.MonkeyPatch, mock_cluster: bool
) -> None:
    monkeypatch.setenv("PLANNER_DETECT_CLUSTER_GPUS", "true")
    monkeypatch.setenv("RHOAI_MCP_MOCK_CLUSTER", "true")
    monkeypatch.setattr(AgentModelSpec, "build", lambda _self: FunctionModel(echo))
    _agent, toolset = runner.build_agent(SPEC, mock_cluster=mock_cluster)
    transport = toolset.client.transport

    assert ("--mock-cluster" in transport.args) is mock_cluster
    assert transport.env == (
        {"PLANNER_DETECT_CLUSTER_GPUS": "false"}
        if mock_cluster
        else {"RHOAI_MCP_MOCK_CLUSTER": "false"}
    )
    assert os.environ["PLANNER_DETECT_CLUSTER_GPUS"] == "true"


async def test_checkpoints_run_in_order_before_replies_with_cumulative_history(
    install_agent: Callable[..., Any],
) -> None:
    events = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        events.append(f"run:{user_prompts(messages)[-1]}")
        return echo(messages, info)

    session, build = install_agent(respond)
    opening_checkpoint = Mock(side_effect=AssertionError("the opening prompt is ungated"))

    def checkpoint(name: str, expected: list[str]) -> runner.Checkpoint:
        def inspect(transcript: AgentTranscript) -> None:
            events.append(f"checkpoint:{name}")
            assert [t.content for t in transcript.turns if t.role == "user"] == expected
            assert transcript.final_output == f"reply: {expected[-1]}"
            return None

        return inspect

    result = await runner.run_conversation(
        SPEC,
        [
            runner.ScriptedTurn("opening", checkpoint=opening_checkpoint),
            runner.ScriptedTurn("confirm", checkpoint=checkpoint("confirm", ["opening"])),
            runner.ScriptedTurn(
                "approve", checkpoint=checkpoint("approve", ["opening", "confirm"])
            ),
        ],
        run_id="run-1",
        attempt_id="attempt-1",
    )

    assert events == [
        "run:opening",
        "checkpoint:confirm",
        "run:confirm",
        "checkpoint:approve",
        "run:approve",
    ]
    opening_checkpoint.assert_not_called()
    build.assert_called_once()
    assert user_prompts(result.messages) == ["opening", "confirm", "approve"]
    assert len([m for m in result.messages if isinstance(m, ModelResponse)]) == 3
    assert result.turns_completed == 3
    assert result.final_output == "reply: approve"
    assert result.outcome == runner.OUTCOME_EVALUATED
    assert result.error is None
    assert session.entries == session.exits == 1


@pytest.mark.parametrize("name,stage", [("approval", "checkpoint:approval"), ("", "checkpoint:1")])
async def test_failed_checkpoint_stops_before_sending_approval_and_saves_history(
    install_agent: Callable[..., Any], tmp_path: Path, name: str, stage: str
) -> None:
    session, _ = install_agent()
    later_checkpoint = Mock(side_effect=AssertionError("the script must have stopped"))
    on_turn = AsyncMock()

    result = await runner.run_conversation(
        SPEC,
        [
            "opening",
            runner.ScriptedTurn("Yes, deploy.", checkpoint=lambda _: "plan missing", name=name),
            runner.ScriptedTurn("later", checkpoint=later_checkpoint),
        ],
        run_id="run-1",
        on_turn=on_turn,
    )

    assert result.outcome == runner.OUTCOME_EVALUATED
    assert result.error_stage == stage
    assert result.error == "plan missing"
    assert result.turns_completed == 1
    assert result.final_output == "reply: opening"
    assert user_prompts(result.messages) == ["opening"]
    later_checkpoint.assert_not_called()
    on_turn.assert_awaited_once()
    assert session.entries == session.exits == 1

    restored = AttemptArtifact.load(result.artifact().save(tmp_path))
    assert restored.metadata["error_stage"] == stage
    assert restored.metadata["error"] == "plan missing"
    assert restored.transcript().to_dict() == result.transcript().to_dict()


@pytest.mark.parametrize(
    "failure,limits,stage",
    [
        ("timeout", {"turn_timeout_s": 0.5}, "turn_timeout:1"),
        ("loop", {"request_limit": 1}, "usage_limit:1"),
        ("loop", {"tool_calls_limit": 1}, "usage_limit:1"),
        ("error", {}, "model_error:1"),
    ],
)
async def test_model_failure_preserves_saveable_partial_history_and_closes_session(
    install_agent: Callable[..., Any],
    tmp_path: Path,
    failure: str,
    limits: dict[str, Any],
    stage: str,
) -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if user_prompts(messages)[-1] == "opening":
            return echo(messages, info)
        if failure == "error":
            raise RuntimeError("model unavailable")
        tool = "wait_for_result" if failure == "timeout" else "read_state"
        return ModelResponse(parts=[ToolCallPart(tool_name=tool, args={})])

    session, _ = install_agent(respond)
    on_turn = AsyncMock()
    result = await runner.run_conversation(
        SPEC, ["opening", "continue", "must not run"], run_id="run-1", on_turn=on_turn, **limits
    )

    assert result.outcome == runner.OUTCOME_MODEL_FAILURE
    assert result.error_stage == stage
    assert result.error
    assert result.turns_completed == 1
    assert result.final_output == "reply: opening"
    assert user_prompts(result.messages) == ["opening", "continue"]
    on_turn.assert_awaited_once()
    assert session.entries == session.exits == 1
    assert not session.active
    if failure != "error":
        assert result.transcript().mcp_tool_calls

    restored = AttemptArtifact.load(result.artifact().save(tmp_path))
    assert restored.metadata["outcome"] == runner.OUTCOME_MODEL_FAILURE
    assert restored.metadata["error_stage"] == stage
    assert restored.metadata["turns_completed"] == 1
    assert restored.transcript().to_dict() == result.transcript().to_dict()
    assert [t.name for t in restored.mcp_tools()] == ["read_state", "wait_for_result"]


async def test_same_tool_session_keeps_state_across_turns(
    install_agent: Callable[..., Any],
) -> None:
    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]
        if isinstance(last_part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content=f"counter: {last_part.content}")])
        return ModelResponse(parts=[ToolCallPart(tool_name="read_state", args={})])

    session, _ = install_agent(respond)
    observed = []

    async def on_turn(index: int, transcript: AgentTranscript) -> None:
        assert session.active
        observed.append((index, transcript.assistant_text))

    result = await runner.run_conversation(
        SPEC, ["first", "second"], run_id="run-1", on_turn=on_turn
    )

    assert result.error is None
    assert result.turns_completed == 2
    assert result.final_output == "counter: 2"
    assert "counter: 1" in observed[0][1]
    assert "counter: 2" in observed[1][1]
    assert [index for index, _ in observed] == [0, 1]
    assert session.entries == session.exits == session.schema_reads == 1


@pytest.mark.parametrize("captured_length", [0, 1, 2, 3])
def test_extending_failed_run_history_does_not_duplicate_or_discard_turns(
    captured_length: int,
) -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart(content="opening")]),
        ModelResponse(parts=[TextPart(content="reply")]),
        ModelRequest(parts=[UserPromptPart(content="continue")]),
    ]
    merged = runner._extend(messages[:2], messages[:captured_length])

    assert merged == (messages if captured_length == 3 else messages[:2])
    assert user_prompts(merged) == (
        ["opening", "continue"] if captured_length == 3 else ["opening"]
    )


async def test_agent_build_failure_returns_saveable_harness_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "build_agent", Mock(side_effect=RuntimeError("setup failed")))

    result = await runner.run_conversation(SPEC, ["opening"], run_id="run-1")

    assert result.outcome == runner.OUTCOME_HARNESS_ERROR
    assert result.error_stage == "agent_build"
    assert result.error == "RuntimeError: setup failed"
    assert result.turns_completed == 0
    assert result.messages == []
    restored = AttemptArtifact.load(result.artifact().save(tmp_path))
    assert restored.metadata["error_stage"] == "agent_build"


async def test_schema_failure_closes_session_without_starting_conversation(
    install_agent: Callable[..., Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = Mock(side_effect=echo)
    session, _ = install_agent(model)
    monkeypatch.setattr(session, "list_tools", AsyncMock(side_effect=RuntimeError("schema failed")))

    result = await runner.run_conversation(SPEC, ["opening"], run_id="run-1")

    assert result.outcome == runner.OUTCOME_HARNESS_ERROR
    assert result.error_stage == "tool_schema_fetch"
    assert result.error == "RuntimeError: schema failed"
    assert result.messages == []
    assert result.turns_completed == 0
    model.assert_not_called()
    assert session.entries == session.exits == 1
    restored = AttemptArtifact.load(result.artifact().save(tmp_path))
    assert restored.metadata["error_stage"] == "tool_schema_fetch"
