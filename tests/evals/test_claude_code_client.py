"""Offline adapter checks; run with the optional claude-check dependency group."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from dataclasses import asdict
from typing import Any
from unittest.mock import AsyncMock

import pytest

sdk = pytest.importorskip("claude_agent_sdk")
code = importlib.import_module("evals.claude_code_client")


def assistant(*blocks: Any, **kwargs: Any) -> Any:
    return sdk.AssistantMessage(content=list(blocks), model="synthetic", **kwargs)


def use(call_id: str, name: str, **arguments: Any) -> Any:
    return sdk.ToolUseBlock(id=call_id, name=code.MCP_PREFIX + name, input=arguments)


def tool_result(call_id: str, content: Any, *, is_error: bool = False) -> Any:
    return sdk.ToolResultBlock(tool_use_id=call_id, content=content, is_error=is_error)


def terminal(**kwargs: Any) -> Any:
    fields = {
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 8,
        "is_error": False,
        "num_turns": 2,
        "session_id": "scenario-1",
        "result": "Complete.",
    }
    return sdk.ResultMessage(**(fields | kwargs))


class FakeSDKClient:
    def __init__(self) -> None:
        self.__aenter__ = AsyncMock(return_value=self)
        self.__aexit__ = AsyncMock(return_value=False)
        self.get_server_info = AsyncMock(
            return_value={"commands": [{"name": "navigator"}, {"name": "navigator-deploy"}]}
        )
        self.query = AsyncMock()
        self.responses: list[list[Any]] = []

    async def receive_response(self) -> Any:
        for message in self.responses.pop(0):
            yield message


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeSDKClient:
    instance = FakeSDKClient()
    monkeypatch.setattr(code, "ClaudeSDKClient", lambda **_kwargs: instance)
    return instance


async def test_pairs_parallel_calls_by_id_and_preserves_message_order(fake: FakeSDKClient) -> None:
    structured = [{"type": "text", "text": '{"users": 50}'}]
    fake.responses = [
        [
            sdk.SystemMessage(subtype="init", data={"slash_commands": ["navigator"]}),
            assistant(
                sdk.TextBlock(text="Checking workload."),
                use("defaults", "get_use_case_defaults", use_case="support_chatbot"),
                use("rps", "get_expected_rps", user_count=50),
            ),
            sdk.UserMessage(
                content=[
                    tool_result("rps", structured),
                    tool_result("defaults", "defaults result"),
                ]
            ),
            assistant(sdk.TextBlock(text="Confirm this workload.")),
            terminal(result="Confirm this workload."),
        ]
    ]

    async with code.ClaudeCodeClient() as client:
        result = await client.query("50 users", skill="navigator")

    assert result.task == "/navigator 50 users"
    assert result.tool_names_used == ["get_use_case_defaults", "get_expected_rps"]
    assert result.tool_calls[0].result == "defaults result"
    assert result.tool_calls[1].result == structured
    assert result.tool_calls[1].arguments == {"user_count": 50}
    assert result.final_output == "Confirm this workload."
    assert result.turns == 2
    assert result.conversation_id == "scenario-1"
    assert [message["role"] for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [
        message["tool_call_id"] for message in result.messages if message["role"] == "tool"
    ] == ["rps", "defaults"]
    calls = result.messages[1]["tool_calls"]
    assert calls[0]["function"]["name"] == "get_use_case_defaults"
    assert json.loads(calls[1]["function"]["arguments"]) == {"user_count": 50}
    assert "navigator" in client.available_commands
    assert client.initialization["slash_commands"] == ["navigator"]
    json.dumps(asdict(result))
    json.dumps(client.diagnostics)


async def test_follow_up_reuses_connection_and_returns_only_current_exchange(
    fake: FakeSDKClient,
) -> None:
    fake.responses = [
        [assistant(sdk.TextBlock(text="Confirm?")), terminal(result="Confirm?")],
        [
            sdk.UserMessage(content="yes"),
            assistant(use("recommend", "recommend_model", user_count=50)),
            sdk.UserMessage(content=[tool_result("recommend", "model-a")]),
            assistant(sdk.TextBlock(text="First paragraph.")),
            assistant(sdk.TextBlock(text="Second paragraph.")),
            terminal(result="Second paragraph."),
        ],
    ]
    async with code.ClaudeCodeClient() as client:
        first = await client.query("Help", skill="navigator")
        second = await client.follow_up("yes", first.conversation_id)
        assert client.last_result is second

    assert [call.args for call in fake.query.await_args_list] == [("/navigator Help",), ("yes",)]
    fake.__aenter__.assert_awaited_once()
    fake.__aexit__.assert_awaited_once()
    assert first.final_output == "Confirm?"
    assert first.tool_calls == []
    assert second.final_output == "First paragraph.\nSecond paragraph."
    assert second.tool_names_used == ["recommend_model"]
    assert [message for message in second.messages if message["role"] == "user"] == [
        {"role": "user", "content": "yes"}
    ]
    assert "Confirm?" not in json.dumps(second.messages)
    assert first.conversation_id == second.conversation_id


async def test_plain_query_uses_explicit_scenario_tool_allowlist(fake: FakeSDKClient) -> None:
    fake.get_server_info.return_value = {"commands": []}
    fake.responses = [[assistant(sdk.TextBlock(text="Cluster summary.")), terminal()]]
    async with code.ClaudeCodeClient(allowed_tools=["explore_cluster"]) as client:
        result = await client.query("Explore the cluster")
    fake.query.assert_awaited_once_with("Explore the cluster")
    assert result.task == "Explore the cluster"
    assert client.options.allowed_tools == ["Skill", code.MCP_PREFIX + "explore_cluster"]


def test_empty_scenario_tool_allowlist_does_not_enable_navigator_tools() -> None:
    assert code.ClaudeCodeClient(allowed_tools=[]).options.allowed_tools == ["Skill"]


async def test_builtin_and_other_server_tools_stay_in_diagnostics(fake: FakeSDKClient) -> None:
    fake.responses = [
        [
            assistant(
                sdk.ToolUseBlock(id="skill", name="Skill", input={"skill": "navigator"}),
                sdk.ToolUseBlock(id="other", name="mcp__other__recommend_model", input={}),
                use("mcp", "recommend_model"),
            ),
            sdk.UserMessage(
                content=[
                    tool_result("skill", "loaded"),
                    tool_result("other", "other result"),
                    tool_result("mcp", "model-a"),
                ]
            ),
            terminal(),
        ]
    ]
    async with code.ClaudeCodeClient() as client:
        result = await client.query("Help", skill="navigator")

    assert result.tool_names_used == ["recommend_model"]
    assert len(result.messages[1]["tool_calls"]) == 1
    assert [
        message["tool_call_id"] for message in result.messages if message["role"] == "tool"
    ] == ["mcp"]
    assert [entry["name"] for entry in client.diagnostics if entry["type"] == "tool_use"] == [
        "Skill",
        "mcp__other__recommend_model",
    ]
    assert len([entry for entry in client.diagnostics if entry["type"] == "tool_result"]) == 2
    assert result.final_output == "Complete."


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([assistant(sdk.TextBlock(text="partial"))], "without a terminal"),
        ([terminal(subtype="error_max_turns", is_error=True)], "error_max_turns"),
        ([terminal(subtype="error_during_execution", errors=["broken"])], "broken"),
        ([terminal(is_error=True, stop_reason="refusal", result=None)], "refusal"),
        ([terminal(errors=["API unavailable"])], "API unavailable"),
        ([terminal(permission_denials=[{"tool_name": "execute_deployment"}])], "permission denied"),
        ([terminal(terminal_reason="aborted_streaming")], "interrupted"),
        ([terminal(session_id="")], "no session ID"),
        ([terminal(result="")], "no final assistant"),
        ([assistant(use("missing", "recommend_model")), terminal()], "Missing tool results"),
        (
            [sdk.UserMessage(content=[tool_result("unknown", "result")]), terminal()],
            "unknown tool-use ID",
        ),
        (
            [assistant(use("same", "recommend_model"), use("same", "get_expected_rps"))],
            "Duplicate tool-use ID",
        ),
        (
            [
                assistant(use("same", "recommend_model")),
                sdk.UserMessage(content=[tool_result("same", "ok"), tool_result("same", "again")]),
            ],
            "Duplicate tool result",
        ),
        (
            [
                assistant(use("failed", "recommend_model")),
                sdk.UserMessage(
                    content=[tool_result("failed", "Permission denied", is_error=True)]
                ),
                terminal(),
            ],
            "Tool execution failed",
        ),
        (
            [assistant(sdk.TextBlock(text="partial"), error="authentication_failed")],
            "authentication_failed",
        ),
    ],
)
async def test_failures_raise_with_partial_trace(
    fake: FakeSDKClient, messages: list[Any], error: str
) -> None:
    fake.responses = [messages]
    async with code.ClaudeCodeClient() as client:
        with pytest.raises(code.ClaudeCodeError, match=error) as raised:
            await client.query("Help", skill="navigator")
        assert raised.value.result is client.last_result
        assert raised.value.result.messages[0]["content"] == "/navigator Help"
        with pytest.raises(RuntimeError, match="already started"):
            await client.query("retry", skill="navigator")
    fake.__aexit__.assert_awaited_once()


async def test_tool_error_retains_result_content_and_flag(fake: FakeSDKClient) -> None:
    content = [{"type": "text", "text": "cannot deploy"}]
    fake.responses = [
        [
            assistant(use("failed", "execute_deployment", namespace="production-models")),
            sdk.UserMessage(content=[tool_result("failed", content, is_error=True)]),
            terminal(),
        ]
    ]
    async with code.ClaudeCodeClient() as client:
        with pytest.raises(code.ClaudeCodeError) as raised:
            await client.query("Deploy", skill="navigator-deploy")
    result = raised.value.result
    assert result.tool_calls[0].is_error is True
    assert result.tool_calls[0].result == content
    assert result.messages[-1]["is_error"] is True


async def test_timeout_preserves_partial_trace_and_closes_connection(fake: FakeSDKClient) -> None:
    async def slow_response() -> Any:
        yield assistant(sdk.TextBlock(text="Working..."))
        await asyncio.Event().wait()

    fake.receive_response = slow_response
    async with code.ClaudeCodeClient(timeout=0.01) as client:
        with pytest.raises(code.ClaudeCodeError, match="timed out") as raised:
            await client.query("Help", skill="navigator")
        assert raised.value.result.final_output == "Working..."
    fake.__aexit__.assert_awaited_once()


async def test_sdk_exception_is_not_retried(fake: FakeSDKClient) -> None:
    fake.query.side_effect = RuntimeError("CLI process exited")
    async with code.ClaudeCodeClient() as client:
        with pytest.raises(code.ClaudeCodeError, match="CLI process exited"):
            await client.query("Help", skill="navigator")
    fake.query.assert_awaited_once()


async def test_wrong_conversation_and_second_query_are_rejected(fake: FakeSDKClient) -> None:
    fake.responses = [[terminal()]]
    async with code.ClaudeCodeClient() as client:
        with pytest.raises(ValueError, match="conversation_id"):
            await client.follow_up("yes", "foreign")
        first = await client.query("Help", skill="navigator")
        with pytest.raises(ValueError, match="conversation_id"):
            await client.follow_up("yes", first.conversation_id + "-other")
        with pytest.raises(RuntimeError, match="follow_up"):
            await client.query("Help again", skill="navigator")
    fake.query.assert_awaited_once()


async def test_changed_session_id_fails_and_cannot_continue(fake: FakeSDKClient) -> None:
    fake.responses = [[terminal()], [terminal(session_id="unexpected")]]
    async with code.ClaudeCodeClient() as client:
        first = await client.query("Help", skill="navigator")
        with pytest.raises(code.ClaudeCodeError, match="session ID changed"):
            await client.follow_up("yes", first.conversation_id)
        with pytest.raises(RuntimeError, match="Scenario failed"):
            await client.follow_up("retry", first.conversation_id)


@pytest.mark.parametrize("metadata", [{"commands": []}, {"slash_commands": ["other"]}, None])
async def test_missing_command_fails_before_prompt_is_sent(
    fake: FakeSDKClient, metadata: Any
) -> None:
    fake.get_server_info.return_value = metadata
    async with code.ClaudeCodeClient() as client:
        with pytest.raises(code.ClaudeCodeError, match="not discovered"):
            await client.query("Help", skill="navigator")
    fake.query.assert_not_awaited()


async def test_discovery_accepts_string_command_names(fake: FakeSDKClient) -> None:
    fake.get_server_info.return_value = {"slash_commands": ["navigator", "navigator-deploy"]}
    fake.responses = [[terminal()]]
    async with code.ClaudeCodeClient() as client:
        await client.query("Help", skill="navigator")


async def test_initialization_failure_closes_sdk(fake: FakeSDKClient) -> None:
    fake.get_server_info.side_effect = RuntimeError("initialization failed")
    with pytest.raises(RuntimeError, match="initialization failed"):
        async with code.ClaudeCodeClient():
            pytest.fail("must not enter")
    fake.__aexit__.assert_awaited_once()


def test_options_use_local_mock_server_and_project_discovery(
    tmp_path: Any, monkeypatch: Any
) -> None:
    monkeypatch.delenv("RHOAI_MCP_ENABLED_PLUGINS", raising=False)
    client = code.ClaudeCodeClient(repo_root=tmp_path)
    options = client.options
    assert options.cwd == tmp_path.resolve()
    assert options.setting_sources == ["project"]
    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {
        "rhoai-mcp": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "rhoai_mcp", "--transport", "stdio", "--mock-cluster"],
            "env": {
                "RHOAI_MCP_PLANNER_MODE": "local",
                "RHOAI_MCP_PLANNER_MODEL_CATALOG_URL": "",
                "PLANNER_DETECT_CLUSTER_GPUS": "false",
            },
        }
    }
    assert options.permission_mode == "dontAsk"
    assert "Skill" in options.allowed_tools
    assert code.MCP_PREFIX + "execute_deployment" in options.allowed_tools
    assert code.MCP_PREFIX + "create_serving_runtime" in options.allowed_tools
    assert options.model is None
    assert code.ClaudeCodeClient().repo_root == code.REPO_ROOT


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        code.ClaudeCodeClient(timeout=0)
