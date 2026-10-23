"""Offline backend routing and event-loop lifecycle checks using actual pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from evals.config import EvalConfig
from evals.lcs_client import LCSResult
from evals.pytest_client import EvalClient
from evals.result_types import ClaudeCodeResult, ClaudeToolCall, combine_results

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]

FAKE_FIXTURES = """
import asyncio
import sys
import types
import pytest
from mcp.types import Tool
from evals import deepeval_helpers as helpers
from evals.config import EvalConfig
from evals.result_types import ClaudeCodeResult
from evals.reporting import recorder

pytest_plugins = ["evals.conftest"]
opened = []
counts = {"http": 0, "stdio": 0, "lcs": 0}

class FakeClaude:
    def __init__(self, *, timeout, allowed_tools):
        assert allowed_tools == ["explore_cluster"]
        self.initialization = {"model": "synthetic-claude"}
        self.closed = False
        self.session_id = "claude-" + str(len(opened))

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        opened.append(self)
        return self

    async def __aexit__(self, *_args):
        assert self.loop is asyncio.get_running_loop()
        self.closed = True

    async def query(self, task, *, skill=None):
        assert self.loop is asyncio.get_running_loop()
        return ClaudeCodeResult(task, "first", conversation_id=self.session_id)

    async def follow_up(self, task, conversation_id):
        assert self.loop is asyncio.get_running_loop()
        assert conversation_id == self.session_id
        return ClaudeCodeResult(task, "second", conversation_id=self.session_id)

class FakeLCS:
    async def query(self, task):
        return ClaudeCodeResult(task, "first", conversation_id="lcs-session")

    async def follow_up(self, task, conversation_id):
        return ClaudeCodeResult(task, "second", conversation_id=conversation_id)

@pytest.fixture(scope="session")
def eval_config():
    return EvalConfig(cluster_mode="mock", _env_file=None)

@pytest.fixture(scope="session")
def lcs_client(pytestconfig):
    assert pytestconfig.getoption("--client") == "lcs", "LCS requested in Claude mode"
    counts["lcs"] += 1
    return FakeLCS()

@pytest.fixture(scope="session", autouse=True)
def patch_backends(pytestconfig):
    backend = pytestconfig.getoption("--client")
    async def http(*_args):
        assert backend == "lcs", "HTTP discovery requested in Claude mode"
        counts["http"] += 1
        return [Tool(name="explore_cluster", inputSchema={})]
    async def stdio():
        assert backend == "claude-code", "stdio discovery requested in LCS mode"
        counts["stdio"] += 1
        return [Tool(name="explore_cluster", inputSchema={})]
    def evaluate(**kwargs):
        record = kwargs["recorder"]
        assert record.agent_backend == backend
        assert record.agent_model == ("synthetic-claude" if backend == "claude-code" else "")
        assert record.phase == ("skills" if backend == "claude-code" else None)
        assert record.path == pytestconfig.getoption("--eval-results")
        return "recorded"
    fake = types.ModuleType("evals.claude_code_client")
    fake.ClaudeCodeClient = FakeClaude
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "evals.claude_code_client", fake)
        patch.setattr(helpers, "fetch_tool_schemas", http)
        patch.setattr(helpers, "fetch_stdio_tool_schemas", stdio)
        patch.setattr(recorder, "evaluate_and_record", evaluate)
        yield
    assert all(client.closed for client in opened)
    if backend == "claude-code":
        assert len(opened) == 2 and len({client.session_id for client in opened}) == 2
        assert counts == {"http": 0, "stdio": 1, "lcs": 0}
    else:
        assert not opened
        assert counts == {"http": 1, "stdio": 0, "lcs": 1}
"""

FAKE_TESTS = """
import asyncio
import pytest

@pytest.mark.parametrize("case", [1, 2])
@pytest.mark.asyncio
async def test_scenario(case, agent_client, evaluate_and_record):
    first = await agent_client.query("First", skill="navigator")
    second = await agent_client.follow_up("Second", first.conversation_id)
    assert first.conversation_id == second.conversation_id
    assert second.final_output == "second"
    if agent_client.backend == "claude-code":
        assert agent_client.client.loop is asyncio.get_running_loop()
        assert not agent_client.client.closed
    assert evaluate_and_record(scenario="synthetic", lcs_result=second, test_cases=[], metrics=[]) == "recorded"
"""


@pytest.mark.parametrize("backend", ["lcs", "claude-code"])
def test_conditional_fixtures_and_same_loop_lifecycle(
    pytester: Any, monkeypatch: Any, backend: str
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    pytester.makeini("[pytest]\nasyncio_mode=auto\n")
    pytester.makeconftest(FAKE_FIXTURES)
    pytester.makepyfile(FAKE_TESTS)
    args = [] if backend == "lcs" else ["--client=claude-code"]
    result = pytester.runpytest_subprocess(
        *args, "--eval-results", str(pytester.path / "run.jsonl"), "-q"
    )
    result.assert_outcomes(passed=2)


def test_live_cluster_setting_is_rejected_for_claude_before_discovery() -> None:
    from evals import conftest

    config = SimpleNamespace(getoption=lambda _option: "claude-code")
    with pytest.raises(pytest.fail.Exception, match="supports only.*mock"):
        conftest.agent_backend.__wrapped__(config, EvalConfig(cluster_mode="live", _env_file=None))


async def test_backend_facade_forwards_skill_only_to_claude() -> None:
    raw = SimpleNamespace(query=AsyncMock(return_value="answer"))
    assert await EvalClient(raw, "lcs").query("task", skill="navigator") == "answer"
    raw.query.assert_awaited_once_with("task")
    raw.query.reset_mock()
    assert await EvalClient(raw, "claude-code").query("task", skill="navigator") == "answer"
    raw.query.assert_awaited_once_with("task", skill="navigator")


def test_combine_results_preserves_both_backend_types_and_rejects_other_sessions() -> None:
    first = LCSResult(
        "task",
        "first",
        messages=[{"role": "user", "content": "task"}],
        turns=2,
        conversation_id="session",
    )
    second = ClaudeCodeResult(
        "reply",
        "second",
        messages=[{"role": "user", "content": "reply"}],
        turns=3,
        conversation_id="session",
    )
    merged = combine_results([first, second])
    assert merged.messages == first.messages + second.messages
    assert merged.turns == 5 and merged.final_output == "second"
    second.conversation_id = "other"
    with pytest.raises(ValueError, match="same nonempty conversation"):
        combine_results([first, second])


@pytest.mark.parametrize("backend", ["lcs", "claude-code"])
async def test_navigator_scenario_uses_backend_specific_confirmation_flow(
    monkeypatch: Any, backend: str
) -> None:
    from evals.scenarios import test_navigator_table_completeness as navigator

    from tests.evals.test_command_skill_check import comparison_table

    results = [
        ClaudeCodeResult("opening", "Workload?", conversation_id="session"),
        ClaudeCodeResult("requirements", "Workload?", conversation_id="session"),
        ClaudeCodeResult("workload", "Which GPUs are available?", conversation_id="session"),
        ClaudeCodeResult(
            "hardware",
            comparison_table(),
            tool_calls=[
                ClaudeToolCall("recommend_model", {"preferred_gpu_types": ["A100-80"]}, {}, "call")
            ],
            conversation_id="session",
        ),
    ]
    if backend == "lcs":
        results = [results[0], results[-1]]
    raw = SimpleNamespace(
        initialization={"tools": [], "model": "synthetic"},
        query=AsyncMock(return_value=results[0]),
        follow_up=AsyncMock(side_effect=results[1:]),
    )
    client = EvalClient(raw, backend)
    monkeypatch.setattr(
        navigator, "result_to_conversational_test_case", lambda result, _server: result
    )
    monkeypatch.setattr(navigator, "create_multi_turn_mcp_use_metric", lambda _config: "use")
    monkeypatch.setattr(navigator, "create_task_completion_metric", lambda _config: "completion")
    captured = []

    def evaluate(**kwargs: Any) -> Any:
        captured.append(kwargs["lcs_result"])
        return SimpleNamespace(
            test_results=[SimpleNamespace(metrics_data=[SimpleNamespace(success=True)])]
        )

    await navigator.TestNavigatorTableCompleteness().test_table_has_all_columns_and_rows(
        EvalConfig(), client, None, evaluate
    )
    if backend == "claude-code":
        raw.query.assert_awaited_once_with(
            navigator.TestNavigatorTableCompleteness.TASK, skill="navigator"
        )
        assert raw.follow_up.await_args_list[-1].args[0] == navigator.SCRIPT["hardware"]
        assert raw.follow_up.await_count == 3
    else:
        raw.query.assert_awaited_once_with(navigator.TestNavigatorTableCompleteness.TASK)
        assert raw.follow_up.await_count == 1
    assert captured[0].tool_names_used == ["recommend_model"]
