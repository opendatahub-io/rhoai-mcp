"""Offline trace conversion, schema discovery, recording, and paired scoring tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from evals import deepeval_helpers as helpers
from evals import score_command_skill_traces as scoring
from evals.config import EvalConfig
from evals.lcs_client import LCSResult, LCSToolCall
from evals.reporting.reader import load_records
from evals.result_types import ClaudeToolCall
from mcp.types import Tool

from tests.evals import test_command_skill_check as workflows


@pytest.fixture
async def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = workflows.repo.__wrapped__(tmp_path / "repo")
    await workflows.paired_artifacts(repo)
    return repo / "before.json", repo / "after.json", tmp_path / "scored"


def schemas_for(path: Path) -> list[Tool]:
    artifact = json.loads(path.read_text())
    names = {
        call["name"]
        for scenario in artifact["scenarios"].values()
        for exchange in scenario["exchanges"]
        for call in exchange["tool_calls"]
    }
    return [Tool(name=name, inputSchema={"type": "object"}) for name in sorted(names)]


def fake_evaluation(*, test_cases: Any, metrics: Any, **_kwargs: Any) -> Any:
    return SimpleNamespace(
        test_results=[
            SimpleNamespace(
                metrics_data=[
                    SimpleNamespace(
                        name=metric,
                        score=0.8,
                        success=True,
                        threshold=0.5,
                        reason="Synthetic judge",
                        error=None,
                    )
                    for metric in metrics
                ]
            )
            for _ in test_cases
        ]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"answer": 42},
        {"structuredContent": {"answer": 42}},
        '{"answer": 42}',
        [{"type": "text", "text": '{"answer": 42}'}],
        {"content": [{"type": "text", "text": "answer"}], "structuredContent": {"answer": 42}},
    ],
)
def test_structured_tool_conversion(payload: Any) -> None:
    call = ClaudeToolCall("example", {"x": 1, "server_label": "rhoai-mcp"}, payload, "a")
    converted = helpers.tool_call_to_deepeval(call)
    assert converted.args == {"x": 1}
    assert not converted.result.isError
    assert converted.result.structuredContent["result"] == {"answer": 42}
    if isinstance(payload, dict):
        assert converted.result.content or "structuredContent" in payload
    else:
        assert json.loads(converted.result.content[0].text) == {"answer": 42}
    call.is_error = True
    assert helpers.tool_call_to_deepeval(call).result.isError


def test_tool_calls_are_paired_by_id_and_legacy_lcs_helpers_work() -> None:
    first = LCSToolCall("first", {}, "first result", "first-id")
    second = LCSToolCall("second", {}, "second result", "second-id")
    result = LCSResult(
        "question",
        "answer",
        tool_calls=[second, first],
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "first-id"}]},
            {"role": "tool", "tool_call_id": "first-id", "content": "first result"},
            {"role": "assistant", "content": "answer", "tool_calls": [{"id": "second-id"}]},
        ],
    )
    server = helpers.build_mcp_server_from_schemas([])
    case = helpers.lcs_result_to_conversational_test_case(result, server)
    assert case.turns[1].mcp_tools_called[0].name == "first"
    assert case.turns[2].mcp_tools_called[0].name == "second"
    assert case.turns[1]._mcp_interaction
    from deepeval.metrics.mcp.utils import turn_mcp_interaction_text

    assert "first result" in turn_mcp_interaction_text(case.turns[1])
    assert helpers.lcs_result_to_single_turn_test_case(result, server).actual_output == "answer"
    result.messages[1]["tool_calls"][0]["id"] = "invented"
    with pytest.raises(ValueError, match="Missing or repeated"):
        helpers.result_to_conversational_test_case(result, server)


async def test_schema_discovery_uses_shared_stdio_config_and_paginates(
    monkeypatch: Any, tmp_path: Path
) -> None:
    tools = [Tool(name="first", inputSchema={}), Tool(name="second", inputSchema={})]
    session = SimpleNamespace(
        initialize=AsyncMock(),
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(tools=tools[:1], nextCursor="page-two"),
                SimpleNamespace(tools=tools[1:], nextCursor=None),
            ]
        ),
    )
    captured = []

    @asynccontextmanager
    async def stdio(parameters: Any) -> Any:
        captured.append(parameters)
        yield ("read", "write")

    @asynccontextmanager
    async def client(_read: Any, _write: Any) -> Any:
        yield session

    monkeypatch.setattr("mcp.client.stdio.stdio_client", stdio)
    monkeypatch.setattr("mcp.ClientSession", client)
    assert await helpers.fetch_stdio_tool_schemas(tmp_path) == tools
    parameters = captured[0]
    assert parameters.cwd == tmp_path
    assert parameters.command == sys.executable
    assert parameters.args == ["-m", "rhoai_mcp", "--transport", "stdio", "--mock-cluster"]
    assert parameters.env["RHOAI_MCP_PLANNER_MODE"] == "local"
    assert parameters.env["RHOAI_MCP_PLANNER_MODEL_CATALOG_URL"] == ""
    assert parameters.env["PLANNER_DETECT_CLUSTER_GPUS"] == "false"
    assert session.list_tools.await_args_list[1].kwargs == {"cursor": "page-two"}


def test_scorer_does_not_import_claude_sdk_or_require_judge_credentials() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import evals.score_command_skill_traces; assert 'claude_agent_sdk' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_complete_pair_preserves_conversations_and_phase_metadata(
    inputs: tuple[Path, Path, Path], monkeypatch: Any
) -> None:
    before, after, output = inputs
    fetcher = AsyncMock(return_value=schemas_for(before))
    evaluate = importlib.import_module("deepeval.evaluate")
    observed = []

    def evaluate_saved(**kwargs: Any) -> Any:
        observed.append(kwargs["test_cases"][0])
        return fake_evaluation(**kwargs)

    monkeypatch.setattr(evaluate, "evaluate", evaluate_saved)
    assert scoring.score_traces(
        before=before,
        after=after,
        output_dir=output,
        config=EvalConfig(),
        schema_fetcher=fetcher,
        metric_factory=lambda _config: ["MCP use", "Task completion"],
    )
    fetcher.assert_awaited_once()
    records = load_records(str(output / "commands.jsonl")) + load_records(
        str(output / "skills.jsonl")
    )
    assert len(records) == 4
    assert {record.phase for record in records} == {"commands", "skills"}
    assert len({record.run_id for record in records}) == 1
    assert all(record.environment.agent_backend == "claude-code" for record in records)
    assert all(record.environment.lcs_url == "" for record in records)
    assert all(record.passed and len(record.metrics) == 2 for record in records)
    assert [case.name for case in observed] == [
        "commands/navigator",
        "commands/deployment",
        "skills/navigator",
        "skills/deployment",
    ]
    assert len([turn for turn in observed[0].turns if turn.role == "user"]) == 4
    assert "Delta" in (output / "comparison.md").read_text()
    assert json.loads((output / "run.json").read_text())["status"] == "passed"


def test_invalid_primary_gate_stops_before_schema_or_judge(inputs: tuple[Path, Path, Path]) -> None:
    before, after, output = inputs
    artifact = json.loads(after.read_text())
    artifact["checks"]["navigator.handoff"]["status"] = "failed"
    after.write_text(json.dumps(artifact))
    fetcher = AsyncMock()
    with pytest.raises(ValueError, match="Primary gate"):
        scoring.score_traces(
            before=before,
            after=after,
            output_dir=output,
            config=EvalConfig(),
            schema_fetcher=fetcher,
        )
    fetcher.assert_not_called()
    assert not output.exists()


def test_failed_judge_metric_is_reported_without_changing_primary_gate(
    inputs: tuple[Path, Path, Path], monkeypatch: Any
) -> None:
    before, after, output = inputs
    original = before.read_bytes(), after.read_bytes()

    def failed(**kwargs: Any) -> Any:
        result = fake_evaluation(**kwargs)
        result.test_results[0].metrics_data[0].success = False
        result.test_results[0].metrics_data[0].score = 0.2
        return result

    monkeypatch.setattr(importlib.import_module("deepeval.evaluate"), "evaluate", failed)
    assert not scoring.score_traces(
        before=before,
        after=after,
        output_dir=output,
        config=EvalConfig(),
        schema_fetcher=AsyncMock(return_value=schemas_for(before)),
        metric_factory=lambda _config: ["MCP use", "Task completion"],
    )
    assert "FAIL" in (output / "comparison.md").read_text()
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert (before.read_bytes(), after.read_bytes()) == original


def test_missing_paired_metric_fails_even_when_each_available_metric_passes(
    inputs: tuple[Path, Path, Path], monkeypatch: Any
) -> None:
    before, after, output = inputs

    def mismatched(**kwargs: Any) -> Any:
        result = fake_evaluation(**kwargs)
        if kwargs["test_cases"][0].name.startswith("skills/"):
            result.test_results[0].metrics_data[0].name = "Unexpected metric"
        return result

    monkeypatch.setattr(importlib.import_module("deepeval.evaluate"), "evaluate", mismatched)
    assert not scoring.score_traces(
        before=before,
        after=after,
        output_dir=output,
        config=EvalConfig(),
        schema_fetcher=AsyncMock(return_value=schemas_for(before)),
        metric_factory=lambda _config: ["MCP use", "Task completion"],
    )
    assert "missing" in (output / "comparison.md").read_text()
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_schema_mismatch_stops_before_judge(inputs: tuple[Path, Path, Path]) -> None:
    before, after, output = inputs
    with pytest.raises(ValueError, match="missing from current schemas"):
        scoring.score_traces(
            before=before,
            after=after,
            output_dir=output,
            config=EvalConfig(),
            schema_fetcher=AsyncMock(return_value=[]),
        )
    assert json.loads((output / "run.json").read_text())["status"] == "error"
    assert not (output / "commands.jsonl").exists()


def test_legacy_records_without_phase_remain_readable(tmp_path: Path) -> None:
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {
                "run_id": "old",
                "timestamp": "2026-01-01",
                "scenario": "legacy",
                "git": {"commit": "abc", "branch": "main"},
                "environment": {"llm_provider": "lcs", "llm_model": "legacy-model"},
            }
        )
        + "\n"
    )
    record = load_records(str(path))[0]
    assert record.phase is None
    assert record.environment.agent_backend == "lcs"
    assert record.environment.agent_model == "legacy-model"


def test_partial_scoring_error_keeps_completed_records(
    inputs: tuple[Path, Path, Path], monkeypatch: Any
) -> None:
    before, after, output = inputs
    evaluate = importlib.import_module("deepeval.evaluate")
    calls = 0

    def interrupted(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic judge failure")
        return fake_evaluation(**kwargs)

    monkeypatch.setattr(evaluate, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="synthetic judge failure"):
        scoring.score_traces(
            before=before,
            after=after,
            output_dir=output,
            config=EvalConfig(),
            schema_fetcher=AsyncMock(return_value=schemas_for(before)),
            metric_factory=lambda _config: ["MCP use", "Task completion"],
        )
    assert len(load_records(str(output / "commands.jsonl"))) == 1
    assert json.loads((output / "run.json").read_text())["status"] == "error"
    assert not (output / "comparison.md").exists()
    with pytest.raises(ValueError, match="already exists"):
        scoring.score_traces(before=before, after=after, output_dir=output, config=EvalConfig())
