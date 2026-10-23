"""Discovery expectations come from the connected server's tool inventory."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from evals.config import EvalConfig
from evals.mcp_config import build_mcp_servers
from evals.result_types import ClaudeCodeResult, ClaudeToolCall
from evals.scenarios import test_tool_discovery as discovery

DEFAULT_SETUP_TOOLS = {
    "create_data_science_project",
    "create_storage",
    "create_s3_data_connection",
}


def test_shared_launch_config_forwards_explicit_plugin_override(monkeypatch: Any) -> None:
    plugins = "projects,notebooks,meta-composites"
    monkeypatch.setenv("RHOAI_MCP_ENABLED_PLUGINS", plugins)
    environment = build_mcp_servers()["rhoai-mcp"]["env"]
    assert environment["RHOAI_MCP_ENABLED_PLUGINS"] == plugins
    assert environment["RHOAI_MCP_PLANNER_MODE"] == "local"
    monkeypatch.delenv("RHOAI_MCP_ENABLED_PLUGINS")
    assert "RHOAI_MCP_ENABLED_PLUGINS" not in build_mcp_servers()["rhoai-mcp"]["env"]


@pytest.mark.parametrize("notebooks_enabled", [False, True])
def test_request_tracks_advertised_workbench_capability(notebooks_enabled: bool) -> None:
    tools = DEFAULT_SETUP_TOOLS | discovery.DISCOVERY_TOOLS
    if notebooks_enabled:
        tools.add("create_workbench")
    task, expected = discovery.discovery_request(tools)
    assert ("workbench" in task) == notebooks_enabled
    assert ("create_workbench" in expected) == notebooks_enabled
    assert set(expected) == tools - discovery.DISCOVERY_TOOLS


def test_filtered_inventory_overrides_local_configuration(monkeypatch: Any) -> None:
    # Even if a local override enables notebooks, the connected server may expose
    # only project creation (different deployment, plugin health, or RBAC).
    monkeypatch.setenv("RHOAI_MCP_ENABLED_PLUGINS", "projects,notebooks,meta-composites")
    task, expected = discovery.discovery_request({"create_data_science_project", "suggest_tools"})
    assert expected == ["create_data_science_project"]
    assert "workbench" not in task and "storage" not in task and "S3" not in task


@pytest.mark.parametrize("available", [set(), {"suggest_tools"}, {"create_storage"}])
def test_missing_setup_or_discovery_capability_is_reported(available: set[str]) -> None:
    with pytest.raises(ValueError, match="No .*tools are enabled"):
        discovery.discovery_request(available)


def result(*names: str, output: str | None = None) -> ClaudeCodeResult:
    return ClaudeCodeResult(
        "discovery",
        output
        if output is not None
        else "Use create_data_science_project, then create_storage and create_s3_data_connection.",
        tool_calls=[
            ClaudeToolCall(name, {}, {}, f"call-{index}") for index, name in enumerate(names)
        ],
    )


def test_grounded_discovery_passes_without_disabled_workbench_expectations() -> None:
    available = DEFAULT_SETUP_TOOLS | discovery.DISCOVERY_TOOLS
    _task, expected = discovery.discovery_request(available)
    discovery.validate_discovery_result(result("suggest_tools"), available, expected)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (result(), "must call"),
        (result("suggest_tools", "list_workbenches"), "absent from the active server"),
        (result("suggest_tools", "create_storage"), "must not create"),
        (result("suggest_tools", output="Use create_storage."), "Missing enabled setup tool"),
    ],
)
def test_discovery_checks_keep_rejecting_ungrounded_or_incomplete_results(
    response: ClaudeCodeResult, message: str
) -> None:
    available = DEFAULT_SETUP_TOOLS | discovery.DISCOVERY_TOOLS
    _task, expected = discovery.discovery_request(available)
    with pytest.raises(AssertionError, match=message):
        discovery.validate_discovery_result(response, available, expected)


@pytest.mark.parametrize("notebooks_enabled", [False, True])
async def test_live_scenario_builds_request_and_assertions_from_mcp_schemas(
    monkeypatch: Any, notebooks_enabled: bool
) -> None:
    available = DEFAULT_SETUP_TOOLS | discovery.DISCOVERY_TOOLS
    if notebooks_enabled:
        available.add("create_workbench")
    response = result("suggest_tools")
    if notebooks_enabled:
        response.final_output += " Then use create_workbench."
    client = SimpleNamespace(query=AsyncMock(return_value=response))
    server = SimpleNamespace(available_tools=[SimpleNamespace(name=name) for name in available])
    monkeypatch.setattr(discovery, "result_to_single_turn_test_case", lambda r, _server: r)
    monkeypatch.setattr(discovery, "create_mcp_use_metric", lambda _config: "unchanged judge")
    evaluated = []

    def evaluate(**kwargs: Any) -> Any:
        evaluated.append(kwargs)
        return SimpleNamespace(
            test_results=[SimpleNamespace(metrics_data=[SimpleNamespace(success=True)])]
        )

    await discovery.TestToolDiscovery().test_tool_discovery(EvalConfig(), client, server, evaluate)
    task = client.query.await_args.args[0]
    assert ("workbench" in task) == notebooks_enabled
    assert len(evaluated) == 1 and evaluated[0]["metrics"] == ["unchanged judge"]


async def test_unsupported_configuration_skips_before_agent_or_judge_query() -> None:
    client = SimpleNamespace(query=AsyncMock())
    evaluate = AsyncMock()
    server = SimpleNamespace(available_tools=[SimpleNamespace(name="get_inference_service")])
    with pytest.raises(pytest.skip.Exception, match="No .*setup tools"):
        await discovery.TestToolDiscovery().test_tool_discovery(
            EvalConfig(), client, server, evaluate
        )
    client.query.assert_not_awaited()
    evaluate.assert_not_called()
