"""Scenario: Tool Discovery.

Tests that the agent can use meta tools to discover which tools
are available for a given workflow.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from evals.config import EvalConfig
from evals.deepeval_helpers import result_to_single_turn_test_case
from evals.metrics.config import create_mcp_use_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepeval.test_case import MCPServer

    from evals.pytest_client import EvalClient
    from evals.result_types import AgentResult


SETUP_CAPABILITIES = (
    ("create_data_science_project", "a new data science project"),
    ("create_storage", "persistent storage"),
    ("create_s3_data_connection", "an S3 data connection"),
    ("create_workbench", "a notebook workbench"),
)
DISCOVERY_TOOLS = {"suggest_tools", "list_tool_categories"}


def discovery_request(available_tools: set[str]) -> tuple[str, list[str]]:
    """Exercise only setup capabilities advertised by the connected MCP server."""
    supported = [
        (tool, description) for tool, description in SETUP_CAPABILITIES if tool in available_tools
    ]
    if not supported:
        raise ValueError(
            "No project, storage, data-connection, or workbench setup tools are enabled"
        )
    if not DISCOVERY_TOOLS & available_tools:
        raise ValueError("No MCP discovery tools are enabled")
    descriptions = [description for _tool, description in supported]
    task = (
        "Use the available MCP discovery tools to inspect this session's capabilities. "
        f"I want to set up {', '.join(descriptions)}. "
        "Which currently available tools should I use and in what order? "
        "Recommend only tools exposed by the connected server. "
        "Discovery hints may describe other configurations; verify tool availability "
        "before including a tool in the workflow. Describe the steps without creating "
        "or changing any resources."
    )
    return task, [tool for tool, _description in supported]


def validate_discovery_result(
    result: AgentResult, available_tools: set[str], expected_tools: list[str]
) -> None:
    """Check actual discovery and enabled setup coverage before invoking the judge."""
    assert result.final_output, "Agent should provide tool recommendations"
    assert DISCOVERY_TOOLS & set(result.tool_names_used), (
        "Agent must call an enabled MCP discovery tool"
    )
    assert not set(result.tool_names_used) - available_tools, (
        "Agent called tools absent from the active server"
    )
    assert not any(
        name.startswith(("create_", "delete_", "start_", "stop_", "update_"))
        or name == "manage_resource"
        for name in result.tool_names_used
    ), "Discovery must not create or change resources"
    for tool in expected_tools:
        assert re.search(rf"\b{re.escape(tool)}\b", result.final_output), (
            f"Missing enabled setup tool: {tool}"
        )


@pytest.mark.eval
class TestToolDiscovery:
    """Evaluate agent's ability to discover relevant tools."""

    @pytest.mark.eval
    async def test_tool_discovery(
        self,
        eval_config: EvalConfig,
        agent_client: EvalClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[..., Any],
    ) -> None:
        """Agent should use meta/discovery tools to suggest a workflow."""
        # MCP list_tools reflects enabled plugins, health, and access controls.
        # Local RHOAIConfig defaults may differ from a remote LCS server's setup.
        available_tools = {tool.name for tool in mcp_server.available_tools or []}
        try:
            task, expected_tools = discovery_request(available_tools)
        except ValueError as exc:
            pytest.skip(str(exc))
        result = await agent_client.query(task)

        validate_discovery_result(result, available_tools, expected_tools)

        test_case = result_to_single_turn_test_case(result, mcp_server)

        metrics = [create_mcp_use_metric(eval_config)]

        eval_result = evaluate_and_record(
            scenario="tool_discovery",
            lcs_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.name} failed: {metric_result.reason}"
            )
