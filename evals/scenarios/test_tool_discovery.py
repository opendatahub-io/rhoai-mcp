"""Scenario: Tool Discovery.

Tests that the agent can use meta tools to discover which tools
are available for a given workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.config import EvalConfig
from evals.deepeval_helpers import lcs_result_to_single_turn_test_case
from evals.metrics.config import create_mcp_use_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepeval.test_case import MCPServer

    from evals.lcs_client import LCSClient, LCSResult


@pytest.mark.eval
class TestToolDiscovery:
    """Evaluate agent's ability to discover relevant tools."""

    TASK = (
        "I want to set up a new project with storage, an S3 data connection, "
        "and a workbench. What tools should I use and in what order?"
    )

    @pytest.mark.eval
    async def test_tool_discovery(
        self,
        eval_config: EvalConfig,
        lcs_client: LCSClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[[str, LCSResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should use meta/discovery tools to suggest a workflow."""
        result = await lcs_client.query(self.TASK)

        # The agent should provide tool recommendations
        assert result.final_output, "Agent should provide tool recommendations"

        test_case = lcs_result_to_single_turn_test_case(result, mcp_server)

        metrics = [create_mcp_use_metric(eval_config)]

        eval_result = evaluate_and_record(
            scenario="tool_discovery",
            lcs_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.metric_name} failed: {metric_result.reason}"
            )
