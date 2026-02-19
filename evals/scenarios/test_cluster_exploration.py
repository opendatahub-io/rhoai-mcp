"""Scenario: Cluster Exploration.

Tests that the agent can explore a cluster, discover projects,
check workbenches, and identify GPU availability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.agent import MCPAgent
from evals.config import EvalConfig
from evals.deepeval_helpers import build_mcp_server, result_to_conversational_test_case
from evals.mcp_harness import MCPHarness
from evals.metrics.config import create_multi_turn_mcp_use_metric, create_task_completion_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    from evals.agent import AgentResult


@pytest.mark.eval
class TestClusterExploration:
    """Evaluate agent's ability to explore a RHOAI cluster."""

    TASK = (
        "Explore the cluster. What data science projects exist, "
        "what workbenches are running, and are GPUs available?"
    )

    @pytest.mark.eval
    async def test_cluster_exploration(
        self,
        eval_config: EvalConfig,
        harness: MCPHarness,
        agent: MCPAgent,
        evaluate_and_record: Callable[[str, AgentResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should use cluster/project exploration tools."""
        result = await agent.run(self.TASK)

        # Verify the agent called at least some relevant tools
        tool_names = result.tool_names_used
        assert len(tool_names) > 0, "Agent should call at least one tool"

        # Build DeepEval test case and evaluate
        mcp_server = build_mcp_server(harness)
        test_case = result_to_conversational_test_case(result, mcp_server)

        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="cluster_exploration",
            agent_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        # Assert all metrics passed
        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.metric_name} failed: {metric_result.reason}"
            )
