"""Scenario: Cluster Exploration.

Tests that the agent can explore a cluster, discover projects,
check workbenches, and identify GPU availability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.config import EvalConfig
from evals.deepeval_helpers import lcs_result_to_conversational_test_case
from evals.metrics.config import create_multi_turn_mcp_use_metric, create_task_completion_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepeval.test_case import MCPServer

    from evals.lcs_client import LCSClient, LCSResult


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
        lcs_client: LCSClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[[str, LCSResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should use cluster/project exploration tools."""
        result = await lcs_client.query(self.TASK)

        # Verify the agent called at least some relevant tools
        tool_names = result.tool_names_used
        assert len(tool_names) > 0, "Agent should call at least one tool"

        # Build DeepEval test case and evaluate
        test_case = lcs_result_to_conversational_test_case(result, mcp_server)

        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="cluster_exploration",
            lcs_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        # Assert all metrics passed
        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.metric_name} failed: {metric_result.reason}"
            )
