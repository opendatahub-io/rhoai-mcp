"""Scenario: Troubleshooting a Failed Training Job.

Tests that the agent can diagnose why a training job failed
by checking status, events, and logs.
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
class TestTroubleshooting:
    """Evaluate agent's ability to diagnose a failed training job."""

    TASK = (
        "The training job 'failed-training-001' in the ml-experiments project "
        "has failed. Diagnose the issue and explain what went wrong."
    )

    @pytest.mark.eval
    async def test_troubleshooting(
        self,
        eval_config: EvalConfig,
        lcs_client: LCSClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[[str, LCSResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should use diagnostic tools to investigate the failure."""
        result = await lcs_client.query(self.TASK)

        tool_names = result.tool_names_used
        assert len(tool_names) > 0, "Agent should call at least one tool"

        # The agent should mention the error in its output
        assert result.final_output, "Agent should produce a diagnostic summary"

        test_case = lcs_result_to_conversational_test_case(result, mcp_server)

        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="troubleshooting",
            lcs_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.name} failed: {metric_result.reason}"
            )
