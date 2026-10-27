"""Scenario: Training Workflow.

Tests that the agent can plan and create a fine-tuning training job,
including checking prerequisites and estimating resources.
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
class TestTrainingWorkflow:
    """Evaluate agent's ability to set up a training job."""

    TASK = (
        "Fine-tune meta-llama/Llama-3.1-8B using LoRA on the 'alpaca-cleaned' "
        "dataset in the ml-experiments project. Check prerequisites, plan "
        "resources, and create the training job."
    )

    @pytest.mark.eval
    async def test_training_workflow(
        self,
        eval_config: EvalConfig,
        lcs_client: LCSClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[[str, LCSResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should use training tools to plan and create a job."""
        result = await lcs_client.query(self.TASK)

        tool_names = result.tool_names_used
        assert len(tool_names) > 0, "Agent should call at least one tool"

        test_case = lcs_result_to_conversational_test_case(result, mcp_server)

        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="training_workflow",
            lcs_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.name} failed: {metric_result.reason}"
            )
