"""Scenario: Model Deployment.

Tests that the agent can deploy a model using KServe/vLLM
and verify the endpoint is ready.
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
class TestModelDeployment:
    """Evaluate agent's ability to deploy a model for inference."""

    TASK = (
        "Deploy the granite model from s3://models/granite-3b using the "
        "vLLM runtime in the production-models project. Check what serving "
        "runtimes are available and verify the deployment status."
    )

    @pytest.mark.eval
    async def test_model_deployment(
        self,
        eval_config: EvalConfig,
        harness: MCPHarness,
        agent: MCPAgent,
        evaluate_and_record: Callable[[str, AgentResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should use inference tools to deploy and verify a model."""
        result = await agent.run(self.TASK)

        tool_names = result.tool_names_used
        assert len(tool_names) > 0, "Agent should call at least one tool"

        mcp_server = build_mcp_server(harness)
        test_case = result_to_conversational_test_case(result, mcp_server)

        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="model_deployment",
            agent_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.metric_name} failed: {metric_result.reason}"
            )
