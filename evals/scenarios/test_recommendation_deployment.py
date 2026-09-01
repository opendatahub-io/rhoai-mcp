"""Scenario: Recommendation-to-Deployment Workflow.

Tests the end-to-end workflow of planning and executing a model
deployment from a pre-provided recommendation. This exercises
plan_deployment and deploy_model tools against the mock cluster.

Note: recommend_model is excluded because it requires an external
planner backend. This scenario focuses on the cluster-side workflow
of translating a recommendation into a running deployment.
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
class TestRecommendationDeployment:
    """Evaluate agent's ability to plan and execute a model deployment from a recommendation."""

    TASK = (
        "I have a model recommendation from our planning tool and I need to "
        "deploy it. The recommendation is for meta-llama/Llama-3.1-8B with "
        "1x A100 GPU in the production-models project. The model artifacts "
        "are at s3://models/llama-3-8b. "
        "Please plan the deployment using plan_deployment (pass the "
        'recommendation as JSON: {"model_id": "meta-llama/Llama-3.1-8B", '
        '"model_name": "Llama 3.1 8B", "gpu_config": {"gpu_type": "A100-80", '
        '"gpu_count": 1, "tensor_parallel": 1, "replicas": 1}, '
        '"cost_per_month_usd": 1200, "meets_slo": true}), '
        "review the plan, check what serving runtimes are available, "
        "and verify the deployment can proceed."
    )

    @pytest.mark.eval
    async def test_recommendation_deployment(
        self,
        eval_config: EvalConfig,
        lcs_client: LCSClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[[str, LCSResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should plan and validate a model deployment from a recommendation."""
        result = await lcs_client.query(self.TASK)

        tool_names = result.tool_names_used
        assert len(tool_names) > 0, "Agent should call at least one tool"

        test_case = lcs_result_to_conversational_test_case(result, mcp_server)

        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="recommendation_deployment",
            lcs_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.name} failed: {metric_result.reason}"
            )
