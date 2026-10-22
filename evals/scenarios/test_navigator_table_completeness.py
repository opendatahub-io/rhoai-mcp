"""Scenario: Navigator recommendation table completeness.

Asserts that the /navigator slash command always outputs a comparison table with
exactly 4 columns (Balanced, Cost, Performance, Quality) and exactly 8
rows (Model, GPU, TTFT p95, E2E p95, Quality score, Cost/month,
Meets SLO, Cluster fit) — with no cells omitted.

The skill makes a single recommend_model call that returns all four slots,
then a second unconstrained call only for any slots that came back empty
(cluster GPU filter found no match). Total calls is 1–5 depending on how
many slots the cluster-constrained call fills.

Uses the mock cluster so no live planner backend is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.command_skill_check import SCRIPT, validate_comparison_table, validate_hardware_request
from evals.config import EvalConfig
from evals.deepeval_helpers import result_to_conversational_test_case
from evals.metrics.config import create_multi_turn_mcp_use_metric, create_task_completion_metric
from evals.result_types import combine_results

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepeval.test_case import MCPServer

    from evals.pytest_client import EvalClient


def _assert_table_completeness(output: str) -> None:
    """Use the primary gate's exact columns, rows, cell counts, and placeholder checks."""
    validate_comparison_table(output)


@pytest.mark.eval
class TestNavigatorTableCompleteness:
    """Ensure the navigator agent always outputs a fully populated comparison table."""

    TASK = (
        "I'm building a customer support chatbot for about 50 concurrent agents. "
        "Cost is my top priority. Get model recommendations and show me the comparison table."
    )

    WORKLOAD_CONFIRMATION = "Yes, that looks right. Please go ahead and get the recommendations."

    @pytest.mark.eval
    async def test_table_has_all_columns_and_rows(
        self,
        eval_config: EvalConfig,
        agent_client: EvalClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[..., Any],
    ) -> None:
        """Navigator must output a table with all 4 columns and all 8 rows populated."""
        # Turn 1: send the task — navigator presents the workload profile and waits for
        # confirmation before calling recommend_model (Phase 2 gate in the skill).
        turn1 = await agent_client.query(self.TASK, skill="navigator")
        assert turn1.conversation_id, "Agent must return a conversation_id for multi-turn flow"
        exchanges = [turn1]

        # Claude invokes the skill with separate requirements/workload confirmations.
        # LCS retains the original two-exchange scenario with its configured agent prompt.
        if agent_client.backend == "claude-code":
            assert "recommend_model" not in turn1.tool_names_used
            requirements = await agent_client.follow_up(
                SCRIPT["requirements"], turn1.conversation_id
            )
            exchanges.append(requirements)
            assert "recommend_model" not in requirements.tool_names_used
            result = await agent_client.follow_up(SCRIPT["workload"], turn1.conversation_id)
            exchanges.append(result)
            if "recommend_model" not in result.tool_names_used:
                validate_hardware_request(
                    {
                        "initialization": agent_client.client.initialization,
                        "exchanges": [
                            {
                                "final_output": exchange.final_output,
                                "tool_calls": [
                                    {
                                        "name": call.name,
                                        "result": call.result,
                                        "is_error": getattr(call, "is_error", False),
                                    }
                                    for call in exchange.tool_calls
                                ],
                            }
                            for exchange in exchanges
                        ],
                    }
                )
                result = await agent_client.follow_up(SCRIPT["hardware"], turn1.conversation_id)
                exchanges.append(result)
        else:
            result = await agent_client.follow_up(self.WORKLOAD_CONFIRMATION, turn1.conversation_id)
            exchanges.append(result)

        # Preserve every observed exchange for metric evaluation.
        merged = combine_results(exchanges)
        all_tool_calls = merged.tool_calls
        recommend_model_calls = [t for t in all_tool_calls if t.name == "recommend_model"]
        assert len(recommend_model_calls) >= 1, (
            f"Navigator must call recommend_model at least once, got {len(recommend_model_calls)}"
        )

        _assert_table_completeness(result.final_output)

        # Give the metric all observed exchanges, including confirmations and hardware input.
        test_case = result_to_conversational_test_case(merged, mcp_server)
        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="navigator_table_completeness",
            lcs_result=merged,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.name} failed: {metric_result.reason}"
            )
