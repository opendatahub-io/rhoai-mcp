"""Scenario: Navigator recommendation table completeness.

Asserts that the navigator agent always outputs a comparison table with
exactly 4 columns (Balanced, Cost, Performance, Quality) and exactly 8
rows (Model, GPU, TTFT p95, E2E p95, Quality score, Cost/month,
Meets SLO, Cluster fit) — with no cells omitted.

Uses the mock cluster so no live planner backend is required.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

from evals.config import EvalConfig
from evals.deepeval_helpers import lcs_result_to_conversational_test_case
from evals.metrics.config import create_multi_turn_mcp_use_metric, create_task_completion_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepeval.test_case import MCPServer

    from evals.lcs_client import LCSClient, LCSResult


EXPECTED_COLUMNS = {"Balanced", "Cost", "Performance", "Quality"}
EXPECTED_ROWS = {
    "Model",
    "GPU",
    "TTFT p95",
    "E2E p95",
    "Quality score",
    "Cost/month",
    "Meets SLO",
    "Cluster fit",
}


def _extract_table_lines(text: str) -> list[str]:
    """Return lines that look like markdown table rows."""
    return [line.strip() for line in text.splitlines() if line.strip().startswith("|")]


def _assert_table_completeness(output: str) -> None:
    """Raise AssertionError if the output table is missing columns or rows."""
    table_lines = _extract_table_lines(output)
    assert table_lines, "No markdown table found in agent output"

    header = table_lines[0]
    for col in EXPECTED_COLUMNS:
        assert col in header, f"Missing column '{col}' in table header: {header}"

    row_labels = {line.split("|")[1].strip() for line in table_lines if "|" in line}
    for row in EXPECTED_ROWS:
        assert any(row in label for label in row_labels), (
            f"Missing row '{row}' in table. Present rows: {sorted(row_labels)}"
        )

    for line in table_lines[2:]:  # skip header and separator
        cells = [c.strip() for c in line.split("|")[1:-1]]
        for cell in cells[1:]:  # skip the row-label cell
            assert cell, (
                f"Empty cell found in table row: {line!r} — must be filled or '—'"
            )


@pytest.mark.eval
class TestNavigatorTableCompleteness:
    """Ensure the navigator agent always outputs a fully populated comparison table."""

    TASK = (
        "I'm building a customer support chatbot for about 50 concurrent agents. "
        "Cost is my top priority. Please use recommend_model to get model "
        "recommendations and show me the comparison table."
    )

    @pytest.mark.eval
    async def test_table_has_all_columns_and_rows(
        self,
        eval_config: EvalConfig,
        lcs_client: LCSClient,
        mcp_server: MCPServer,
        evaluate_and_record: Callable[[str, LCSResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent must output a table with all 4 columns and all 8 rows populated."""
        result = await lcs_client.query(self.TASK)

        assert result.tool_names_used, "Agent should call at least one tool"
        assert "recommend_model" in result.tool_names_used, (
            "Agent must call recommend_model"
        )

        _assert_table_completeness(result.final_output)

        test_case = lcs_result_to_conversational_test_case(result, mcp_server)
        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="navigator_table_completeness",
            lcs_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.name} failed: {metric_result.reason}"
            )
