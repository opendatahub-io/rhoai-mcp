"""Metric configuration and factory functions for RHOAI MCP evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepeval.metrics import MCPUseMetric, MultiTurnMCPUseMetric

from evals.providers import create_judge_llm

if TYPE_CHECKING:
    from deepeval.metrics import MCPTaskCompletionMetric as _MCPTaskCompletionMetric

    from evals.config import EvalConfig


def create_mcp_use_metric(config: EvalConfig) -> MCPUseMetric:
    """Create a single-turn MCP use metric with configured thresholds."""
    return MCPUseMetric(
        threshold=config.mcp_use_threshold,
        model=create_judge_llm(config),
        include_reason=True,
    )


def create_multi_turn_mcp_use_metric(config: EvalConfig) -> MultiTurnMCPUseMetric:
    """Create a multi-turn MCP use metric with configured thresholds."""
    return MultiTurnMCPUseMetric(
        threshold=config.mcp_use_threshold,
        model=create_judge_llm(config),
        include_reason=True,
    )


def create_task_completion_metric(config: EvalConfig) -> _MCPTaskCompletionMetric:
    """Create an MCP task completion metric with configured thresholds."""
    from deepeval.metrics import MCPTaskCompletionMetric

    return MCPTaskCompletionMetric(
        threshold=config.task_completion_threshold,
        model=create_judge_llm(config),
        include_reason=True,
    )
