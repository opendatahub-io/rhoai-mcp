"""Shared pytest fixtures for RHOAI MCP evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.config import ClusterMode, EvalConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from evals.agent import AgentResult, MCPAgent
    from evals.mcp_harness import MCPHarness
    from evals.reporting.recorder import EvalRecorder


@pytest.fixture(scope="session")
def eval_config() -> EvalConfig:
    """Load evaluation configuration from environment."""
    return EvalConfig()


@pytest.fixture(scope="session")
def is_mock(eval_config: EvalConfig) -> bool:
    """Whether we're running against a mock cluster."""
    return eval_config.cluster_mode == ClusterMode.MOCK


@pytest.fixture(scope="session")
def eval_recorder(eval_config: EvalConfig) -> EvalRecorder:
    """Session-scoped eval result recorder."""
    from evals.reporting.recorder import EvalRecorder

    return EvalRecorder(eval_config)


@pytest.fixture
async def harness(eval_config: EvalConfig) -> AsyncIterator[MCPHarness]:
    """Create an MCP harness with the configured cluster mode."""
    from evals.mcp_harness import MCPHarness

    async with MCPHarness.running(eval_config) as h:
        yield h


@pytest.fixture
async def agent(eval_config: EvalConfig, harness: MCPHarness) -> MCPAgent:
    """Create an LLM agent connected to the MCP harness."""
    from evals.agent import MCPAgent

    return MCPAgent(config=eval_config, harness=harness)


@pytest.fixture
def evaluate_and_record(
    eval_recorder: EvalRecorder,
) -> Callable[[str, AgentResult, list[Any], list[Any]], Any]:
    """Return a callable that wraps deepeval.evaluate() with recording."""
    from evals.reporting.recorder import evaluate_and_record as _evaluate_and_record

    def _wrapper(
        scenario: str,
        agent_result: AgentResult,
        test_cases: list[Any],
        metrics: list[Any],
    ) -> Any:
        return _evaluate_and_record(
            recorder=eval_recorder,
            scenario=scenario,
            agent_result=agent_result,
            test_cases=test_cases,
            metrics=metrics,
        )

    return _wrapper
