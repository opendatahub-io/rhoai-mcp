"""Shared pytest fixtures for RHOAI MCP evaluations."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import pytest

from evals.config import EvalConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepeval.test_case import MCPServer

    from evals.lcs_client import LCSClient, LCSResult
    from evals.reporting.recorder import EvalRecorder

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def eval_config() -> EvalConfig:
    """Load evaluation configuration from environment."""
    return EvalConfig()


@pytest.fixture(scope="session")
def lcs_client(eval_config: EvalConfig) -> LCSClient:
    """Create an LCS client and verify the service is healthy.

    Retries health checks for up to 120 seconds to allow container startup.
    """
    from evals.lcs_client import LCSClient

    client = LCSClient(
        base_url=eval_config.lcs_url,
        timeout=eval_config.lcs_timeout,
    )

    # Wait for LCS to be healthy
    max_retries = 24
    retry_interval = 5

    for attempt in range(max_retries):
        healthy = asyncio.get_event_loop().run_until_complete(client.health_check())
        if healthy:
            logger.info(f"LCS is healthy at {eval_config.lcs_url}")
            return client
        if attempt < max_retries - 1:
            logger.info(
                f"LCS not ready (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {retry_interval}s..."
            )
            import time

            time.sleep(retry_interval)

    pytest.fail(f"LCS not healthy after {max_retries * retry_interval}s at {eval_config.lcs_url}")


@pytest.fixture(scope="session")
def tool_schemas(eval_config: EvalConfig) -> list[dict[str, Any]]:
    """Fetch tool schemas from the rhoai-mcp server."""
    from evals.deepeval_helpers import fetch_tool_schemas

    return asyncio.get_event_loop().run_until_complete(
        fetch_tool_schemas(eval_config.rhoai_mcp_url)
    )


@pytest.fixture(scope="session")
def mcp_server(tool_schemas: list[dict[str, Any]]) -> MCPServer:
    """Build a DeepEval MCPServer from fetched tool schemas."""
    from evals.deepeval_helpers import build_mcp_server_from_schemas

    return build_mcp_server_from_schemas(tool_schemas)


@pytest.fixture(scope="session")
def eval_recorder(eval_config: EvalConfig) -> EvalRecorder:
    """Session-scoped eval result recorder."""
    from evals.reporting.recorder import EvalRecorder

    return EvalRecorder(eval_config)


@pytest.fixture
def evaluate_and_record(
    eval_recorder: EvalRecorder,
) -> Callable[[str, LCSResult, list[Any], list[Any]], Any]:
    """Return a callable that wraps deepeval.evaluate() with recording."""
    from evals.reporting.recorder import evaluate_and_record as _evaluate_and_record

    def _wrapper(
        scenario: str,
        lcs_result: LCSResult,
        test_cases: list[Any],
        metrics: list[Any],
    ) -> Any:
        return _evaluate_and_record(
            recorder=eval_recorder,
            scenario=scenario,
            lcs_result=lcs_result,
            test_cases=test_cases,
            metrics=metrics,
        )

    return _wrapper
