"""Shared pytest fixtures for RHOAI MCP evaluations."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio

from evals.config import ClusterMode, EvalConfig
from evals.pytest_client import EvalClient
from evals.result_types import AgentResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepeval.test_case import MCPServer

    from evals.lcs_client import LCSClient
    from evals.reporting.recorder import EvalRecorder

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--client",
        choices=("lcs", "claude-code"),
        default="lcs",
        help="Agent backend for live evaluations (default: lcs)",
    )
    parser.addoption("--eval-results", type=Path, default=None, help="Explicit JSONL output path")


@pytest.fixture(scope="session")
def agent_backend(pytestconfig: pytest.Config, eval_config: EvalConfig) -> str:
    backend = str(pytestconfig.getoption("--client"))
    if backend == "claude-code" and eval_config.cluster_mode != ClusterMode.MOCK:
        pytest.fail("--client=claude-code currently supports only RHOAI_EVAL_CLUSTER_MODE=mock")
    return backend


@pytest.fixture(scope="session")
def eval_config() -> EvalConfig:
    """Load evaluation configuration from environment."""
    return EvalConfig()


@pytest.fixture(scope="session")
def lcs_client(eval_config: EvalConfig) -> LCSClient:
    """Create an LCS client and verify the service is healthy.

    Retries health checks for up to 120 seconds to allow container startup.
    Uses synchronous HTTP for health checks to avoid event loop issues
    (the httpx.AsyncClient is created fresh for the test event loop).
    """
    from evals.lcs_client import LCSClient

    # Wait for LCS to be healthy using synchronous HTTP
    max_retries = 24
    retry_interval = 5
    health_url = f"{eval_config.lcs_url.rstrip('/')}/readiness"

    for attempt in range(max_retries):
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code == 200:
                logger.info(f"LCS is healthy at {eval_config.lcs_url}")
                return LCSClient(
                    base_url=eval_config.lcs_url,
                    timeout=eval_config.lcs_timeout,
                )
        except httpx.HTTPError:
            pass
        if attempt < max_retries - 1:
            logger.info(
                f"LCS not ready (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {retry_interval}s..."
            )
            time.sleep(retry_interval)

    pytest.fail(f"LCS not healthy after {max_retries * retry_interval}s at {eval_config.lcs_url}")


@pytest.fixture(scope="session")
def tool_schemas(eval_config: EvalConfig, agent_backend: str) -> list[Any]:
    """Discover local stdio tools for Claude, HTTP tools for LCS."""
    from evals.deepeval_helpers import fetch_stdio_tool_schemas, fetch_tool_schemas

    if agent_backend == "claude-code":
        return asyncio.run(fetch_stdio_tool_schemas())
    return asyncio.run(
        fetch_tool_schemas(eval_config.rhoai_mcp_url, eval_config.rhoai_mcp_transport)
    )


@pytest_asyncio.fixture(scope="function", loop_scope="function")
async def agent_client(
    request: pytest.FixtureRequest,
    agent_backend: str,
    eval_config: EvalConfig,
    tool_schemas: list[Any],
) -> AsyncIterator[EvalClient]:
    """One persistent client per scenario, opened/closed in the test's event loop."""
    if agent_backend == "lcs":
        yield EvalClient(request.getfixturevalue("lcs_client"), "lcs")
        return
    try:
        from evals.claude_code_client import ClaudeCodeClient
    except ImportError as exc:
        pytest.fail(f"Claude Code requires the claude-check dependency group: {exc}")
    async with ClaudeCodeClient(
        timeout=eval_config.lcs_timeout,
        allowed_tools=[tool.name for tool in tool_schemas],
    ) as client:
        yield EvalClient(client, "claude-code")


@pytest.fixture(scope="session")
def mcp_server(tool_schemas: list[Any]) -> MCPServer:
    """Build a DeepEval MCPServer from fetched tool schemas."""
    from evals.deepeval_helpers import build_mcp_server_from_schemas

    return build_mcp_server_from_schemas(tool_schemas)


@pytest.fixture(scope="session")
def eval_recorder(
    eval_config: EvalConfig, agent_backend: str, pytestconfig: pytest.Config
) -> EvalRecorder:
    """Session-scoped eval result recorder."""
    from evals.reporting.recorder import EvalRecorder

    return EvalRecorder(
        eval_config, path=pytestconfig.getoption("--eval-results"), agent_backend=agent_backend
    )


@pytest.fixture
def evaluate_and_record(
    eval_recorder: EvalRecorder,
    agent_client: EvalClient,
) -> Callable[..., Any]:
    """Return a callable that wraps deepeval.evaluate() with recording."""
    from evals.reporting.recorder import evaluate_and_record as _evaluate_and_record

    def _wrapper(
        scenario: str,
        lcs_result: AgentResult,
        test_cases: list[Any],
        metrics: list[Any],
    ) -> Any:
        eval_recorder.agent_model = agent_client.agent_model
        eval_recorder.phase = agent_client.phase
        return _evaluate_and_record(
            recorder=eval_recorder,
            scenario=scenario,
            lcs_result=lcs_result,
            test_cases=test_cases,
            metrics=metrics,
        )

    return _wrapper
