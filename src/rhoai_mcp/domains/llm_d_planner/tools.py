"""MCP tools for llm-d-planner model recommendation workflow.

Provides 4 workflow-token-chained tools:
  extract_intent → prepare_model_tech_specs → get_recommended_models → get_deployment_config
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.llm_d_planner.client import (
    PlannerAPIError,
    PlannerClient,
    PlannerConnectionError,
)
from rhoai_mcp.utils.workflow_token import workflow_step

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

_MAX_TEXT_LENGTH = 4000


def _error_result(
    error: str,
    *,
    hint: str | None = None,
    status_code: int | None = None,
) -> dict[str, Any]:
    """Build a standardized error dict."""
    result: dict[str, Any] = {"error": error}
    if hint:
        result["hint"] = hint
    if status_code is not None:
        result["status_code"] = status_code
    return result


def _handle_planner_error(
    e: PlannerConnectionError | PlannerAPIError,
    server: RHOAIServer,
) -> dict[str, Any]:
    """Convert a planner client error to a tool error dict."""
    if isinstance(e, PlannerConnectionError):
        return _error_result(
            f"llm-d-planner service unavailable: {e}",
            hint=(
                "llm-d-planner may be warming up (first request loads the LLM model). "
                "Check RHOAI_MCP_PLANNER_URL "
                f"(currently: {server.config.planner_url}) and RHOAI_MCP_PLANNER_TIMEOUT "
                f"(currently: {server.config.planner_timeout}s)."
            ),
        )
    return _error_result(
        f"llm-d-planner API error: {e.detail}",
        status_code=e.status_code,
    )


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register llm-d-planner tools with the MCP server."""

    @mcp.tool()
    @workflow_step(produces="intent_extracted")
    def extract_intent(text: str) -> dict[str, Any]:
        """Extract deployment intent from a natural language description.

        Analyzes the user's text to identify use case, scale, priorities,
        and other deployment requirements. Returns structured intent data
        and a workflow_token for the next step (prepare_model_tech_specs).
        """
        if not text or not text.strip():
            return _error_result("'text' must be non-empty")
        if len(text) > _MAX_TEXT_LENGTH:
            return _error_result(f"'text' exceeds max length ({_MAX_TEXT_LENGTH} characters)")

        try:
            client = PlannerClient(
                server.config.planner_url,
                timeout=float(server.config.planner_timeout),
            )
            intent = client.extract_intent(text)
            return intent.model_dump()
        except (PlannerConnectionError, PlannerAPIError) as e:
            return _handle_planner_error(e, server)
