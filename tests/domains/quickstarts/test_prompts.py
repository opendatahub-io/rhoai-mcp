"""Tests for quickstart workflow prompts."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.quickstarts.prompts import register_prompts


@pytest.fixture
def mock_mcp() -> MagicMock:
    """A mock FastMCP that captures @mcp.prompt() registrations."""
    mock = MagicMock()
    registered: dict[str, dict[str, Any]] = {}

    def capture_prompt(name: str | None = None, description: str | None = None) -> Any:
        def decorator(f: Any) -> Any:
            registered[name or f.__name__] = {"function": f, "description": description}
            return f

        return decorator

    mock.prompt = capture_prompt
    mock._registered_prompts = registered
    return mock


class TestQuickstartPrompts:
    def test_prompt_registered_with_description(self, mock_mcp: MagicMock) -> None:
        register_prompts(mock_mcp, MagicMock())

        assert "deploy-quickstart" in mock_mcp._registered_prompts
        description = mock_mcp._registered_prompts["deploy-quickstart"]["description"]
        assert "quickstart" in description.lower()

    def test_prompt_covers_full_workflow(self, mock_mcp: MagicMock) -> None:
        register_prompts(mock_mcp, MagicMock())
        prompt = mock_mcp._registered_prompts["deploy-quickstart"]["function"]

        result = prompt(quickstart="peoplemesh", target_namespace="peoplemesh-quickstart")

        # Every stage of the discover → manifest → params → run → poll loop
        # references its tool so the agent can follow the chain.
        assert "list_quickstarts" in result
        assert "get_quickstart_manifest" in result
        assert "run_quickstart_action" in result
        assert "get_quickstart_action_status" in result
        assert "get_quickstart_action_logs" in result
        assert "peoplemesh" in result

    def test_prompt_handles_unknown_quickstart(self, mock_mcp: MagicMock) -> None:
        register_prompts(mock_mcp, MagicMock())
        prompt = mock_mcp._registered_prompts["deploy-quickstart"]["function"]

        # With no name chosen, the workflow still starts at discovery.
        result = prompt()

        assert "list_quickstarts" in result
        assert "demo" in result
