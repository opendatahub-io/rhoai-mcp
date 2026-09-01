"""Tests for Planner prompt registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from rhoai_mcp.composites.planner.prompts import register_prompts


def _make_mock_mcp() -> MagicMock:
    """Create a mock FastMCP that captures prompt registrations."""
    mock = MagicMock()
    registered_prompts: dict = {}

    def capture_prompt(**kwargs):
        def decorator(f):
            registered_prompts[kwargs.get("name", f.__name__)] = {
                "func": f,
                "kwargs": kwargs,
            }
            return f

        return decorator

    mock.prompt = capture_prompt
    mock._registered_prompts = registered_prompts
    return mock


class TestRegisterPrompts:
    """Tests for register_prompts."""

    def test_recommend_and_deploy_registered(self) -> None:
        """recommend-and-deploy prompt is registered."""
        mock_mcp = _make_mock_mcp()
        register_prompts(mock_mcp, MagicMock())
        assert "recommend-and-deploy" in mock_mcp._registered_prompts

    def test_prompt_description(self) -> None:
        """Prompt has a description."""
        mock_mcp = _make_mock_mcp()
        register_prompts(mock_mcp, MagicMock())
        prompt = mock_mcp._registered_prompts["recommend-and-deploy"]
        assert "description" in prompt["kwargs"]
        assert prompt["kwargs"]["description"]

    def test_prompt_returns_workflow_guidance(self) -> None:
        """Prompt function returns string with workflow steps."""
        mock_mcp = _make_mock_mcp()
        register_prompts(mock_mcp, MagicMock())
        func = mock_mcp._registered_prompts["recommend-and-deploy"]["func"]

        result = func(
            description="chatbot for 1000 users",
            namespace="production",
            priority="balanced",
        )

        assert isinstance(result, str)
        assert "chatbot for 1000 users" in result
        assert "production" in result
        assert "balanced" in result
        assert "recommend_model" in result
        assert "plan_deployment" in result
        assert "execute_deployment" in result

    def test_prompt_includes_all_four_steps(self) -> None:
        """Prompt includes all four workflow steps."""
        mock_mcp = _make_mock_mcp()
        register_prompts(mock_mcp, MagicMock())
        func = mock_mcp._registered_prompts["recommend-and-deploy"]["func"]

        result = func(description="test", namespace="ns", priority="cost")

        assert "Get Recommendations" in result
        assert "Plan the Deployment" in result
        assert "Deploy and Validate" in result
        assert "Report Results" in result
