"""Tests for NeuralNav composite tools."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.composites.neuralnav.tools import _base_url, register_tools


class TestBaseUrl:
    """Tests for _base_url helper."""

    def test_returns_url_stripped(self) -> None:
        """Base URL is stripped of trailing slash."""
        server = MagicMock()
        server.config.neuralnav_backend_url = "http://localhost:8000/"
        assert _base_url(server) == "http://localhost:8000"

    def test_returns_none_when_unset(self) -> None:
        """Returns None when URL is not configured."""
        server = MagicMock()
        server.config.neuralnav_backend_url = None
        assert _base_url(server) is None

    def test_returns_none_when_empty(self) -> None:
        """Returns None when URL is empty string."""
        server = MagicMock()
        server.config.neuralnav_backend_url = ""
        assert _base_url(server) is None


class TestGetDeploymentRecommendation:
    """Tests for get_deployment_recommendation tool."""

    @pytest.fixture
    def mock_mcp(self) -> MagicMock:
        """Create a mock MCP server that captures tool registrations."""
        mock = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mock.tool = capture_tool
        mock._registered_tools = registered_tools
        return mock

    @pytest.fixture
    def mock_server_no_url(self) -> MagicMock:
        """Server with no NeuralNav URL configured."""
        server = MagicMock()
        server.config.neuralnav_backend_url = None
        return server

    @pytest.fixture
    def mock_server_with_url(self) -> MagicMock:
        """Server with NeuralNav URL configured."""
        server = MagicMock()
        server.config.neuralnav_backend_url = "http://localhost:8000"
        return server

    def test_returns_error_when_url_not_configured(
        self, mock_mcp: MagicMock, mock_server_no_url: MagicMock
    ) -> None:
        """Returns error dict when NeuralNav URL is not set."""
        register_tools(mock_mcp, mock_server_no_url)
        tool = mock_mcp._registered_tools["get_deployment_recommendation"]
        result = tool(
            use_case="chatbot_conversational",
            user_count=10,
        )
        assert "error" in result
        assert "not configured" in result["error"].lower()
        assert "hint" in result

    def test_success_returns_ranked_views(
        self, mock_mcp: MagicMock, mock_server_with_url: MagicMock
    ) -> None:
        """Successful call returns balanced, best_accuracy, etc."""
        register_tools(mock_mcp, mock_server_with_url)
        tool = mock_mcp._registered_tools["get_deployment_recommendation"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "balanced": [{"model_name": "test-model", "gpu_config": {}}],
            "best_accuracy": [],
            "lowest_cost": [],
            "lowest_latency": [],
            "simplest": [],
            "specification": {},
            "stats": {},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("rhoai_mcp.composites.neuralnav.tools.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_httpx.Client.return_value = mock_client

            result = tool(
                use_case="chatbot_conversational",
                user_count=10,
            )

        assert "balanced" in result
        assert len(result["balanced"]) == 1
        assert result["balanced"][0]["model_name"] == "test-model"
        assert "recommended_agent" not in result
        assert "recommended_system_prompt" not in result

    def test_strips_agent_prompt_fields(
        self, mock_mcp: MagicMock, mock_server_with_url: MagicMock
    ) -> None:
        """Response strips recommended_agent, recommended_system_prompt, etc."""
        register_tools(mock_mcp, mock_server_with_url)
        tool = mock_mcp._registered_tools["get_deployment_recommendation"]

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "balanced": [],
            "best_accuracy": [],
            "lowest_cost": [],
            "lowest_latency": [],
            "simplest": [],
            "recommended_agent": "some-agent",
            "recommended_system_prompt": "You are helpful",
            "prompt_eval_dataset_path": "/path",
            "recommended_agent_explanation": "explanation",
            "specification": {},
            "stats": {},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("rhoai_mcp.composites.neuralnav.tools.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_httpx.Client.return_value = mock_client

            result = tool(
                use_case="chatbot_conversational",
                user_count=10,
            )

        assert "recommended_agent" not in result
        assert "recommended_system_prompt" not in result
        assert "prompt_eval_dataset_path" not in result
        assert "recommended_agent_explanation" not in result
