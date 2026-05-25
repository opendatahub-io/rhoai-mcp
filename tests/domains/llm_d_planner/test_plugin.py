"""Tests for LlmDPlannerPlugin registration."""

from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.domains.registry import LlmDPlannerPlugin, get_core_plugins


class TestLlmDPlannerPlugin:
    """Tests for LlmDPlannerPlugin."""

    def test_plugin_metadata(self) -> None:
        """Plugin has correct metadata."""
        plugin = LlmDPlannerPlugin()
        metadata = plugin.rhoai_get_plugin_metadata()
        assert metadata.name == "llm_d_planner"
        assert metadata.version == "0.1.0"
        assert metadata.requires_crds == []

    def test_plugin_in_core_plugins(self) -> None:
        """LlmDPlannerPlugin is in get_core_plugins list."""
        plugins = get_core_plugins()
        names = [p.rhoai_get_plugin_metadata().name for p in plugins]
        assert "llm_d_planner" in names

    def test_register_tools_calls_register(self) -> None:
        """rhoai_register_tools calls tools.register_tools."""
        plugin = LlmDPlannerPlugin()
        mock_mcp = MagicMock()
        mock_server = MagicMock()

        with patch(
            "rhoai_mcp.domains.llm_d_planner.tools.register_tools"
        ) as mock_reg:
            plugin.rhoai_register_tools(mcp=mock_mcp, server=mock_server)
            mock_reg.assert_called_once_with(mock_mcp, mock_server)

    @patch("rhoai_mcp.domains.llm_d_planner.client.PlannerClient")
    def test_health_check_delegates(self, mock_client_class: MagicMock) -> None:
        """rhoai_health_check delegates to PlannerClient.health_check."""
        mock_client_class.return_value.health_check.return_value = (
            True,
            "llm-d-planner healthy",
        )
        plugin = LlmDPlannerPlugin()
        mock_server = MagicMock()
        mock_server.config.planner_url = "http://localhost:8000"
        mock_server.config.planner_timeout = 120

        ok, msg = plugin.rhoai_health_check(server=mock_server)

        assert ok is True
        assert "healthy" in msg
        mock_client_class.assert_called_once_with(
            "http://localhost:8000", timeout=120
        )


class TestNeuralNavRemoved:
    """Verify NeuralNav is removed from composites."""

    def test_neuralnav_not_in_composites(self) -> None:
        """NeuralNavCompositesPlugin is not in composite plugins."""
        from rhoai_mcp.composites.registry import get_composite_plugins

        plugins = get_composite_plugins()
        names = [p.rhoai_get_plugin_metadata().name for p in plugins]
        assert "neuralnav-composites" not in names

    def test_neuralnav_class_not_importable(self) -> None:
        """NeuralNavCompositesPlugin class no longer exists."""
        with pytest.raises(ImportError):
            from rhoai_mcp.composites.registry import NeuralNavCompositesPlugin  # noqa: F401
