"""Tests for Planner plugin registration."""

from unittest.mock import MagicMock, patch

from rhoai_mcp.composites.registry import PlannerCompositesPlugin, get_composite_plugins


class TestPlannerPlugin:
    """Tests for PlannerCompositesPlugin."""

    def test_plugin_in_registry(self) -> None:
        """Planner plugin is included in composite plugins."""
        plugins = get_composite_plugins()
        names = [p.rhoai_get_plugin_metadata().name for p in plugins]
        assert "planner-composites" in names

    def test_plugin_metadata(self) -> None:
        """Plugin metadata is correct."""
        plugin = PlannerCompositesPlugin()
        meta = plugin.rhoai_get_plugin_metadata()
        assert meta.name == "planner-composites"
        assert meta.requires_crds == []

    def test_register_tools(self) -> None:
        """Plugin registers tools with MCP."""
        plugin = PlannerCompositesPlugin()
        mock_mcp = MagicMock()
        mock_server = MagicMock()

        with patch("rhoai_mcp.composites.planner.tools.register_tools") as mock_reg:
            plugin.rhoai_register_tools(mock_mcp, mock_server)
            mock_reg.assert_called_once_with(mock_mcp, mock_server)

    @patch("rhoai_mcp.composites.planner.client.PlannerClient")
    def test_health_check_healthy(self, mock_client_class: MagicMock) -> None:
        """Health check delegates to PlannerClient."""
        mock_client_class.return_value.health_check.return_value = (
            True,
            "Planner available",
        )
        plugin = PlannerCompositesPlugin()
        mock_server = MagicMock()
        mock_server.config.planner_url = "http://localhost:8000"
        mock_server.config.planner_timeout = 120

        healthy, msg = plugin.rhoai_health_check(mock_server)

        assert healthy is True
        mock_client_class.assert_called_once_with(
            "http://localhost:8000",
            timeout=120,
        )

    @patch("rhoai_mcp.composites.planner.client.PlannerClient")
    def test_health_check_unhealthy(self, mock_client_class: MagicMock) -> None:
        """Health check returns False when Planner is unreachable."""
        mock_client_class.return_value.health_check.return_value = (
            False,
            "Planner unavailable: Connection refused",
        )
        plugin = PlannerCompositesPlugin()
        mock_server = MagicMock()
        mock_server.config.planner_url = "http://localhost:8000"
        mock_server.config.planner_timeout = 120

        healthy, msg = plugin.rhoai_health_check(mock_server)

        assert healthy is False
        mock_client_class.assert_called_once_with(
            "http://localhost:8000",
            timeout=120,
        )
