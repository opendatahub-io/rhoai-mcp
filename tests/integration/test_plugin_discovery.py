"""Integration tests for plugin discovery."""

import pytest


def test_all_plugins_discovered():
    """Verify all expected plugins are discovered via entry points."""
    from rhoai_mcp_core.server import RHOAIServer

    server = RHOAIServer()
    # Call _discover_plugins directly since _plugins is populated during create_mcp()
    plugins = server._discover_plugins()

    expected_plugins = {
        "notebooks",
        "inference",
        "pipelines",
        "connections",
        "storage",
        "projects",
        "training",
    }

    discovered = set(plugins.keys())
    assert expected_plugins.issubset(discovered), (
        f"Missing plugins: {expected_plugins - discovered}"
    )


def test_plugin_metadata():
    """Verify all plugins have valid metadata."""
    from rhoai_mcp_core.server import RHOAIServer

    server = RHOAIServer()
    plugins = server._discover_plugins()

    for name, plugin in plugins.items():
        meta = plugin.metadata
        assert meta.name == name
        assert meta.version
        assert meta.description
        assert meta.maintainer


def test_plugins_can_register():
    """Verify plugins can register tools and resources without error."""
    from unittest.mock import MagicMock

    from rhoai_mcp_core.server import RHOAIServer

    server = RHOAIServer()
    plugins = server._discover_plugins()
    mock_mcp = MagicMock()

    for name, plugin in plugins.items():
        # Should not raise
        plugin.register_tools(mock_mcp, server)
        plugin.register_resources(mock_mcp, server)
