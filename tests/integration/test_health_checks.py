"""Integration tests for plugin health checks."""

from rhoai_mcp.plugin_manager import PluginManager
from rhoai_mcp.server import RHOAIServer

ALL_PLUGIN_NAMES = {
    # Domain plugins (10)
    "projects",
    "notebooks",
    "inference",
    "pipelines",
    "connections",
    "storage",
    "training",
    "prompts",
    "model_registry",
    # Composite plugins (4)
    "cluster-composites",
    "training-composites",
    "meta-composites",
    "planner-composites",
}

# Plugins whose health checks always return True (no external dependencies)
ALWAYS_HEALTHY_PLUGINS = {
    "projects",
    "connections",
    "storage",
    "prompts",
}


def test_all_plugins_respond_to_health_check() -> None:
    """Every loaded plugin must return a health check result."""
    pm = PluginManager()
    pm.load_core_plugins()
    server = RHOAIServer()

    results = pm.run_health_checks(server)

    assert set(results.keys()) == ALL_PLUGIN_NAMES, (
        f"Missing health check entries for: {ALL_PLUGIN_NAMES - set(results.keys())}"
    )
    # Every result must be a (bool, str) tuple
    for name, (healthy, message) in results.items():
        assert isinstance(healthy, bool), f"Plugin {name}: healthy must be bool"
        assert isinstance(message, str), f"Plugin {name}: message must be str"
        assert message, f"Plugin {name}: health check message must not be empty"


def test_plugins_without_external_deps_return_healthy() -> None:
    """Plugins that have no external dependencies must always report healthy."""
    pm = PluginManager()
    pm.load_core_plugins()
    server = RHOAIServer()

    results = pm.run_health_checks(server)

    for name in ALWAYS_HEALTHY_PLUGINS:
        healthy, message = results[name]
        assert healthy, f"Plugin {name} should be healthy but returned: {message}"


def test_model_registry_unhealthy_when_disabled() -> None:
    """Model registry must report unhealthy when model_registry_enabled is False."""
    pm = PluginManager()
    pm.load_core_plugins()

    server = RHOAIServer()
    object.__setattr__(server.config, "model_registry_enabled", False)

    results = pm.run_health_checks(server)

    healthy, message = results["model_registry"]
    assert not healthy, f"model_registry should be unhealthy when disabled: {message}"
    assert "disabled" in message.lower(), f"Expected 'disabled' in message, got: {message}"
