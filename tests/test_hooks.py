"""Unit tests for hook specifications."""

import pluggy

from rhoai_mcp.hooks import PROJECT_NAME, RHOAIMCPHookSpec, hookimpl
from rhoai_mcp.plugin import BasePlugin, PluginMetadata


class TestHookSpec:
    """Tests for RHOAIMCPHookSpec."""

    def test_project_name_defined(self) -> None:
        """Verify project name is defined correctly."""
        assert PROJECT_NAME == "rhoai_mcp"

    def test_hookspec_has_required_methods(self) -> None:
        """Verify hookspec defines all required hook methods."""
        spec = RHOAIMCPHookSpec()

        # Check all expected hooks exist
        assert hasattr(spec, "rhoai_get_plugin_metadata")
        assert hasattr(spec, "rhoai_register_tools")
        assert hasattr(spec, "rhoai_register_resources")
        assert hasattr(spec, "rhoai_register_prompts")
        assert hasattr(spec, "rhoai_get_crd_definitions")
        assert hasattr(spec, "rhoai_health_check")
        assert hasattr(spec, "rhoai_get_tool_permissions")

    def test_hookspec_can_be_added_to_pluggy(self) -> None:
        """Verify hookspec can be registered with pluggy."""
        pm = pluggy.PluginManager(PROJECT_NAME)
        pm.add_hookspecs(RHOAIMCPHookSpec)

        # Should have the hook caller
        assert hasattr(pm.hook, "rhoai_get_plugin_metadata")
        assert hasattr(pm.hook, "rhoai_register_tools")
        assert hasattr(pm.hook, "rhoai_register_resources")
        assert hasattr(pm.hook, "rhoai_register_prompts")
        assert hasattr(pm.hook, "rhoai_get_crd_definitions")
        assert hasattr(pm.hook, "rhoai_health_check")
        assert hasattr(pm.hook, "rhoai_get_tool_permissions")


class TestHookImpl:
    """Tests for hook implementations."""

    def test_hookimpl_decorator_works(self) -> None:
        """Verify hookimpl decorator can be used on methods."""

        class TestPlugin:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> dict:
                return {"name": "test"}

        plugin = TestPlugin()
        assert hasattr(plugin.rhoai_get_plugin_metadata, "rhoai_mcp_impl")

    def test_plugin_with_hookimpl_can_register(self) -> None:
        """Verify plugin with hookimpl decorators can be registered."""
        from rhoai_mcp.plugin import PluginMetadata

        class TestPlugin:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="test",
                    version="1.0.0",
                    description="Test plugin",
                    maintainer="test@example.com",
                )

        pm = pluggy.PluginManager(PROJECT_NAME)
        pm.add_hookspecs(RHOAIMCPHookSpec)
        pm.register(TestPlugin())

        results = pm.hook.rhoai_get_plugin_metadata()
        assert len(results) == 1
        assert results[0].name == "test"

    def test_multiple_plugins_can_register(self) -> None:
        """Verify multiple plugins can be registered and hooks called."""
        from rhoai_mcp.plugin import PluginMetadata

        class PluginA:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="plugin_a",
                    version="1.0.0",
                    description="Plugin A",
                    maintainer="test@example.com",
                )

        class PluginB:
            @hookimpl
            def rhoai_get_plugin_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="plugin_b",
                    version="2.0.0",
                    description="Plugin B",
                    maintainer="test@example.com",
                )

        pm = pluggy.PluginManager(PROJECT_NAME)
        pm.add_hookspecs(RHOAIMCPHookSpec)
        pm.register(PluginA())
        pm.register(PluginB())

        results = pm.hook.rhoai_get_plugin_metadata()
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"plugin_a", "plugin_b"}


class TestToolPermissionsHook:
    """Tests for the rhoai_get_tool_permissions hook."""

    def test_base_plugin_returns_empty_permissions(self) -> None:
        plugin = BasePlugin(
            PluginMetadata(name="test", version="0.1.0", description="test", maintainer="test")
        )
        result = plugin.rhoai_get_tool_permissions()
        assert result == {}

    def test_plugin_manager_collects_permissions(self) -> None:
        from rhoai_mcp.plugin_manager import PluginManager

        class TestPlugin(BasePlugin):
            @hookimpl
            def rhoai_get_tool_permissions(self) -> dict[str, list[dict[str, str]]]:
                return {
                    "test_tool": [{"apiGroup": "", "resource": "pods", "verb": "list"}],
                }

        pm = PluginManager()
        plugin = TestPlugin(
            PluginMetadata(name="test", version="0.1.0", description="test", maintainer="test")
        )
        pm.register_plugin(plugin, "test")

        result = pm.collect_tool_permissions()
        assert "test_tool" in result
        assert len(result["test_tool"]) == 1
        assert result["test_tool"][0]["resource"] == "pods"

    def test_plugin_manager_merges_permissions_from_multiple_plugins(self) -> None:
        from rhoai_mcp.plugin_manager import PluginManager

        class PluginA(BasePlugin):
            @hookimpl
            def rhoai_get_tool_permissions(self) -> dict[str, list[dict[str, str]]]:
                return {"tool_a": [{"apiGroup": "", "resource": "pods", "verb": "list"}]}

        class PluginB(BasePlugin):
            @hookimpl
            def rhoai_get_tool_permissions(self) -> dict[str, list[dict[str, str]]]:
                return {"tool_b": [{"apiGroup": "", "resource": "secrets", "verb": "get"}]}

        pm = PluginManager()
        pm.register_plugin(
            PluginA(
                PluginMetadata(name="a", version="0.1.0", description="a", maintainer="a")
            ),
            "a",
        )
        pm.register_plugin(
            PluginB(
                PluginMetadata(name="b", version="0.1.0", description="b", maintainer="b")
            ),
            "b",
        )

        result = pm.collect_tool_permissions()
        assert "tool_a" in result
        assert "tool_b" in result
