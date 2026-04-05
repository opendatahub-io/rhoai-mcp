"""Tests for tool permission declarations."""

from rhoai_mcp.composites.registry import get_composite_plugins
from rhoai_mcp.domains.registry import get_core_plugins


class TestToolPermissionDeclarations:
    def test_all_domain_plugins_declare_permissions(self):
        """Every plugin that registers tools should declare permissions."""
        for plugin in get_core_plugins():
            meta = plugin.rhoai_get_plugin_metadata()
            perms = plugin.rhoai_get_tool_permissions()
            # Prompts-only plugins (no tools) may return empty
            if meta.name == "prompts":
                assert perms == {}
                continue
            # Model registry uses REST API, not K8s RBAC
            if meta.name == "model_registry":
                assert perms == {}
                continue
            # All other plugins should have at least one tool mapped
            assert isinstance(perms, dict), f"Plugin {meta.name} returned non-dict"
            assert len(perms) > 0, f"Plugin {meta.name} has no tool permissions"

    def test_all_composite_plugins_declare_permissions(self):
        """Every composite plugin that registers tools should declare permissions."""
        for plugin in get_composite_plugins():
            meta = plugin.rhoai_get_plugin_metadata()
            perms = plugin.rhoai_get_tool_permissions()
            assert isinstance(perms, dict), f"Plugin {meta.name} returned non-dict"
            # Meta and NeuralNav composites don't access K8s resources
            if meta.name in ("meta-composites", "neuralnav-composites"):
                assert perms == {}, f"Plugin {meta.name} should return empty dict"
                continue
            assert len(perms) > 0, f"Plugin {meta.name} has no tool permissions"

    def test_permission_dicts_have_required_keys(self):
        """All permission dicts must have apiGroup, resource, verb."""
        all_plugins = get_core_plugins() + get_composite_plugins()
        for plugin in all_plugins:
            meta = plugin.rhoai_get_plugin_metadata()
            perms = plugin.rhoai_get_tool_permissions()
            for tool_name, perm_list in perms.items():
                assert isinstance(perm_list, list), (
                    f"{meta.name}/{tool_name} permissions must be a list"
                )
                for perm in perm_list:
                    assert "apiGroup" in perm, f"{meta.name}/{tool_name} missing apiGroup"
                    assert "resource" in perm, f"{meta.name}/{tool_name} missing resource"
                    assert "verb" in perm, f"{meta.name}/{tool_name} missing verb"

    def test_no_duplicate_tool_names_across_plugins(self):
        """Tool names must be unique across all plugins."""
        all_plugins = get_core_plugins() + get_composite_plugins()
        seen: dict[str, str] = {}
        for plugin in all_plugins:
            meta = plugin.rhoai_get_plugin_metadata()
            perms = plugin.rhoai_get_tool_permissions()
            for tool_name in perms:
                assert tool_name not in seen, (
                    f"Duplicate tool '{tool_name}' in {meta.name} and {seen[tool_name]}"
                )
                seen[tool_name] = meta.name

    def test_verb_values_are_valid_k8s_verbs(self):
        """All verb values should be valid Kubernetes API verbs."""
        valid_verbs = {"get", "list", "create", "update", "patch", "delete", "watch"}
        all_plugins = get_core_plugins() + get_composite_plugins()
        for plugin in all_plugins:
            meta = plugin.rhoai_get_plugin_metadata()
            perms = plugin.rhoai_get_tool_permissions()
            for tool_name, perm_list in perms.items():
                for perm in perm_list:
                    assert perm["verb"] in valid_verbs, (
                        f"{meta.name}/{tool_name} has invalid verb '{perm['verb']}'"
                    )
