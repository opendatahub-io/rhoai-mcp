"""Integration tests for tool registration across all plugins."""

import re
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from rhoai_mcp.config import RHOAIConfig
from rhoai_mcp.plugin_manager import PluginManager
from rhoai_mcp.server import RHOAIServer

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _capture_all_tools() -> dict[str, object]:
    """Load all plugins and capture every tool function registered via mcp.tool()."""
    pm = PluginManager()
    pm.load_core_plugins()

    mcp = MagicMock()
    registered_tools: dict[str, object] = {}

    def capture_tool(*_args: object, **kwargs: object) -> object:
        tool_name = kwargs.get("name")

        def decorator(func: object) -> object:
            assert callable(func)
            key = str(tool_name) if tool_name else func.__name__
            registered_tools[key] = func
            return func

        return decorator

    mcp.tool = capture_tool

    server = RHOAIServer()
    pm.register_all_tools(mcp, server)
    return registered_tools


@pytest.fixture(scope="module")
def all_tools() -> dict[str, object]:
    """Module-scoped fixture so we only capture tools once."""
    return _capture_all_tools()


def test_all_plugins_register_tools_without_error() -> None:
    """Loading all plugins and calling register_all_tools must not raise."""
    pm = PluginManager()
    pm.load_core_plugins()

    mcp = MagicMock()
    server = RHOAIServer()

    # Should not raise any exception
    pm.register_all_tools(mcp, server)


def test_registered_tool_names_are_snake_case(all_tools: dict[str, object]) -> None:
    """Every registered tool name must be valid snake_case."""
    violations = [name for name in all_tools if not SNAKE_CASE_RE.match(name)]
    assert not violations, f"Tool names not in snake_case: {violations}"


def test_no_tool_name_collisions() -> None:
    """All registered tool names must be unique across all plugins."""
    pm = PluginManager()
    pm.load_core_plugins()

    mcp = MagicMock()
    seen_names: list[str] = []

    def capture_tool(*_args: object, **kwargs: object) -> object:
        tool_name = kwargs.get("name")

        def decorator(func: object) -> object:
            assert callable(func)
            seen_names.append(str(tool_name) if tool_name else func.__name__)
            return func

        return decorator

    mcp.tool = capture_tool

    server = RHOAIServer()
    pm.register_all_tools(mcp, server)

    duplicates = [name for name in seen_names if seen_names.count(name) > 1]
    assert not duplicates, f"Duplicate tool names detected: {set(duplicates)}"


def test_all_tools_have_descriptions(all_tools: dict[str, object]) -> None:
    """Every registered tool function must have a non-empty docstring."""
    missing_docs = [
        name
        for name, func in all_tools.items()
        if not getattr(func, "__doc__", None) or not getattr(func, "__doc__", "").strip()
    ]
    assert not missing_docs, f"Tools without docstrings: {missing_docs}"


def test_minimum_tool_count(all_tools: dict[str, object]) -> None:
    """Sanity check: at least 40 tools must be registered across all plugins."""
    assert len(all_tools) >= 40, (
        f"Expected at least 40 tools, got {len(all_tools)}: {sorted(all_tools.keys())}"
    )


@pytest.mark.parametrize(
    ("plugins", "expected"),
    [
        (RHOAIConfig.model_fields["enabled_plugins"].get_default(), True),
        (["cluster-composites"], True),
        (["training"], False),
    ],
)
@pytest.mark.asyncio
async def test_cluster_resources_registration_and_permissions(
    plugins: list[str] | None, expected: bool
) -> None:
    """Only the cluster plugin exposes and governs the inventory tool."""
    pm = PluginManager()
    pm.load_core_plugins(enabled_plugins=plugins)
    mcp = FastMCP("test")
    pm.register_all_tools(mcp, RHOAIServer())
    names = [tool.name for tool in await mcp.list_tools()]
    permissions = pm.collect_tool_permissions()
    assert names.count("get_cluster_resources") == int(expected)
    assert ("get_cluster_resources" in permissions) is expected
    if expected:
        assert permissions["get_cluster_resources"] == [
            {"apiGroup": "", "resource": "nodes", "verb": "list"}
        ]
