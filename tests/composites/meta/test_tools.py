"""Tests for meta composite tools."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.composites.meta.tools import (
    INTENT_PATTERNS,
    TOOL_CATEGORIES,
    register_tools,
)


@pytest.fixture
def mock_mcp() -> MagicMock:
    """Create a mock FastMCP server that captures tool registrations."""
    mock = MagicMock()
    registered_tools: dict = {}

    def capture_tool():
        def decorator(f):
            registered_tools[f.__name__] = f
            return f

        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    return mock


@pytest.fixture
def mock_server() -> MagicMock:
    """Create a mock RHOAIServer.

    Defaults get_allowed_tools to None (OIDC disabled, all tools allowed).
    """
    server = MagicMock()
    server.get_allowed_tools.return_value = None
    return server


class TestToolCategories:
    """Tests for tool category definitions."""

    def test_categories_defined(self) -> None:
        """All expected categories are defined."""
        expected = ["discovery", "training", "inference", "workbenches", "diagnostics", "resources", "storage"]
        for cat in expected:
            assert cat in TOOL_CATEGORIES

    def test_discovery_has_use_first(self) -> None:
        """Discovery category has use_first flag."""
        assert TOOL_CATEGORIES["discovery"].get("use_first") is True

    def test_all_categories_have_description_and_tools(self) -> None:
        """All categories have description and tools."""
        for name, info in TOOL_CATEGORIES.items():
            assert "description" in info, f"{name} missing description"
            assert "tools" in info, f"{name} missing tools"
            assert len(info["tools"]) > 0, f"{name} has no tools"


class TestIntentPatterns:
    """Tests for intent pattern definitions."""

    def test_patterns_defined(self) -> None:
        """Intent patterns are defined."""
        assert len(INTENT_PATTERNS) > 0

    def test_all_patterns_have_required_fields(self) -> None:
        """All patterns have required fields."""
        for pattern in INTENT_PATTERNS:
            assert "patterns" in pattern
            assert "category" in pattern
            assert "workflow" in pattern
            assert "explanation" in pattern


class TestSuggestTools:
    """Tests for suggest_tools function."""

    def test_tool_registration(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """suggest_tools is registered as a tool."""
        register_tools(mock_mcp, mock_server)
        assert "suggest_tools" in mock_mcp._registered_tools

    def test_suggest_training_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Training intent returns training workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("I want to train a model", None)

        assert result["category"] == "training"
        assert "prepare_training" in result["workflow"]
        assert "train" in result["workflow"]

    def test_suggest_deploy_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Deploy intent returns inference workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("deploy a model for inference", None)

        assert result["category"] == "inference"
        assert "prepare_model_deployment" in result["workflow"]

    def test_suggest_debug_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Debug intent returns diagnostics workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("debug failed job", None)

        assert result["category"] == "diagnostics"
        assert "diagnose_resource" in result["workflow"]

    def test_suggest_explore_intent(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Explore intent returns discovery workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("what's running in the cluster", None)

        assert result["category"] == "discovery"
        assert "explore_cluster" in result["workflow"]

    def test_suggest_with_context(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Context is used in example calls."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("train a model", {"namespace": "my-project"})

        # Check that namespace from context is used
        assert result["example_calls"][0]["args"]["namespace"] == "my-project"

    def test_suggest_unknown_intent_defaults_to_discovery(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Unknown intent falls back to discovery workflow."""
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("completely unrelated gibberish xyz", None)

        assert result["category"] == "discovery"
        assert "explore_cluster" in result["workflow"]


class TestListToolCategories:
    """Tests for list_tool_categories function."""

    def test_tool_registration(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """list_tool_categories is registered as a tool."""
        register_tools(mock_mcp, mock_server)
        assert "list_tool_categories" in mock_mcp._registered_tools

    def test_returns_all_categories(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Returns all defined categories."""
        register_tools(mock_mcp, mock_server)
        list_categories = mock_mcp._registered_tools["list_tool_categories"]

        result = list_categories()

        assert "categories" in result
        assert len(result["categories"]) == len(TOOL_CATEGORIES)

    def test_returns_recommendation(self, mock_mcp: MagicMock, mock_server: MagicMock) -> None:
        """Returns a recommendation."""
        register_tools(mock_mcp, mock_server)
        list_categories = mock_mcp._registered_tools["list_tool_categories"]

        result = list_categories()

        assert "recommendation" in result


class TestSuggestToolsRBACFiltering:
    """Tests that suggest_tools filters results by user RBAC permissions."""

    def test_oidc_disabled_returns_all_tools(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """When OIDC is disabled (None), all workflow tools are returned."""
        mock_server.get_allowed_tools.return_value = None
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("train a model", None)

        assert result["workflow"] == ["prepare_training", "train"]
        assert len(result["example_calls"]) == 2

    def test_filters_workflow_by_rbac(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Only allowed tools appear in workflow when RBAC is active."""
        # User can access prepare_training but not train
        allowed = {"prepare_training"}
        governed = {"prepare_training", "train"}
        mock_server.get_allowed_tools.return_value = (allowed, governed)
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("train a model", None)

        assert result["workflow"] == ["prepare_training"]
        assert len(result["example_calls"]) == 1
        assert result["example_calls"][0]["tool"] == "prepare_training"

    def test_ungoverned_tools_pass_through(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Tools without RBAC mappings (ungoverned) are always included."""
        # explore_cluster has no permission mapping (not governed)
        allowed: set[str] = set()
        governed: set[str] = set()
        mock_server.get_allowed_tools.return_value = (allowed, governed)
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("what's running in the cluster", None)

        assert "explore_cluster" in result["workflow"]

    def test_all_workflow_tools_denied_returns_empty(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """When all workflow tools are denied, workflow and example_calls are empty."""
        allowed: set[str] = set()
        governed = {"prepare_training", "train"}
        mock_server.get_allowed_tools.return_value = (allowed, governed)
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("train a model", None)

        assert result["workflow"] == []
        assert result["example_calls"] == []

    def test_rbac_exception_returns_empty(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """On RBAC check failure, return empty results (fail-closed)."""
        mock_server.get_allowed_tools.side_effect = RuntimeError("K8s API unreachable")
        register_tools(mock_mcp, mock_server)
        suggest_tools = mock_mcp._registered_tools["suggest_tools"]

        result = suggest_tools("train a model", None)

        # Fail-closed: all tools are treated as governed + denied
        assert result["workflow"] == []
        assert result["example_calls"] == []


class TestListToolCategoriesRBACFiltering:
    """Tests that list_tool_categories filters results by user RBAC permissions."""

    def test_oidc_disabled_returns_all_categories(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """When OIDC is disabled, all categories and tools are returned."""
        mock_server.get_allowed_tools.return_value = None
        register_tools(mock_mcp, mock_server)
        list_categories = mock_mcp._registered_tools["list_tool_categories"]

        result = list_categories()

        assert len(result["categories"]) == len(TOOL_CATEGORIES)

    def test_filters_key_tools_by_rbac(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """key_tools are filtered to only include allowed tools."""
        # Allow only list_workbenches, deny create/start/stop/get_workbench_url
        allowed = {"list_workbenches"}
        governed = {
            "list_workbenches", "create_workbench", "start_workbench",
            "stop_workbench", "get_workbench_url",
        }
        mock_server.get_allowed_tools.return_value = (allowed, governed)
        register_tools(mock_mcp, mock_server)
        list_categories = mock_mcp._registered_tools["list_tool_categories"]

        result = list_categories()

        workbench_cat = next(c for c in result["categories"] if c["category"] == "workbenches")
        assert workbench_cat["key_tools"] == ["list_workbenches"]

    def test_omits_categories_with_no_accessible_tools(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Categories where all tools are denied are omitted entirely."""
        # Deny every training tool
        governed = {
            "prepare_training", "train", "get_training_progress",
            "get_training_logs", "analyze_training_failure",
        }
        allowed: set[str] = set()
        mock_server.get_allowed_tools.return_value = (allowed, governed)
        register_tools(mock_mcp, mock_server)
        list_categories = mock_mcp._registered_tools["list_tool_categories"]

        result = list_categories()

        category_names = [c["category"] for c in result["categories"]]
        assert "training" not in category_names
