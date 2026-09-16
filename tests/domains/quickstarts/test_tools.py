"""Tests for quickstart MCP tools."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register quickstart tools and return captured tool functions."""
    from rhoai_mcp.domains.quickstarts.tools import register_tools

    mcp = MagicMock()
    registered_tools: dict[str, Any] = {}

    def capture_tool() -> Any:
        def decorator(func: Any) -> Any:
            registered_tools[func.__name__] = func
            return func

        return decorator

    mcp.tool = capture_tool
    register_tools(mcp, mock_server)
    return registered_tools


@pytest.fixture
def mock_server() -> MagicMock:
    server = MagicMock()
    server.config.is_operation_allowed.return_value = (True, None)
    server.config.read_only_mode = False
    server.config.enable_dangerous_operations = True
    server.config.max_list_limit = 100
    server.config.default_list_limit = None
    server.config.quickstart_job_namespace = "openshift-quickstarts"
    return server


class TestToolRegistration:
    def test_tools_registered(self, mock_server: MagicMock) -> None:
        tools = _register_tools(mock_server)
        assert set(tools.keys()) == {
            "list_quickstarts",
            "get_quickstart_manifest",
            "run_quickstart_action",
            "get_quickstart_action_status",
            "get_quickstart_action_logs",
        }


class TestListQuickstarts:
    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_list_returns_paginated(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        summary = MagicMock()
        summary.to_dict.return_value = {"name": "peoplemesh"}
        mock_client = MagicMock()
        mock_client.get_registry.return_value.quickstarts = [summary]
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["list_quickstarts"]()

        mock_client_cls.assert_called_once_with(mock_server.k8s, mock_server.config)
        assert result["total"] == 1
        assert result["items"][0]["name"] == "peoplemesh"


class TestGetManifest:
    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_get_manifest_shapes_response(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        from rhoai_mcp.domains.quickstarts.models import QuickstartManifest
        from tests.domains.quickstarts.conftest import MANIFEST_YAML

        mock_client = MagicMock()
        mock_client.get_manifest.return_value = QuickstartManifest.from_yaml(MANIFEST_YAML)
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["get_quickstart_manifest"](name="peoplemesh")

        assert result["name"] == "peoplemesh"
        assert result["supported_actions"] == [
            "CHECK_PRE_REQS",
            "STATUS",
            "INSTALL",
            "UNINSTALL_DELETE_ALL",
        ]
        secrets = result["parameters"]["secrets"]
        assert secrets[0]["llm_guidance"].startswith("Always ask")
        assert result["llm_context"] == {"whenToRecommend": "Recommend for talent discovery."}


class TestRunAction:
    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_read_only_blocked(self, mock_client_cls: MagicMock, mock_server: MagicMock) -> None:
        mock_server.config.is_operation_allowed.return_value = (False, "Read-only mode is enabled")

        tools = _register_tools(mock_server)
        result = tools["run_quickstart_action"](name="peoplemesh", action="INSTALL")

        assert result == {"error": "Read-only mode is enabled"}
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_destructive_requires_dangerous_ops(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        mock_server.config.enable_dangerous_operations = False

        tools = _register_tools(mock_server)
        result = tools["run_quickstart_action"](
            name="peoplemesh", action="UNINSTALL_DELETE_ALL", confirm=True
        )

        assert result["error"] == "Dangerous operations are disabled"
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_destructive_requires_confirm(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        tools = _register_tools(mock_server)
        result = tools["run_quickstart_action"](
            name="peoplemesh", action="UNINSTALL_DELETE_ALL", confirm=False
        )

        assert result["error"] == "Action not confirmed"
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_success_passes_through(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.run_action.return_value = {"job_name": "qs-x", "action": "INSTALL"}
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["run_quickstart_action"](
            name="peoplemesh",
            action="INSTALL",
            parameters={"keycloak.realm.testUser.password": "x"},
        )

        assert result == {"job_name": "qs-x", "action": "INSTALL"}
        mock_client.run_action.assert_called_once()

    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_validation_error_returned_as_dict(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        from rhoai_mcp.utils.errors import ValidationError

        mock_client = MagicMock()
        mock_client.run_action.side_effect = ValidationError("missing required parameter(s): p")
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["run_quickstart_action"](name="peoplemesh", action="INSTALL")

        assert "error" in result
        assert "missing required" in result["error"]


class TestStatusAndLogs:
    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_status_defaults_namespace(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.get_action_status.return_value = {"phase": "Running"}
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        result = tools["get_quickstart_action_status"](job_name="qs-x")

        mock_client.get_action_status.assert_called_once_with("qs-x", "openshift-quickstarts")
        assert result["phase"] == "Running"

    @patch("rhoai_mcp.domains.quickstarts.tools.QuickstartsClient")
    def test_logs_defaults_namespace(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_client.get_action_logs.return_value = {"logs": "..."}
        mock_client_cls.return_value = mock_client

        tools = _register_tools(mock_server)
        tools["get_quickstart_action_logs"](job_name="qs-x")

        mock_client.get_action_logs.assert_called_once_with("qs-x", "openshift-quickstarts", 200)
