"""Tests for projects MCP tools."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.domains.projects.models import DataScienceProject
from rhoai_mcp.models.common import ResourceMetadata, ResourceStatus
from rhoai_mcp.utils.errors import NotManagedByMCPError


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register project tools and return captured tool functions."""
    from rhoai_mcp.domains.projects.tools import register_tools

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
    """Create a mock server with config defaults."""
    server = MagicMock()
    server.config.is_operation_allowed.return_value = (True, None)
    server.config.read_only_mode = False
    server.config.max_list_limit = 100
    server.config.default_list_limit = None
    return server


def _make_project(
    name: str = "test-project",
    display_name: str | None = "Test Project",
    description: str | None = "A test project",
    modelmesh: bool = False,
) -> DataScienceProject:
    """Build a DataScienceProject model instance for testing."""
    return DataScienceProject(
        metadata=ResourceMetadata(
            name=name,
            namespace=None,
            uid="test-uid",
            kind="Project",
            api_version="project.openshift.io/v1",
        ),
        display_name=display_name,
        description=description,
        is_modelmesh_enabled=modelmesh,
        status=ResourceStatus.READY,
    )


class TestToolRegistration:
    """Verify all expected tools are registered."""

    def test_tools_registered(self, mock_server: MagicMock) -> None:
        """Should register all 5 project management tools."""
        tools = _register_tools(mock_server)

        expected = {
            "list_data_science_projects",
            "get_project_details",
            "create_data_science_project",
            "delete_data_science_project",
            "set_model_serving_mode",
        }
        assert set(tools.keys()) == expected


class TestListDataScienceProjects:
    """Tests for list_data_science_projects tool."""

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_list_data_science_projects_calls_client(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should create a ProjectClient and call list_projects."""
        mock_client = mock_client_cls.return_value
        mock_client.list_projects.return_value = [_make_project()]

        tools = _register_tools(mock_server)
        result = tools["list_data_science_projects"]()

        mock_client_cls.assert_called_once_with(mock_server.k8s)
        mock_client.list_projects.assert_called_once()
        assert result["total"] == 1
        assert len(result["items"]) == 1


class TestGetProjectDetails:
    """Tests for get_project_details tool."""

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_get_project_details_calls_client(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should create a ProjectClient and call get_project."""
        project = _make_project(name="my-project")
        mock_client = mock_client_cls.return_value
        mock_client.get_project.return_value = project

        tools = _register_tools(mock_server)
        result = tools["get_project_details"](name="my-project")

        mock_client_cls.assert_called_once_with(mock_server.k8s)
        mock_client.get_project.assert_called_once_with("my-project", include_summary=True)
        assert result["name"] == "my-project"


class TestCreateDataScienceProject:
    """Tests for create_data_science_project tool."""

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_create_project_read_only_blocked(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should return error and not instantiate client when create is not allowed."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Read-only mode",
        )

        tools = _register_tools(mock_server)
        result = tools["create_data_science_project"](name="new-project")

        assert "error" in result
        assert result["error"] == "Read-only mode"
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_create_project_success(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should call create_project when operation is allowed."""
        project = _make_project(name="new-project", display_name="New Project")
        mock_client = mock_client_cls.return_value
        mock_client.create_project.return_value = project

        tools = _register_tools(mock_server)
        result = tools["create_data_science_project"](
            name="new-project",
            display_name="New Project",
            description="desc",
        )

        mock_client.create_project.assert_called_once()
        assert result["name"] == "new-project"
        assert result["display_name"] == "New Project"
        assert "message" in result
        assert "created successfully" in result["message"]


class TestDeleteDataScienceProject:
    """Tests for delete_data_science_project tool."""

    def test_delete_project_read_only_blocked(self, mock_server: MagicMock) -> None:
        """Should return error when read-only mode is enabled."""
        mock_server.config.read_only_mode = True

        tools = _register_tools(mock_server)
        result = tools["delete_data_science_project"](name="some-project", confirm=True)

        assert "error" in result
        assert "Read-only mode is enabled" in result["error"]

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_delete_project_not_managed_blocked(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should return error when resource is not managed by MCP."""
        mock_client = mock_client_cls.return_value
        mock_client.delete_project.side_effect = NotManagedByMCPError(
            "Namespace", "some-project"
        )

        tools = _register_tools(mock_server)
        result = tools["delete_data_science_project"](name="some-project", confirm=True)

        assert "error" in result
        assert "not created by this MCP server" in result["error"]

    def test_delete_project_no_confirm(self, mock_server: MagicMock) -> None:
        """Should return error when confirm is False."""
        tools = _register_tools(mock_server)
        result = tools["delete_data_science_project"](name="some-project", confirm=False)

        assert "error" in result
        assert "not confirmed" in result["error"]

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_delete_project_success(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should call delete_project when confirmed and allowed."""
        mock_client = mock_client_cls.return_value

        tools = _register_tools(mock_server)
        result = tools["delete_data_science_project"](name="doomed-project", confirm=True)

        mock_client.delete_project.assert_called_once_with("doomed-project")
        assert result["deleted"] is True
        assert result["name"] == "doomed-project"


class TestSetModelServingMode:
    """Tests for set_model_serving_mode tool."""

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_set_model_serving_mode_read_only_blocked(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should return error and not instantiate client when update is not allowed."""
        mock_server.config.is_operation_allowed.return_value = (
            False,
            "Read-only mode",
        )

        tools = _register_tools(mock_server)
        result = tools["set_model_serving_mode"](name="my-project", enable_modelmesh=True)

        assert "error" in result
        assert result["error"] == "Read-only mode"
        mock_client_cls.assert_not_called()

    @patch("rhoai_mcp.domains.projects.tools.ProjectClient")
    def test_set_model_serving_mode_success(
        self, mock_client_cls: MagicMock, mock_server: MagicMock
    ) -> None:
        """Should call set_model_serving_mode when update is allowed."""
        project = _make_project(name="my-project", modelmesh=True)
        mock_client = mock_client_cls.return_value
        mock_client.set_model_serving_mode.return_value = project

        tools = _register_tools(mock_server)
        result = tools["set_model_serving_mode"](name="my-project", enable_modelmesh=True)

        mock_client.set_model_serving_mode.assert_called_once_with("my-project", True)
        assert result["name"] == "my-project"
        assert result["is_modelmesh_enabled"] is True
        assert "multi-model" in result["message"]
