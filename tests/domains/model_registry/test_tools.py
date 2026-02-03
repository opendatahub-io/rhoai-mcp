"""Tests for Model Registry MCP tools."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rhoai_mcp.domains.model_registry.models import (
    CustomProperties,
    ModelArtifact,
    ModelVersion,
    RegisteredModel,
)


class TestListRegisteredModels:
    """Test list_registered_models tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        server.config.default_list_limit = None
        server.config.max_list_limit = 100
        return server

    def test_list_models_disabled(self, mock_server: MagicMock) -> None:
        """Test listing when registry is disabled."""
        mock_server.config.model_registry_enabled = False

        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        result = registered_tools["list_registered_models"]()
        assert "error" in result
        assert "disabled" in result["error"]

    def test_list_models_success(self, mock_server: MagicMock) -> None:
        """Test successful model listing."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        models = [
            RegisteredModel(id="model-1", name="model-one", state="LIVE"),
            RegisteredModel(id="model-2", name="model-two", state="LIVE"),
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_registered_models = AsyncMock(return_value=models)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = registered_tools["list_registered_models"]()

        assert "items" in result
        assert result["total"] == 2
        assert len(result["items"]) == 2

    def test_list_models_with_pagination(self, mock_server: MagicMock) -> None:
        """Test model listing with pagination."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        models = [
            RegisteredModel(id=f"model-{i}", name=f"model-{i}", state="LIVE")
            for i in range(10)
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_registered_models = AsyncMock(return_value=models)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = registered_tools["list_registered_models"](limit=5, offset=2)

        assert result["total"] == 10
        assert len(result["items"]) == 5
        assert result["offset"] == 2


class TestGetRegisteredModel:
    """Test get_registered_model tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        return server

    def test_get_model_success(self, mock_server: MagicMock) -> None:
        """Test getting a model successfully."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        model = RegisteredModel(
            id="model-123",
            name="llama-2-7b",
            description="Fine-tuned model",
            owner="data-team",
            state="LIVE",
        )

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model = AsyncMock(return_value=model)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = registered_tools["get_registered_model"]("model-123")

        assert result["id"] == "model-123"
        assert result["name"] == "llama-2-7b"
        assert result["owner"] == "data-team"

    def test_get_model_with_versions(self, mock_server: MagicMock) -> None:
        """Test getting a model with versions."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        model = RegisteredModel(id="model-123", name="test-model", state="LIVE")
        versions = [
            ModelVersion(id="v1", name="1.0", registered_model_id="model-123"),
            ModelVersion(id="v2", name="2.0", registered_model_id="model-123"),
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model = AsyncMock(return_value=model)
            mock_client.get_model_versions = AsyncMock(return_value=versions)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = registered_tools["get_registered_model"](
                "model-123", include_versions=True
            )

        assert "versions" in result
        assert len(result["versions"]) == 2

    def test_get_model_not_found(self, mock_server: MagicMock) -> None:
        """Test getting a model that doesn't exist."""
        from rhoai_mcp.domains.model_registry.errors import ModelNotFoundError
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model = AsyncMock(
                side_effect=ModelNotFoundError("Not found")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = registered_tools["get_registered_model"]("nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()


class TestGetModelArtifacts:
    """Test get_model_artifacts tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        return server

    def test_get_artifacts_success(self, mock_server: MagicMock) -> None:
        """Test getting artifacts successfully."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        artifacts = [
            ModelArtifact(
                id="artifact-1",
                name="weights",
                uri="s3://bucket/model/weights.bin",
                model_format_name="pytorch",
            ),
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_model_artifacts = AsyncMock(return_value=artifacts)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = registered_tools["get_model_artifacts"]("version-123")

        assert result["version_id"] == "version-123"
        assert result["count"] == 1
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["uri"] == "s3://bucket/model/weights.bin"
