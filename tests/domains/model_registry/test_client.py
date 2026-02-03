"""Tests for ModelRegistryClient."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rhoai_mcp.config import RHOAIConfig
from rhoai_mcp.domains.model_registry.client import ModelRegistryClient
from rhoai_mcp.domains.model_registry.errors import (
    ModelNotFoundError,
    ModelRegistryConnectionError,
    ModelRegistryError,
)


class TestModelRegistryClient:
    """Test ModelRegistryClient operations."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config."""
        config = MagicMock(spec=RHOAIConfig)
        config.model_registry_url = "http://model-registry.test:8080"
        config.model_registry_timeout = 30
        return config

    @pytest.fixture
    def client(self, mock_config: MagicMock) -> ModelRegistryClient:
        """Create a client with mocked config."""
        return ModelRegistryClient(mock_config)

    @pytest.mark.asyncio
    async def test_list_registered_models_empty(
        self,
        client: ModelRegistryClient,
    ) -> None:
        """Test listing models when none exist."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            models = await client.list_registered_models()

        assert models == []
        mock_http.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_registered_models_with_results(
        self,
        client: ModelRegistryClient,
        sample_registered_model: dict[str, Any],
    ) -> None:
        """Test listing models returns parsed models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [sample_registered_model]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            models = await client.list_registered_models()

        assert len(models) == 1
        assert models[0].id == "model-123"
        assert models[0].name == "llama-2-7b"
        assert models[0].owner == "data-science-team"

    @pytest.mark.asyncio
    async def test_get_registered_model(
        self,
        client: ModelRegistryClient,
        sample_registered_model: dict[str, Any],
    ) -> None:
        """Test getting a specific model."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_registered_model
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            model = await client.get_registered_model("model-123")

        assert model.id == "model-123"
        assert model.name == "llama-2-7b"
        mock_http.get.assert_called_once_with(
            "/api/model_registry/v1alpha3/registered_models/model-123"
        )

    @pytest.mark.asyncio
    async def test_get_registered_model_not_found(
        self,
        client: ModelRegistryClient,
    ) -> None:
        """Test getting a model that doesn't exist."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            with pytest.raises(ModelNotFoundError):
                await client.get_registered_model("nonexistent")

    @pytest.mark.asyncio
    async def test_get_model_versions(
        self,
        client: ModelRegistryClient,
        sample_model_version: dict[str, Any],
    ) -> None:
        """Test getting model versions."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [sample_model_version]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            versions = await client.get_model_versions("model-123")

        assert len(versions) == 1
        assert versions[0].id == "version-456"
        assert versions[0].name == "v1.0.0"

    @pytest.mark.asyncio
    async def test_get_model_version(
        self,
        client: ModelRegistryClient,
        sample_model_version: dict[str, Any],
    ) -> None:
        """Test getting a specific version."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_model_version
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            version = await client.get_model_version("version-456")

        assert version.id == "version-456"
        assert version.author == "ml-engineer"

    @pytest.mark.asyncio
    async def test_get_model_version_not_found(
        self,
        client: ModelRegistryClient,
    ) -> None:
        """Test getting a version that doesn't exist."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            with pytest.raises(ModelNotFoundError):
                await client.get_model_version("nonexistent")

    @pytest.mark.asyncio
    async def test_get_model_artifacts(
        self,
        client: ModelRegistryClient,
        sample_model_artifact: dict[str, Any],
    ) -> None:
        """Test getting model artifacts."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [sample_model_artifact]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            artifacts = await client.get_model_artifacts("version-456")

        assert len(artifacts) == 1
        assert artifacts[0].id == "artifact-789"
        assert artifacts[0].uri == "s3://models/llama-2-7b/v1.0.0/weights.safetensors"
        assert artifacts[0].model_format_name == "safetensors"

    @pytest.mark.asyncio
    async def test_get_registered_model_by_name(
        self,
        client: ModelRegistryClient,
        sample_registered_model: dict[str, Any],
    ) -> None:
        """Test finding a model by name."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [sample_registered_model]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            model = await client.get_registered_model_by_name("llama-2-7b")

        assert model is not None
        assert model.name == "llama-2-7b"

    @pytest.mark.asyncio
    async def test_get_registered_model_by_name_not_found(
        self,
        client: ModelRegistryClient,
    ) -> None:
        """Test finding a model by name when it doesn't exist."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            model = await client.get_registered_model_by_name("nonexistent")

        assert model is None

    @pytest.mark.asyncio
    async def test_connection_error(
        self,
        client: ModelRegistryClient,
    ) -> None:
        """Test handling connection errors."""
        import httpx

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get_client.return_value = mock_http

            with pytest.raises(ModelRegistryConnectionError):
                await client.list_registered_models()

    @pytest.mark.asyncio
    async def test_http_error(
        self,
        client: ModelRegistryClient,
    ) -> None:
        """Test handling HTTP errors."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(spec=httpx.Request),
            response=MagicMock(spec=httpx.Response),
        )

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            with pytest.raises(ModelRegistryError):
                await client.list_registered_models()

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Test async context manager."""
        async with ModelRegistryClient(mock_config) as client:
            assert client is not None
            # Client should be usable within context

    @pytest.mark.asyncio
    async def test_close(
        self,
        client: ModelRegistryClient,
    ) -> None:
        """Test closing the client."""
        # Create an internal client first
        mock_http_client = AsyncMock()
        client._http_client = mock_http_client

        await client.close()

        mock_http_client.aclose.assert_called_once()
        assert client._http_client is None
