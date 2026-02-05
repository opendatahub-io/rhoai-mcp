"""Tests for ModelCatalogClient."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rhoai_mcp.config import ModelRegistryAuthMode, RHOAIConfig
from rhoai_mcp.domains.model_registry.catalog_client import ModelCatalogClient
from rhoai_mcp.domains.model_registry.discovery import DiscoveredModelRegistry
from rhoai_mcp.domains.model_registry.errors import (
    ModelNotFoundError,
    ModelRegistryConnectionError,
    ModelRegistryError,
)


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock config."""
    config = MagicMock(spec=RHOAIConfig)
    config.model_registry_url = "http://model-catalog.test:8080"
    config.model_registry_timeout = 30
    config.model_registry_auth_mode = ModelRegistryAuthMode.NONE
    config.model_registry_skip_tls_verify = False
    config.model_registry_token = None
    return config


@pytest.fixture
def mock_discovery() -> DiscoveredModelRegistry:
    """Create a mock discovery result."""
    return DiscoveredModelRegistry(
        url="https://model-catalog.example.com",
        namespace="rhoai-model-registries",
        service_name="model-catalog",
        port=443,
        source="crd_route",
        requires_auth=True,
        is_external=True,
        api_type="model_catalog",
    )


@pytest.fixture
def sample_catalog_model() -> dict[str, Any]:
    """Sample catalog model API response."""
    return {
        "name": "granite-3b-code-instruct",
        "description": "IBM Granite 3B Code Instruct Model",
        "provider": "IBM",
        "sourceLabel": "Red Hat AI validated",
        "taskType": "text-generation",
        "tags": ["code", "instruct", "3b"],
        "size": "3B parameters",
        "license": "Apache 2.0",
        "artifacts": [
            {
                "uri": "oci://registry.example.com/models/granite-3b:v1",
                "format": "safetensors",
                "size": "6.5 GB",
            }
        ],
    }


@pytest.fixture
def sample_catalog_source() -> dict[str, Any]:
    """Sample catalog source API response."""
    return {
        "name": "rhoai",
        "label": "Red Hat AI validated",
        "modelCount": 12,
        "description": "Models validated by Red Hat AI team",
    }


class TestModelCatalogClient:
    """Test ModelCatalogClient operations."""

    @pytest.fixture
    def client(
        self, mock_config: MagicMock, mock_discovery: DiscoveredModelRegistry
    ) -> ModelCatalogClient:
        """Create a client with mocked config and discovery."""
        return ModelCatalogClient(mock_config, mock_discovery)

    @pytest.mark.asyncio
    async def test_list_models_empty(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test listing models when none exist."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            models = await client.list_models()

        assert models == []
        mock_http.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_models_with_results(
        self,
        client: ModelCatalogClient,
        sample_catalog_model: dict[str, Any],
    ) -> None:
        """Test listing models returns parsed models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [sample_catalog_model]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            models = await client.list_models()

        assert len(models) == 1
        assert models[0].name == "granite-3b-code-instruct"
        assert models[0].provider == "IBM"
        assert models[0].source_label == "Red Hat AI validated"
        assert models[0].task_type == "text-generation"
        assert len(models[0].artifacts) == 1

    @pytest.mark.asyncio
    async def test_list_models_with_source_filter(
        self,
        client: ModelCatalogClient,
        sample_catalog_model: dict[str, Any],
    ) -> None:
        """Test listing models with source label filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [sample_catalog_model]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            models = await client.list_models(source_label="Red Hat AI validated")

        mock_http.get.assert_called_once()
        call_args = mock_http.get.call_args
        assert call_args[1]["params"]["sourceLabel"] == "Red Hat AI validated"

    @pytest.mark.asyncio
    async def test_list_models_with_items_key(
        self,
        client: ModelCatalogClient,
        sample_catalog_model: dict[str, Any],
    ) -> None:
        """Test listing models handles 'items' key in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [sample_catalog_model]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            models = await client.list_models()

        assert len(models) == 1

    @pytest.mark.asyncio
    async def test_get_sources(
        self,
        client: ModelCatalogClient,
        sample_catalog_source: dict[str, Any],
    ) -> None:
        """Test getting catalog sources."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sources": [sample_catalog_source]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            sources = await client.get_sources()

        assert len(sources) == 1
        assert sources[0].name == "rhoai"
        assert sources[0].label == "Red Hat AI validated"
        assert sources[0].model_count == 12

    @pytest.mark.asyncio
    async def test_get_model_artifacts(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test getting model artifacts."""
        artifacts_response = [
            {
                "uri": "oci://registry/model:v1",
                "format": "safetensors",
                "size": "7.5 GB",
                "quantization": "fp16",
            }
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"artifacts": artifacts_response}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            artifacts = await client.get_model_artifacts("rhoai", "granite-3b")

        assert len(artifacts) == 1
        assert artifacts[0].uri == "oci://registry/model:v1"
        assert artifacts[0].format == "safetensors"
        assert artifacts[0].quantization == "fp16"

    @pytest.mark.asyncio
    async def test_get_model_artifacts_not_found(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test getting artifacts for a model that doesn't exist."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            with pytest.raises(ModelNotFoundError):
                await client.get_model_artifacts("unknown", "missing-model")

    @pytest.mark.asyncio
    async def test_probe_availability_success(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test probing API availability when successful."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            result = await client.probe_availability()

        assert result is True

    @pytest.mark.asyncio
    async def test_probe_availability_failure(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test probing API availability when it fails."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http

            result = await client.probe_availability()

        assert result is False

    @pytest.mark.asyncio
    async def test_probe_availability_exception(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test probing API availability when exception occurs."""
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=Exception("Network error"))
            mock_get_client.return_value = mock_http

            result = await client.probe_availability()

        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test handling connection errors."""
        import httpx

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get_client.return_value = mock_http

            with pytest.raises(ModelRegistryConnectionError):
                await client.list_models()

    @pytest.mark.asyncio
    async def test_http_error(
        self,
        client: ModelCatalogClient,
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
                await client.list_models()

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        mock_config: MagicMock,
        mock_discovery: DiscoveredModelRegistry,
    ) -> None:
        """Test async context manager."""
        async with ModelCatalogClient(mock_config, mock_discovery) as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_close(
        self,
        client: ModelCatalogClient,
    ) -> None:
        """Test closing the client."""
        mock_http_client = AsyncMock()
        client._http_client = mock_http_client

        await client.close()

        mock_http_client.aclose.assert_called_once()
        assert client._http_client is None


class TestModelCatalogClientAuth:
    """Test ModelCatalogClient authentication functionality."""

    @pytest.fixture
    def mock_config_no_auth(self) -> MagicMock:
        """Create a mock config with no auth."""
        config = MagicMock()
        config.model_registry_url = "http://model-catalog.test:8080"
        config.model_registry_timeout = 30
        config.model_registry_auth_mode = ModelRegistryAuthMode.NONE
        config.model_registry_skip_tls_verify = False
        return config

    @pytest.fixture
    def mock_discovery_no_auth(self) -> DiscoveredModelRegistry:
        """Create a mock discovery result without auth requirement."""
        return DiscoveredModelRegistry(
            url="http://model-catalog.test:8080",
            namespace="test",
            service_name="model-catalog",
            port=8080,
            source="test",
            requires_auth=False,
            api_type="model_catalog",
        )

    @pytest.fixture
    def mock_discovery_with_auth(self) -> DiscoveredModelRegistry:
        """Create a mock discovery result with auth requirement."""
        return DiscoveredModelRegistry(
            url="https://model-catalog.example.com",
            namespace="test",
            service_name="model-catalog",
            port=443,
            source="test_route",
            requires_auth=True,
            api_type="model_catalog",
        )

    def test_no_auth_returns_empty_headers(
        self,
        mock_config_no_auth: MagicMock,
        mock_discovery_no_auth: DiscoveredModelRegistry,
    ) -> None:
        """When auth is not required, no headers are added."""
        client = ModelCatalogClient(mock_config_no_auth, mock_discovery_no_auth)
        headers = client._get_auth_headers()

        assert headers == {}

    def test_discovery_requires_auth(
        self,
        mock_config_no_auth: MagicMock,
        mock_discovery_with_auth: DiscoveredModelRegistry,
    ) -> None:
        """When discovery indicates auth required, auth is attempted."""
        client = ModelCatalogClient(mock_config_no_auth, mock_discovery_with_auth)

        with patch(
            "rhoai_mcp.domains.model_registry.catalog_client._is_running_in_cluster",
            return_value=False,
        ), patch(
            "rhoai_mcp.domains.model_registry.catalog_client._get_oauth_token_from_kubeconfig",
            return_value="sha256~oauth-token",
        ):
            headers = client._get_auth_headers()

        assert headers["Authorization"] == "Bearer sha256~oauth-token"

    def test_get_base_url_from_discovery(
        self,
        mock_config_no_auth: MagicMock,
        mock_discovery_with_auth: DiscoveredModelRegistry,
    ) -> None:
        """Client uses URL from discovery result."""
        client = ModelCatalogClient(mock_config_no_auth, mock_discovery_with_auth)

        assert client._get_base_url() == "https://model-catalog.example.com"

    def test_get_base_url_from_config(
        self,
        mock_config_no_auth: MagicMock,
    ) -> None:
        """Client uses URL from config when no discovery result."""
        client = ModelCatalogClient(mock_config_no_auth, None)

        assert client._get_base_url() == "http://model-catalog.test:8080"
