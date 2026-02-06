"""Tests for ModelRegistryClient."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rhoai_mcp.config import ModelRegistryAuthMode, RHOAIConfig
from rhoai_mcp.domains.model_registry.auth import (
    _get_in_cluster_token,
    _is_running_in_cluster,
)
from rhoai_mcp.domains.model_registry.client import (
    ModelRegistryClient,
    _format_connection_error,
    _is_internal_k8s_url,
)
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


class TestIsRunningInCluster:
    """Test _is_running_in_cluster helper function."""

    def test_in_cluster_token_exists(self) -> None:
        """When service account token exists, we're in-cluster."""
        with patch("pathlib.Path.exists", return_value=True):
            assert _is_running_in_cluster() is True

    def test_outside_cluster_token_missing(self) -> None:
        """When service account token doesn't exist, we're outside cluster."""
        with patch("pathlib.Path.exists", return_value=False):
            assert _is_running_in_cluster() is False


class TestIsInternalK8sUrl:
    """Test _is_internal_k8s_url helper function."""

    def test_internal_url_with_svc(self) -> None:
        """URLs with .svc: are internal."""
        assert _is_internal_k8s_url("https://model-registry.ns.svc:8443") is True
        assert _is_internal_k8s_url("http://service.namespace.svc:8080") is True

    def test_internal_url_with_cluster_local(self) -> None:
        """URLs with .svc.cluster.local are internal."""
        assert _is_internal_k8s_url("https://svc.ns.svc.cluster.local/path") is True

    def test_external_url(self) -> None:
        """External URLs are not internal."""
        assert _is_internal_k8s_url("http://localhost:8080") is False
        assert _is_internal_k8s_url("https://model-registry.example.com:443") is False

    def test_url_without_port(self) -> None:
        """URLs without port don't match the .svc: pattern."""
        assert _is_internal_k8s_url("https://model-registry.ns.svc/path") is False


class TestFormatConnectionError:
    """Test _format_connection_error helper function."""

    def test_dns_error_outside_cluster(self) -> None:
        """DNS error for internal URL outside cluster provides guidance."""
        url = "https://model-catalog.rhoai-model-registries.svc:8443"
        error = Exception("[Errno -2] Name or service not known")

        with patch(
            "rhoai_mcp.domains.model_registry.client._is_running_in_cluster",
            return_value=False,
        ):
            msg = _format_connection_error(url, error)

        assert "outside the cluster" in msg
        assert "oc login" in msg
        assert "Copy login command" in msg
        assert "oc port-forward" in msg
        assert "-n rhoai-model-registries" in msg
        assert "svc/model-catalog" in msg
        assert "RHOAI_MCP_MODEL_REGISTRY_URL" in msg

    def test_dns_error_inside_cluster(self) -> None:
        """DNS error for internal URL inside cluster doesn't add guidance."""
        url = "https://model-catalog.rhoai-model-registries.svc:8443"
        error = Exception("[Errno -2] Name or service not known")

        with patch(
            "rhoai_mcp.domains.model_registry.client._is_running_in_cluster",
            return_value=True,
        ):
            msg = _format_connection_error(url, error)

        assert "outside the cluster" not in msg
        assert "kubectl port-forward" not in msg

    def test_non_dns_error(self) -> None:
        """Non-DNS errors don't trigger guidance."""
        url = "https://model-catalog.rhoai-model-registries.svc:8443"
        error = Exception("Connection refused")

        with patch(
            "rhoai_mcp.domains.model_registry.client._is_running_in_cluster",
            return_value=False,
        ):
            msg = _format_connection_error(url, error)

        assert "outside the cluster" not in msg
        assert "Connection refused" in msg

    def test_external_url_with_dns_error(self) -> None:
        """DNS errors for external URLs don't trigger port-forward guidance."""
        url = "http://localhost:8080"
        error = Exception("Name or service not known")

        with patch(
            "rhoai_mcp.domains.model_registry.client._is_running_in_cluster",
            return_value=False,
        ):
            msg = _format_connection_error(url, error)

        assert "outside the cluster" not in msg
        assert "kubectl port-forward" not in msg


class TestGetInClusterToken:
    """Test _get_in_cluster_token helper function."""

    def test_token_file_exists(self, tmp_path: Any) -> None:
        """When token file exists, return its contents."""
        token_file = tmp_path / "token"
        token_file.write_text("my-service-account-token")

        with patch(
            "rhoai_mcp.domains.model_registry.auth.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.read_text.return_value = "my-service-account-token\n"
            token = _get_in_cluster_token()

        assert token == "my-service-account-token"

    def test_token_file_missing(self) -> None:
        """When token file doesn't exist, return None."""
        with patch(
            "rhoai_mcp.domains.model_registry.auth.Path"
        ) as mock_path:
            mock_path.return_value.exists.return_value = False
            token = _get_in_cluster_token()

        assert token is None


class TestModelRegistryClientAuth:
    """Test ModelRegistryClient authentication functionality."""

    @pytest.fixture
    def mock_config_no_auth(self) -> MagicMock:
        """Create a mock config with no auth."""
        config = MagicMock()
        config.model_registry_url = "http://model-registry.test:8080"
        config.model_registry_timeout = 30
        config.model_registry_auth_mode = ModelRegistryAuthMode.NONE
        config.model_registry_skip_tls_verify = False
        return config

    @pytest.fixture
    def mock_config_oauth(self) -> MagicMock:
        """Create a mock config with OAuth auth."""
        config = MagicMock()
        config.model_registry_url = "https://model-registry.example.com"
        config.model_registry_timeout = 30
        config.model_registry_auth_mode = ModelRegistryAuthMode.OAUTH
        config.model_registry_skip_tls_verify = False
        config.effective_kubeconfig_path.exists.return_value = True
        config.kubeconfig_context = None
        return config

    @pytest.fixture
    def mock_config_token(self) -> MagicMock:
        """Create a mock config with explicit token auth."""
        config = MagicMock()
        config.model_registry_url = "https://model-registry.example.com"
        config.model_registry_timeout = 30
        config.model_registry_auth_mode = ModelRegistryAuthMode.TOKEN
        config.model_registry_token = "my-explicit-token"
        config.model_registry_skip_tls_verify = False
        return config

    def test_no_auth_returns_empty_headers(self, mock_config_no_auth: MagicMock) -> None:
        """When auth mode is NONE, no headers are added."""
        client = ModelRegistryClient(mock_config_no_auth)
        headers = client._get_auth_headers()

        assert headers == {}

    def test_token_auth_adds_bearer_header(self, mock_config_token: MagicMock) -> None:
        """When auth mode is TOKEN, Authorization header is added."""
        client = ModelRegistryClient(mock_config_token)
        headers = client._get_auth_headers()

        assert headers["Authorization"] == "Bearer my-explicit-token"

    def test_oauth_auth_outside_cluster(self, mock_config_oauth: MagicMock) -> None:
        """When auth mode is OAUTH outside cluster, gets token from CLI."""
        client = ModelRegistryClient(mock_config_oauth)

        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=False,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_cli_token",
            return_value="cli-token-123",
        ):
            headers = client._get_auth_headers()

        assert headers["Authorization"] == "Bearer cli-token-123"

    def test_oauth_auth_outside_cluster_no_token(self, mock_config_oauth: MagicMock) -> None:
        """When auth mode is OAUTH outside cluster but no CLI token, no headers."""
        client = ModelRegistryClient(mock_config_oauth)

        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=False,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_cli_token",
            return_value=None,
        ):
            headers = client._get_auth_headers()

        assert headers == {}

    def test_oauth_auth_inside_cluster(self, mock_config_oauth: MagicMock) -> None:
        """When auth mode is OAUTH inside cluster, gets token from SA."""
        client = ModelRegistryClient(mock_config_oauth)

        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=True,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_in_cluster_token",
            return_value="sa-token-123",
        ):
            headers = client._get_auth_headers()

        assert headers["Authorization"] == "Bearer sa-token-123"

    def test_oauth_auth_inside_cluster_no_token(self, mock_config_oauth: MagicMock) -> None:
        """When auth mode is OAUTH inside cluster but no token found, returns empty headers."""
        client = ModelRegistryClient(mock_config_oauth)

        with patch(
            "rhoai_mcp.domains.model_registry.auth._is_running_in_cluster",
            return_value=True,
        ), patch(
            "rhoai_mcp.domains.model_registry.auth._get_in_cluster_token",
            return_value=None,
        ):
            headers = client._get_auth_headers()

        assert headers == {}
