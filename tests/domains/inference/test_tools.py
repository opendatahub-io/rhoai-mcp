"""Tests for inference MCP tools."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rhoai_mcp.domains.model_registry.catalog_models import (
    CatalogModel,
    CatalogModelArtifact,
)


def _register_tools(mock_server: MagicMock) -> dict[str, Any]:
    """Register inference tools and return captured tool functions."""
    from rhoai_mcp.domains.inference.tools import register_tools

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


class TestPrepareModelDeploymentCatalogLookup:
    """Test that prepare_model_deployment resolves storage_uri from catalog."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server with k8s client."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        # Mock k8s client for list_serving_runtimes
        mock_runtime = MagicMock()
        mock_runtime.metadata.name = "vllm-runtime"
        mock_runtime.metadata.annotations = {"openshift.io/display-name": "vLLM"}
        mock_runtime.spec = {
            "supportedModelFormats": [{"name": "pytorch"}],
            "multiModel": False,
        }
        server.k8s.list_resources.return_value = [mock_runtime]
        return server

    async def test_resolves_storage_uri_from_catalog(self, mock_server: MagicMock) -> None:
        """When no storage_uri provided, should look it up from model catalog."""
        registered_tools = _register_tools(mock_server)

        with patch(
            "rhoai_mcp.domains.inference.tools._resolve_catalog_storage_uri",
            new_callable=AsyncMock,
            return_value="oci://registry.redhat.io/rhoai/granite-8b:latest",
        ):
            result = await registered_tools["prepare_model_deployment"](
                namespace="medical-llm",
                model_id="ibm-granite/granite-3.1-8b-instruct",
            )

        # Should have resolved the storage_uri, not warn about it missing
        warnings = result.get("warnings") or []
        assert not any("No storage_uri provided" in w for w in warnings)
        assert (
            result["suggested_deploy_params"]["storage_uri"]
            == "oci://registry.redhat.io/rhoai/granite-8b:latest"
        )

    async def test_falls_back_when_catalog_unavailable(self, mock_server: MagicMock) -> None:
        """When catalog lookup fails, should fall back to warning."""
        mock_server.config.model_registry_enabled = False
        registered_tools = _register_tools(mock_server)

        with patch(
            "rhoai_mcp.domains.inference.tools._resolve_catalog_storage_uri",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await registered_tools["prepare_model_deployment"](
                namespace="medical-llm",
                model_id="ibm-granite/granite-3.1-8b-instruct",
            )

        warnings = result.get("warnings") or []
        assert any("No storage_uri provided" in w for w in warnings)
        assert result["suggested_deploy_params"]["storage_uri"] is None


class TestPrepareModelDeploymentOciValidation:
    """Test that prepare_model_deployment validates pull secrets for OCI URIs."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server with k8s client."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        mock_runtime = MagicMock()
        mock_runtime.metadata.name = "vllm-runtime"
        mock_runtime.metadata.annotations = {"openshift.io/display-name": "vLLM"}
        mock_runtime.spec = {
            "supportedModelFormats": [{"name": "pytorch"}],
            "multiModel": False,
        }
        server.k8s.list_resources.return_value = [mock_runtime]
        return server

    async def test_oci_uri_warns_when_no_pull_secrets(self, mock_server: MagicMock) -> None:
        """OCI URI should warn when default SA has no imagePullSecrets."""
        registered_tools = _register_tools(mock_server)

        # Default service account with no imagePullSecrets
        mock_sa = MagicMock()
        mock_sa.image_pull_secrets = None
        mock_server.k8s.core_v1.read_namespaced_service_account.return_value = mock_sa

        with patch(
            "rhoai_mcp.domains.inference.tools._resolve_catalog_storage_uri",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await registered_tools["prepare_model_deployment"](
                namespace="medical-llm",
                model_id="ibm-granite/granite-3.1-8b-instruct",
                storage_uri="oci://quay.io/modh/granite-3.1-8b-instruct:latest",
            )

        warnings = result.get("warnings") or []
        assert any("pull secret" in w.lower() for w in warnings)
        assert any("quay.io" in w for w in warnings)

    async def test_oci_uri_no_warning_when_pull_secrets_exist(self, mock_server: MagicMock) -> None:
        """OCI URI should not warn when SA has imagePullSecrets."""
        registered_tools = _register_tools(mock_server)

        mock_secret_ref = MagicMock()
        mock_secret_ref.name = "quay-pull-secret"
        mock_sa = MagicMock()
        mock_sa.image_pull_secrets = [mock_secret_ref]
        mock_server.k8s.core_v1.read_namespaced_service_account.return_value = mock_sa

        with patch(
            "rhoai_mcp.domains.inference.tools._resolve_catalog_storage_uri",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await registered_tools["prepare_model_deployment"](
                namespace="medical-llm",
                model_id="ibm-granite/granite-3.1-8b-instruct",
                storage_uri="oci://quay.io/modh/granite-3.1-8b-instruct:latest",
            )

        warnings = result.get("warnings") or []
        assert not any("pull secret" in w.lower() for w in warnings)


class TestExtractOciRegistry:
    """Test _extract_oci_registry helper."""

    def test_oci_scheme(self) -> None:
        from rhoai_mcp.domains.inference.tools import _extract_oci_registry

        assert _extract_oci_registry("oci://quay.io/modh/model:v1") == "quay.io"

    def test_docker_scheme(self) -> None:
        from rhoai_mcp.domains.inference.tools import _extract_oci_registry

        assert (
            _extract_oci_registry("docker://registry.redhat.io/rhoai/model:latest")
            == "registry.redhat.io"
        )

    def test_localhost_returns_host(self) -> None:
        from rhoai_mcp.domains.inference.tools import _extract_oci_registry

        assert _extract_oci_registry("oci://localhost/model:v1") == "localhost"

    def test_localhost_with_port_returns_host(self) -> None:
        from rhoai_mcp.domains.inference.tools import _extract_oci_registry

        assert _extract_oci_registry("oci://localhost:5000/model:v1") == "localhost:5000"

    def test_bare_name_returns_none(self) -> None:
        from rhoai_mcp.domains.inference.tools import _extract_oci_registry

        assert _extract_oci_registry("oci://myimage:latest") is None


class TestResolveCatalogStorageUri:
    """Test _resolve_catalog_storage_uri performs discovery when cache is empty."""

    async def test_discovers_and_resolves_when_cache_empty(self) -> None:
        """Should run discovery when cache is not populated."""
        from rhoai_mcp.domains.inference.tools import _resolve_catalog_storage_uri

        config = MagicMock()
        config.model_registry_enabled = True
        config.model_registry_url = "http://registry:8080"
        config.model_registry_discovery_mode = "auto"
        config.model_registry_skip_tls_verify = False
        config.model_registry_token = None
        config.model_registry_auth_mode = "none"

        k8s = MagicMock()

        artifact = CatalogModelArtifact(
            uri="oci://registry.redhat.io/rhoai/granite-8b:latest",
            format="safetensors",
        )
        catalog_model = CatalogModel(
            name="ibm-granite/granite-3.1-8b-instruct",
            source_id="rhoai",
            source_label="Red Hat AI validated",
            artifacts=[artifact],
        )

        # Clear the cache to simulate first call
        import rhoai_mcp.domains.model_registry.tools as mr_tools

        original_api_type = mr_tools._cached_api_type
        original_url = mr_tools._cached_discovery_url
        original_auth = mr_tools._cached_requires_auth
        mr_tools._cached_api_type = None
        mr_tools._cached_discovery_url = None
        mr_tools._cached_requires_auth = False

        try:
            mock_discovery_result = MagicMock()
            mock_discovery_result.url = "https://catalog.example.com"
            mock_discovery_result.requires_auth = False
            mock_discovery_result.api_type = "model_catalog"

            with (
                patch(
                    "rhoai_mcp.domains.model_registry.discovery.ModelRegistryDiscovery"
                ) as mock_discovery_class,
                patch(
                    "rhoai_mcp.domains.model_registry.catalog_client.ModelCatalogClient"
                ) as mock_client_class,
            ):
                mock_discovery = MagicMock()
                mock_discovery.discover_with_port_forward = AsyncMock(
                    return_value=mock_discovery_result
                )
                mock_discovery_class.return_value = mock_discovery

                mock_client = AsyncMock()
                mock_client.list_models = AsyncMock(return_value=[catalog_model])
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _resolve_catalog_storage_uri(
                    config, k8s, "ibm-granite/granite-3.1-8b-instruct"
                )

            assert result == "oci://registry.redhat.io/rhoai/granite-8b:latest"
            # Cache should now be populated
            assert mr_tools._cached_api_type == "model_catalog"
            assert mr_tools._cached_discovery_url == "https://catalog.example.com"
        finally:
            # Restore cache
            mr_tools._cached_api_type = original_api_type
            mr_tools._cached_discovery_url = original_url
            mr_tools._cached_requires_auth = original_auth

    async def test_returns_none_when_registry_disabled(self) -> None:
        """Should return None immediately when model registry is disabled."""
        from rhoai_mcp.domains.inference.tools import _resolve_catalog_storage_uri

        config = MagicMock()
        config.model_registry_enabled = False

        result = await _resolve_catalog_storage_uri(config, MagicMock(), "some-model")
        assert result is None

    async def test_returns_none_when_not_catalog(self) -> None:
        """Should return None when discovered API is not model_catalog."""
        from rhoai_mcp.domains.inference.tools import _resolve_catalog_storage_uri

        config = MagicMock()
        config.model_registry_enabled = True
        config.model_registry_url = "http://registry:8080"
        config.model_registry_discovery_mode = "auto"

        import rhoai_mcp.domains.model_registry.tools as mr_tools

        original_api_type = mr_tools._cached_api_type
        original_url = mr_tools._cached_discovery_url
        original_auth = mr_tools._cached_requires_auth
        mr_tools._cached_api_type = None
        mr_tools._cached_discovery_url = None

        try:
            mock_discovery_result = MagicMock()
            mock_discovery_result.url = "http://registry:8080"
            mock_discovery_result.requires_auth = False
            mock_discovery_result.api_type = "model_registry"

            with patch(
                "rhoai_mcp.domains.model_registry.discovery.ModelRegistryDiscovery"
            ) as mock_discovery_class:
                mock_discovery = MagicMock()
                mock_discovery.discover_with_port_forward = AsyncMock(
                    return_value=mock_discovery_result
                )
                mock_discovery_class.return_value = mock_discovery

                result = await _resolve_catalog_storage_uri(config, MagicMock(), "some-model")

            assert result is None
        finally:
            mr_tools._cached_api_type = original_api_type
            mr_tools._cached_discovery_url = original_url
            mr_tools._cached_requires_auth = original_auth
