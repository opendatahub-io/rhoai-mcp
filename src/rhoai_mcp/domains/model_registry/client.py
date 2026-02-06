"""Model Registry REST API client."""

from __future__ import annotations

import logging
import ssl
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from rhoai_mcp.domains.model_registry.auth import (
    _is_running_in_cluster,
    build_auth_headers,
)
from rhoai_mcp.domains.model_registry.errors import (
    ModelNotFoundError,
    ModelRegistryConnectionError,
    ModelRegistryError,
)
from rhoai_mcp.domains.model_registry.models import (
    CustomProperties,
    ModelArtifact,
    ModelVersion,
    RegisteredModel,
)

if TYPE_CHECKING:
    from rhoai_mcp.config import RHOAIConfig
    from rhoai_mcp.domains.model_registry.discovery import DiscoveredModelRegistry

logger = logging.getLogger(__name__)


def _is_internal_k8s_url(url: str) -> bool:
    """Check if the URL is a Kubernetes internal service URL.

    Args:
        url: The URL to check.

    Returns:
        True if the URL uses Kubernetes internal DNS (*.svc or *.svc.cluster.local).
    """
    return ".svc:" in url or ".svc.cluster.local" in url


def _format_connection_error(url: str, error: Exception) -> str:
    """Format a connection error with helpful context.

    If running outside the cluster with an internal URL, provides
    guidance on how to set up port-forwarding.

    Args:
        url: The URL that failed to connect.
        error: The original connection error.

    Returns:
        Formatted error message with actionable guidance.
    """
    base_msg = f"Failed to connect to Model Registry at {url}: {error}"

    # Check if this is a DNS resolution failure for an internal URL
    error_str = str(error).lower()
    is_dns_error = "name or service not known" in error_str or "nodename nor servname" in error_str

    if is_dns_error and _is_internal_k8s_url(url) and not _is_running_in_cluster():
        # Extract service name and namespace from URL for port-forward command
        # URL format: https://service-name.namespace.svc:port
        try:
            # Parse: protocol://service.namespace.svc:port
            host_port = url.split("://")[1]
            host = host_port.split(":")[0]
            port = host_port.split(":")[1].split("/")[0]
            parts = host.split(".")
            if len(parts) >= 2:
                service_name = parts[0]
                namespace = parts[1]

                return (
                    f"{base_msg}\n\n"
                    f"The Model Registry service was discovered in the cluster, but "
                    f"you are running outside the cluster where Kubernetes internal DNS "
                    f"is not available.\n\n"
                    f"First, ensure you are logged in to the cluster:\n"
                    f"   oc login <cluster-url>\n"
                    f"   (Get the login command from your OpenShift web console by clicking "
                    f"your username → 'Copy login command')\n\n"
                    f"Then, to connect, either:\n"
                    f"1. Set up port-forwarding:\n"
                    f"   oc port-forward -n {namespace} svc/{service_name} 8080:{port}\n"
                    f"   Then set: RHOAI_MCP_MODEL_REGISTRY_URL=http://localhost:8080\n\n"
                    f"2. Or configure a direct URL if the service is exposed externally:\n"
                    f"   RHOAI_MCP_MODEL_REGISTRY_URL=<external-url>\n"
                    f"   RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE=manual"
                )
        except (IndexError, ValueError):
            pass  # Fall through to default message

    return base_msg


class ModelRegistryClient:
    """Client for OpenShift AI Model Registry REST API.

    This client communicates with the Model Registry service using HTTP REST
    calls. Unlike other domain clients that use Kubernetes CRDs, the Model
    Registry has its own REST API.

    Usage:
        async with ModelRegistryClient(config) as client:
            models = await client.list_registered_models()
    """

    def __init__(
        self,
        config: RHOAIConfig,
        discovery_result: DiscoveredModelRegistry | None = None,
    ) -> None:
        """Initialize client.

        Args:
            config: RHOAI configuration with Model Registry settings.
            discovery_result: Optional discovery result with URL and auth info.
        """
        self._config = config
        self._discovery = discovery_result
        self._http_client: httpx.AsyncClient | None = None

    def _get_base_url(self) -> str:
        """Get the base URL for API calls."""
        if self._discovery:
            return self._discovery.url
        return self._config.model_registry_url

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers based on config and discovery.

        Returns:
            Dict of headers to include in requests.
        """
        requires_auth_override = self._discovery.requires_auth if self._discovery else False
        return build_auth_headers(self._config, requires_auth_override=requires_auth_override)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with appropriate auth and SSL settings."""
        if self._http_client is None:
            # Get auth headers
            headers = self._get_auth_headers()

            # Configure SSL verification
            verify: bool | ssl.SSLContext = True
            if self._config.model_registry_skip_tls_verify:
                verify = False
                logger.warning(
                    "TLS verification disabled for Model Registry. "
                    "This is not recommended for production."
                )

            self._http_client = httpx.AsyncClient(
                base_url=self._get_base_url(),
                timeout=self._config.model_registry_timeout,
                headers=headers,
                verify=verify,
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def list_registered_models(
        self,
        page_size: int = 100,
    ) -> list[RegisteredModel]:
        """List all registered models.

        Args:
            page_size: Maximum number of models per page.

        Returns:
            List of all registered models (iterates through all pages).

        Raises:
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        all_models: list[RegisteredModel] = []
        next_page_token: str | None = None

        while True:
            models, next_page_token = await self._list_registered_models_page(
                page_size=page_size,
                page_token=next_page_token,
            )
            all_models.extend(models)
            if not next_page_token:
                break

        return all_models

    async def _list_registered_models_page(
        self,
        page_size: int = 100,
        page_token: str | None = None,
    ) -> tuple[list[RegisteredModel], str | None]:
        """List a single page of registered models.

        Args:
            page_size: Maximum number of models to return (not used - RHOAI doesn't support pagination).
            page_token: Token for the next page (not used - RHOAI doesn't support pagination).

        Returns:
            Tuple of (models, next_page_token). next_page_token is always None for RHOAI.

        Raises:
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        # Note: OpenShift AI Model Registry doesn't support pageSize/pageToken parameters
        # on the v1alpha3 API. We fetch all models in a single request.
        _ = page_size  # unused - RHOAI returns all models
        _ = page_token  # unused - RHOAI doesn't support pagination
        client = await self._get_client()

        try:
            response = await client.get(
                "/api/model_registry/v1alpha3/registered_models",
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data.get("items", []):
                models.append(self._parse_registered_model(item))

            next_token = data.get("nextPageToken")
            return models, next_token

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Registry at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to list models: {e}") from e

    async def get_registered_model(self, model_id: str) -> RegisteredModel:
        """Get a registered model by ID.

        Args:
            model_id: The model ID.

        Returns:
            The registered model.

        Raises:
            ModelNotFoundError: If the model doesn't exist.
            ModelRegistryError: If the API request fails.
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"/api/model_registry/v1alpha3/registered_models/{model_id}"
            )
            if response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {model_id}")
            response.raise_for_status()
            return self._parse_registered_model(response.json())

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Registry at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get model: {e}") from e

    async def get_registered_model_by_name(self, name: str) -> RegisteredModel | None:
        """Get a registered model by name, searching all pages.

        Args:
            name: The model name.

        Returns:
            The registered model, or None if not found.
        """
        next_page_token: str | None = None

        while True:
            models, next_page_token = await self._list_registered_models_page(
                page_size=100,
                page_token=next_page_token,
            )
            for model in models:
                if model.name == name:
                    return model
            if not next_page_token:
                break

        return None

    async def get_model_versions(
        self,
        model_id: str,
        page_size: int = 100,
    ) -> list[ModelVersion]:
        """Get all versions of a registered model.

        Args:
            model_id: The parent model ID.
            page_size: Maximum number of versions per page.

        Returns:
            List of all model versions (iterates through all pages).

        Raises:
            ModelRegistryError: If the API request fails.
        """
        all_versions: list[ModelVersion] = []
        next_page_token: str | None = None

        while True:
            versions, next_page_token = await self._get_model_versions_page(
                model_id=model_id,
                page_size=page_size,
                page_token=next_page_token,
            )
            all_versions.extend(versions)
            if not next_page_token:
                break

        return all_versions

    async def _get_model_versions_page(
        self,
        model_id: str,
        page_size: int = 100,
        page_token: str | None = None,
    ) -> tuple[list[ModelVersion], str | None]:
        """Get a single page of model versions.

        Args:
            model_id: The parent model ID.
            page_size: Maximum number of versions to return (not used - RHOAI doesn't support pagination).
            page_token: Token for the next page (not used - RHOAI doesn't support pagination).

        Returns:
            Tuple of (versions, next_page_token). next_page_token is always None for RHOAI.

        Raises:
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        # Note: OpenShift AI Model Registry doesn't support pageSize/pageToken parameters
        # on the v1alpha3 API. We fetch all versions in a single request.
        _ = page_size  # unused - RHOAI returns all versions
        _ = page_token  # unused - RHOAI doesn't support pagination
        client = await self._get_client()

        try:
            response = await client.get(
                f"/api/model_registry/v1alpha3/registered_models/{model_id}/versions",
            )
            response.raise_for_status()
            data = response.json()

            versions = []
            for item in data.get("items", []):
                versions.append(self._parse_model_version(item))

            next_token = data.get("nextPageToken")
            return versions, next_token

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Registry at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get model versions: {e}") from e

    async def get_model_version(self, version_id: str) -> ModelVersion:
        """Get a specific model version.

        Args:
            version_id: The version ID.

        Returns:
            The model version.

        Raises:
            ModelNotFoundError: If the version doesn't exist.
            ModelRegistryError: If the API request fails.
        """
        client = await self._get_client()

        try:
            response = await client.get(f"/api/model_registry/v1alpha3/model_versions/{version_id}")
            if response.status_code == 404:
                raise ModelNotFoundError(f"Version not found: {version_id}")
            response.raise_for_status()
            return self._parse_model_version(response.json())

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Registry at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get model version: {e}") from e

    async def get_model_artifacts(self, version_id: str) -> list[ModelArtifact]:
        """Get artifacts for a model version.

        Args:
            version_id: The model version ID.

        Returns:
            List of model artifacts.

        Raises:
            ModelRegistryError: If the API request fails.
        """
        client = await self._get_client()

        try:
            response = await client.get(
                f"/api/model_registry/v1alpha3/model_versions/{version_id}/artifacts"
            )
            response.raise_for_status()
            data = response.json()

            artifacts = []
            for item in data.get("items", []):
                artifacts.append(self._parse_model_artifact(item))
            return artifacts

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Registry at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get artifacts: {e}") from e

    def _parse_timestamp(self, timestamp_str: str | None) -> datetime | None:
        """Parse ISO 8601 timestamp from API response.

        Args:
            timestamp_str: ISO 8601 timestamp string (e.g., "2024-01-15T10:30:00Z").

        Returns:
            Parsed datetime object, or None if parsing fails.
        """
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    def _parse_registered_model(self, data: dict[str, Any]) -> RegisteredModel:
        """Parse registered model from API response."""
        return RegisteredModel(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            owner=data.get("owner"),
            state=data.get("state", "LIVE"),
            custom_properties=CustomProperties(properties=data.get("customProperties", {})),
            create_time=self._parse_timestamp(data.get("createTime")),
            update_time=self._parse_timestamp(data.get("updateTime")),
        )

    def _parse_model_version(self, data: dict[str, Any]) -> ModelVersion:
        """Parse model version from API response."""
        return ModelVersion(
            id=data.get("id", ""),
            name=data.get("name", ""),
            registered_model_id=data.get("registeredModelId", ""),
            state=data.get("state", "LIVE"),
            description=data.get("description"),
            author=data.get("author"),
            custom_properties=CustomProperties(properties=data.get("customProperties", {})),
            create_time=self._parse_timestamp(data.get("createTime")),
            update_time=self._parse_timestamp(data.get("updateTime")),
        )

    def _parse_model_artifact(self, data: dict[str, Any]) -> ModelArtifact:
        """Parse model artifact from API response."""
        return ModelArtifact(
            id=data.get("id", ""),
            name=data.get("name", ""),
            uri=data.get("uri", ""),
            description=data.get("description"),
            model_format_name=data.get("modelFormatName"),
            model_format_version=data.get("modelFormatVersion"),
            storage_key=data.get("storageKey"),
            storage_path=data.get("storagePath"),
            service_account_name=data.get("serviceAccountName"),
            custom_properties=CustomProperties(properties=data.get("customProperties", {})),
            create_time=self._parse_timestamp(data.get("createTime")),
            update_time=self._parse_timestamp(data.get("updateTime")),
        )

    async def __aenter__(self) -> ModelRegistryClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
