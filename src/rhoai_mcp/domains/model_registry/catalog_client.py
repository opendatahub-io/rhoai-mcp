"""Red Hat AI Model Catalog REST API client.

This client communicates with the Model Catalog API which uses a different
endpoint structure (/api/model_catalog/v1alpha1) than the standard Kubeflow
Model Registry (/api/model_registry/v1alpha3).
"""

from __future__ import annotations

import logging
import ssl
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx

from rhoai_mcp.domains.model_registry.auth import build_auth_headers
from rhoai_mcp.domains.model_registry.catalog_models import (
    CatalogModel,
    CatalogModelArtifact,
    CatalogSource,
)
from rhoai_mcp.domains.model_registry.client import _format_connection_error
from rhoai_mcp.domains.model_registry.errors import (
    ModelNotFoundError,
    ModelRegistryConnectionError,
    ModelRegistryError,
)

if TYPE_CHECKING:
    from rhoai_mcp.config import RHOAIConfig
    from rhoai_mcp.domains.model_registry.discovery import DiscoveredModelRegistry

logger = logging.getLogger(__name__)


class ModelCatalogClient:
    """Client for Red Hat AI Model Catalog REST API.

    This client communicates with the Model Catalog service using HTTP REST
    calls. The Model Catalog provides a curated set of validated AI models
    from Red Hat.

    Usage:
        async with ModelCatalogClient(config, discovery_result) as client:
            models = await client.list_models()
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
                    "TLS verification disabled for Model Catalog. "
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

    async def probe_availability(self) -> bool:
        """Test if the Model Catalog API is available.

        Returns:
            True if the API responds successfully, False otherwise.
        """
        client = await self._get_client()
        try:
            response = await client.get(
                "/api/model_catalog/v1alpha1/models",
                params={"pageSize": 1},
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Model Catalog probe failed: {e}")
            return False

    async def list_models(
        self,
        page_size: int = 50,
        source_label: str | None = None,
    ) -> list[CatalogModel]:
        """List models in the catalog.

        Args:
            page_size: Maximum number of models to return.
            source_label: Optional source label filter (e.g., 'Red Hat AI validated').

        Returns:
            List of catalog models.

        Raises:
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        client = await self._get_client()
        params: dict[str, str | int] = {"pageSize": page_size}
        if source_label:
            params["sourceLabel"] = source_label

        try:
            response = await client.get(
                "/api/model_catalog/v1alpha1/models",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data.get("models", data.get("items", [])):
                models.append(self._parse_catalog_model(item))
            return models

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Catalog at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to list catalog models: {e}") from e

    async def get_sources(self) -> list[CatalogSource]:
        """List available sources in the catalog.

        Sources are categories or providers of models, such as
        'Red Hat AI validated' or 'Community'.

        Returns:
            List of catalog sources.

        Raises:
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        client = await self._get_client()

        try:
            response = await client.get("/api/model_catalog/v1alpha1/sources")
            response.raise_for_status()
            data = response.json()

            sources = []
            for item in data.get("sources", data.get("items", [])):
                sources.append(self._parse_catalog_source(item))
            return sources

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Catalog at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to list catalog sources: {e}") from e

    async def get_model_artifacts(
        self,
        source: str,
        model_name: str,
    ) -> list[CatalogModelArtifact]:
        """Get artifacts for a specific model.

        Args:
            source: Source label (e.g., 'rhoai' or the source name).
            model_name: Name of the model.

        Returns:
            List of model artifacts with storage URIs.

        Raises:
            ModelNotFoundError: If the model doesn't exist.
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        client = await self._get_client()
        # URL-encode the source and model name
        encoded_source = quote(source, safe="")
        encoded_name = quote(model_name, safe="")

        try:
            response = await client.get(
                f"/api/model_catalog/v1alpha1/sources/{encoded_source}/models/{encoded_name}/artifacts"
            )
            if response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {source}/{model_name}")
            response.raise_for_status()
            data = response.json()

            artifacts = []
            for item in data.get("artifacts", data.get("items", [])):
                artifacts.append(self._parse_catalog_artifact(item))
            return artifacts

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(
                _format_connection_error(self._get_base_url(), e)
            ) from e
        except httpx.TimeoutException as e:
            raise ModelRegistryConnectionError(
                f"Timeout connecting to Model Catalog at {self._get_base_url()}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ModelRegistryError(f"Failed to get model artifacts: {e}") from e

    def _parse_catalog_model(self, data: dict[str, Any]) -> CatalogModel:
        """Parse a catalog model from API response."""
        # Handle artifacts if present
        artifacts = []
        for artifact_data in data.get("artifacts", []):
            artifacts.append(self._parse_catalog_artifact(artifact_data))

        # Extract size from customProperties if not at top level
        size = data.get("size")
        if not size:
            custom_props = data.get("customProperties", {})
            size_prop = custom_props.get("size", {})
            if isinstance(size_prop, dict):
                size = size_prop.get("string_value")

        return CatalogModel(
            name=data.get("name", ""),
            description=data.get("description"),
            provider=data.get("provider"),
            source_id=data.get("source_id", ""),
            source_label=data.get("sourceLabel", data.get("source_label", "")),
            task_type=data.get("taskType", data.get("task_type")),
            tags=data.get("tags", []),
            size=size,
            license=data.get("license"),
            artifacts=artifacts,
            long_description=data.get("longDescription", data.get("long_description")),
            readme=data.get("readme"),
        )

    def _parse_catalog_source(self, data: dict[str, Any]) -> CatalogSource:
        """Parse a catalog source from API response."""
        # The API returns 'id' as the source identifier for API calls
        # and 'name' as the display name, with optional 'labels' array
        labels = data.get("labels", [])
        label = labels[0] if labels else data.get("label", data.get("name", ""))
        return CatalogSource(
            id=data.get("id", ""),
            name=data.get("name", ""),
            label=label,
            model_count=data.get("modelCount", data.get("model_count", 0)),
            description=data.get("description"),
        )

    def _parse_catalog_artifact(self, data: dict[str, Any]) -> CatalogModelArtifact:
        """Parse a catalog artifact from API response."""
        return CatalogModelArtifact(
            uri=data.get("uri", ""),
            format=data.get("format"),
            size=data.get("size"),
            quantization=data.get("quantization"),
        )

    async def __aenter__(self) -> ModelCatalogClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
