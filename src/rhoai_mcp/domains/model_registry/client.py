"""Model Registry REST API client."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

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


class ModelRegistryClient:
    """Client for OpenShift AI Model Registry REST API.

    This client communicates with the Model Registry service using HTTP REST
    calls. Unlike other domain clients that use Kubernetes CRDs, the Model
    Registry has its own REST API.

    Usage:
        async with ModelRegistryClient(config) as client:
            models = await client.list_registered_models()
    """

    def __init__(self, config: "RHOAIConfig") -> None:
        """Initialize client.

        Args:
            config: RHOAI configuration with Model Registry settings.
        """
        self._config = config
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self._config.model_registry_url,
                timeout=self._config.model_registry_timeout,
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
        order_by: str = "UPDATE_TIME",
    ) -> list[RegisteredModel]:
        """List all registered models.

        Args:
            page_size: Maximum number of models per page.
            order_by: Field to order by (UPDATE_TIME, CREATE_TIME, NAME).

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
                order_by=order_by,
                page_token=next_page_token,
            )
            all_models.extend(models)
            if not next_page_token:
                break

        return all_models

    async def _list_registered_models_page(
        self,
        page_size: int = 100,
        order_by: str = "UPDATE_TIME",
        page_token: str | None = None,
    ) -> tuple[list[RegisteredModel], str | None]:
        """List a single page of registered models.

        Args:
            page_size: Maximum number of models to return.
            order_by: Field to order by (UPDATE_TIME, CREATE_TIME, NAME).
            page_token: Token for the next page (None for first page).

        Returns:
            Tuple of (models, next_page_token). next_page_token is None if no more pages.

        Raises:
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        client = await self._get_client()
        params: dict[str, str | int] = {
            "pageSize": page_size,
            "orderBy": order_by,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            response = await client.get(
                "/api/model_registry/v1alpha3/registered_models",
                params=params,
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
                f"Failed to connect to Model Registry at {self._config.model_registry_url}: {e}"
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
            raise ModelRegistryConnectionError(f"Failed to connect to Model Registry: {e}") from e
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
            page_size: Maximum number of versions to return.
            page_token: Token for the next page (None for first page).

        Returns:
            Tuple of (versions, next_page_token). next_page_token is None if no more pages.

        Raises:
            ModelRegistryError: If the API request fails.
            ModelRegistryConnectionError: If connection fails.
        """
        client = await self._get_client()
        params: dict[str, str | int] = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token

        try:
            response = await client.get(
                f"/api/model_registry/v1alpha3/registered_models/{model_id}/versions",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            versions = []
            for item in data.get("items", []):
                versions.append(self._parse_model_version(item))

            next_token = data.get("nextPageToken")
            return versions, next_token

        except httpx.ConnectError as e:
            raise ModelRegistryConnectionError(f"Failed to connect to Model Registry: {e}") from e
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
            raise ModelRegistryConnectionError(f"Failed to connect to Model Registry: {e}") from e
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
            raise ModelRegistryConnectionError(f"Failed to connect to Model Registry: {e}") from e
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

    async def __aenter__(self) -> "ModelRegistryClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
