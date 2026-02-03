"""MCP Tools for Model Registry operations."""

import asyncio
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.model_registry.client import ModelRegistryClient
from rhoai_mcp.domains.model_registry.errors import (
    ModelNotFoundError,
    ModelRegistryConnectionError,
    ModelRegistryError,
)
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    Verbosity,
    paginate,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register Model Registry tools with the MCP server."""

    @mcp.tool()
    def list_registered_models(
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List registered models in the Model Registry with pagination.

        The Model Registry stores metadata about ML models, including versions,
        artifacts, and custom properties. Use this to discover available models
        before deployment.

        Args:
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Paginated list of registered models with metadata.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _list() -> list[dict[str, Any]]:
            async with ModelRegistryClient(server.config) as client:
                models = await client.list_registered_models()
                return [_format_model(m, Verbosity.from_str(verbosity)) for m in models]

        try:
            all_items = asyncio.run(_list())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(all_items, offset, effective_limit)

        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_registered_model(
        model_id: str,
        include_versions: bool = False,
    ) -> dict[str, Any]:
        """Get detailed information about a registered model.

        Args:
            model_id: The model ID in the registry.
            include_versions: If True, also fetch all versions for this model.

        Returns:
            Model details including name, description, owner, and optionally versions.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> dict[str, Any]:
            async with ModelRegistryClient(server.config) as client:
                model = await client.get_registered_model(model_id)
                result = _format_model(model, Verbosity.FULL)

                if include_versions:
                    versions = await client.get_model_versions(model_id)
                    result["versions"] = [_format_version(v, Verbosity.STANDARD) for v in versions]

                return result

        try:
            return asyncio.run(_get())
        except ModelNotFoundError:
            return {"error": f"Model not found: {model_id}"}
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    def list_model_versions(
        model_id: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List all versions of a registered model with pagination.

        Each version represents a specific iteration of the model with its own
        artifacts and metadata.

        Args:
            model_id: The parent model ID.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            Paginated list of model versions.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _list() -> list[dict[str, Any]]:
            async with ModelRegistryClient(server.config) as client:
                versions = await client.get_model_versions(model_id)
                return [_format_version(v, Verbosity.from_str(verbosity)) for v in versions]

        try:
            all_items = asyncio.run(_list())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(all_items, offset, effective_limit)

        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_model_version(version_id: str) -> dict[str, Any]:
        """Get detailed information about a specific model version.

        Args:
            version_id: The version ID.

        Returns:
            Version details including state, author, and custom properties.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> dict[str, Any]:
            async with ModelRegistryClient(server.config) as client:
                version = await client.get_model_version(version_id)
                return _format_version(version, Verbosity.FULL)

        try:
            return asyncio.run(_get())
        except ModelNotFoundError:
            return {"error": f"Version not found: {version_id}"}
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_model_artifacts(
        version_id: str,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """Get artifacts (storage URIs) for a model version.

        Artifacts contain the actual model files stored in object storage.
        Use this to find the storage location for model deployment.

        Args:
            version_id: The model version ID.
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            List of artifacts with storage URIs and format information.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> list[dict[str, Any]]:
            async with ModelRegistryClient(server.config) as client:
                artifacts = await client.get_model_artifacts(version_id)
                return [_format_artifact(a, Verbosity.from_str(verbosity)) for a in artifacts]

        try:
            artifacts = asyncio.run(_get())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        return {
            "version_id": version_id,
            "artifacts": artifacts,
            "count": len(artifacts),
        }


def _format_model(model: Any, verbosity: Verbosity) -> dict[str, Any]:
    """Format a registered model for response."""
    if verbosity == Verbosity.MINIMAL:
        return {
            "id": model.id,
            "name": model.name,
            "state": model.state,
        }

    result: dict[str, Any] = {
        "id": model.id,
        "name": model.name,
        "state": model.state,
        "owner": model.owner,
        "description": model.description,
    }

    if verbosity == Verbosity.FULL:
        result["custom_properties"] = model.custom_properties.properties
        if model.create_time:
            result["create_time"] = model.create_time.isoformat()
        if model.update_time:
            result["update_time"] = model.update_time.isoformat()

    return result


def _format_version(version: Any, verbosity: Verbosity) -> dict[str, Any]:
    """Format a model version for response."""
    if verbosity == Verbosity.MINIMAL:
        return {
            "id": version.id,
            "name": version.name,
            "state": version.state,
        }

    result: dict[str, Any] = {
        "id": version.id,
        "name": version.name,
        "registered_model_id": version.registered_model_id,
        "state": version.state,
        "author": version.author,
        "description": version.description,
    }

    if verbosity == Verbosity.FULL:
        result["custom_properties"] = version.custom_properties.properties
        if version.create_time:
            result["create_time"] = version.create_time.isoformat()
        if version.update_time:
            result["update_time"] = version.update_time.isoformat()

    return result


def _format_artifact(artifact: Any, verbosity: Verbosity) -> dict[str, Any]:
    """Format a model artifact for response."""
    if verbosity == Verbosity.MINIMAL:
        return {
            "id": artifact.id,
            "name": artifact.name,
            "uri": artifact.uri,
        }

    result: dict[str, Any] = {
        "id": artifact.id,
        "name": artifact.name,
        "uri": artifact.uri,
        "model_format_name": artifact.model_format_name,
        "model_format_version": artifact.model_format_version,
    }

    if verbosity == Verbosity.FULL:
        result["description"] = artifact.description
        result["storage_key"] = artifact.storage_key
        result["storage_path"] = artifact.storage_path
        result["service_account_name"] = artifact.service_account_name
        result["custom_properties"] = artifact.custom_properties.properties
        if artifact.create_time:
            result["create_time"] = artifact.create_time.isoformat()
        if artifact.update_time:
            result["update_time"] = artifact.update_time.isoformat()

    return result
