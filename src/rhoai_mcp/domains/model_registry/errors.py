"""Exceptions for Model Registry operations."""

from mcp.server.mcpserver.exceptions import ToolError


class ModelRegistryError(ToolError):
    """Base exception for Model Registry errors.

    Raised when there's a problem communicating with the Model Registry
    REST API or processing its responses.
    """

    pass


class ModelNotFoundError(ModelRegistryError):
    """Model or version not found in the registry.

    Raised when attempting to access a model, version, or artifact
    that doesn't exist in the registry.
    """

    pass


class ModelRegistryConnectionError(ModelRegistryError):
    """Failed to connect to the Model Registry service.

    Raised when the Model Registry service is unreachable or
    connection times out.
    """

    pass
