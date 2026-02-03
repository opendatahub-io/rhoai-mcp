"""Model Registry domain module.

This domain provides MCP tools for interacting with the OpenShift AI
Model Registry service. Unlike other domains that use Kubernetes CRDs,
the Model Registry uses a REST API.

Exports:
    Models:
        - RegisteredModel: Top-level model entity
        - ModelVersion: Version of a registered model
        - ModelArtifact: Storage artifact for a version
        - CustomProperties: Arbitrary key-value metadata

    Client:
        - ModelRegistryClient: Async HTTP client for the registry API

    Errors:
        - ModelRegistryError: Base exception
        - ModelNotFoundError: Model/version not found
        - ModelRegistryConnectionError: Connection failure

    Tools:
        - register_tools: Register MCP tools with server
"""

from rhoai_mcp.domains.model_registry.client import ModelRegistryClient
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
from rhoai_mcp.domains.model_registry.tools import register_tools

__all__ = [
    # Models
    "RegisteredModel",
    "ModelVersion",
    "ModelArtifact",
    "CustomProperties",
    # Client
    "ModelRegistryClient",
    # Errors
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelRegistryConnectionError",
    # Tools
    "register_tools",
]
