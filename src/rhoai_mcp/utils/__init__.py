"""Utility functions and helpers for RHOAI MCP server."""

from rhoai_mcp.utils.annotations import RHOAIAnnotations
from rhoai_mcp.utils.errors import (
    AuthenticationError,
    ConfigurationError,
    NotFoundError,
    OperationNotAllowedError,
    ResourceExistsError,
    RHOAIError,
    ValidationError,
)
from rhoai_mcp.utils.labels import RHOAILabels
from rhoai_mcp.utils.port_forward import (
    PortForwardConnection,
    PortForwardError,
    PortForwardManager,
)
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    ResponseBuilder,
    Verbosity,
    paginate,
)

__all__ = [
    # Errors
    "RHOAIError",
    "NotFoundError",
    "AuthenticationError",
    "ConfigurationError",
    "ValidationError",
    "OperationNotAllowedError",
    "ResourceExistsError",
    # Labels and annotations
    "RHOAIAnnotations",
    "RHOAILabels",
    # Response formatting
    "Verbosity",
    "ResponseBuilder",
    "PaginatedResponse",
    "paginate",
    # Port forwarding
    "PortForwardConnection",
    "PortForwardError",
    "PortForwardManager",
]
