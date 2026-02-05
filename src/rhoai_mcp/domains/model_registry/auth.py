"""Authentication utilities for Model Registry clients.

This module provides shared authentication functions used by both the
ModelRegistryClient and ModelCatalogClient to avoid code duplication.

When running outside the cluster, port-forwarding is used to access the
Model Registry service directly on port 8080, bypassing OAuth authentication.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rhoai_mcp.config import RHOAIConfig

logger = logging.getLogger(__name__)


def _is_running_in_cluster() -> bool:
    """Check if we're running inside a Kubernetes cluster.

    Returns:
        True if running in-cluster (as a pod), False otherwise.
    """
    return Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()


def _get_in_cluster_token() -> str | None:
    """Get the service account token when running in-cluster.

    Returns:
        The service account token, or None if not running in-cluster.
    """
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    if token_path.exists():
        try:
            return token_path.read_text().strip()
        except Exception as e:
            logger.debug(f"Error reading in-cluster token: {e}")
    return None


def build_auth_headers(
    config: RHOAIConfig,
    requires_auth_override: bool = False,
) -> dict[str, str]:
    """Build authentication headers for Model Registry API calls.

    This function consolidates the auth header construction logic used by
    ModelRegistryClient, ModelCatalogClient, and the probe_api_type function.

    When running outside the cluster, port-forwarding is typically used to
    access the service directly on port 8080, which does not require
    authentication. This function primarily handles in-cluster SA token
    auth and explicit token configuration.

    Args:
        config: RHOAI configuration with auth settings.
        requires_auth_override: If True, attempt auth even if auth_mode is NONE.
            Used when discovery indicates the endpoint requires authentication.

    Returns:
        Dict of headers to include in requests.
    """
    from rhoai_mcp.config import ModelRegistryAuthMode

    auth_mode = config.model_registry_auth_mode
    headers: dict[str, str] = {}

    # Check if auth is required
    if auth_mode == ModelRegistryAuthMode.NONE and not requires_auth_override:
        return headers

    token: str | None = None

    if auth_mode == ModelRegistryAuthMode.TOKEN:
        # Use explicit token from config
        token = config.model_registry_token
        if not token:
            logger.warning(
                "Model Registry auth_mode is 'token' but no token configured. "
                "Set RHOAI_MCP_MODEL_REGISTRY_TOKEN environment variable."
            )

    elif auth_mode == ModelRegistryAuthMode.OAUTH or requires_auth_override:
        # When running in-cluster, use the service account token
        if _is_running_in_cluster():
            token = _get_in_cluster_token()
            if not token:
                logger.warning(
                    "Model Registry auth_mode is 'oauth' but no in-cluster token found."
                )
        else:
            # Outside cluster: port-forwarding should handle access without auth
            # Only warn if auth is explicitly required
            if requires_auth_override:
                logger.debug(
                    "Running outside cluster with requires_auth=True. "
                    "Port-forwarding to internal service should bypass auth."
                )

    if token:
        headers["Authorization"] = f"Bearer {token}"
        logger.debug("Added Authorization header for Model Registry")

    return headers
