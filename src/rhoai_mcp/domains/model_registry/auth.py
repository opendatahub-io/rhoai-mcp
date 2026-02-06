"""Authentication utilities for Model Registry clients.

This module provides shared authentication functions used by both the
ModelRegistryClient and ModelCatalogClient to avoid code duplication.

When running outside the cluster with port 8080, port-forwarding bypasses
OAuth authentication. When using port 8443 (kube-rbac-proxy), authentication
is still required and is obtained from the oc/kubectl CLI.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
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


def _get_cli_token() -> str | None:
    """Get authentication token from oc or kubectl CLI.

    This is used when running outside the cluster and needing to
    authenticate to services that require it (e.g., kube-rbac-proxy on 8443).

    Returns:
        The authentication token, or None if not available.
    """
    # Try oc first (OpenShift), then kubectl
    for cli in ("oc", "kubectl"):
        cli_path = shutil.which(cli)
        if not cli_path:
            continue

        try:
            if cli == "oc":
                # oc whoami -t returns the token directly
                result = subprocess.run(
                    [cli_path, "whoami", "-t"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    logger.debug(f"Got auth token from {cli}")
                    return result.stdout.strip()
            else:
                # kubectl config view to get token from kubeconfig
                # This is more complex; for now, we rely on oc
                pass
        except subprocess.TimeoutExpired:
            logger.debug(f"Timeout getting token from {cli}")
        except Exception as e:
            logger.debug(f"Error getting token from {cli}: {e}")

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
                logger.warning("Model Registry auth_mode is 'oauth' but no in-cluster token found.")
        else:
            # Outside cluster: get token from oc/kubectl CLI
            # This is needed for port 8443 (kube-rbac-proxy) which requires auth
            token = _get_cli_token()
            if not token and requires_auth_override:
                logger.warning(
                    "Running outside cluster with requires_auth=True but no CLI token found. "
                    "Ensure you are logged in with 'oc login' or have valid kubeconfig."
                )

    if token:
        headers["Authorization"] = f"Bearer {token}"
        logger.debug("Added Authorization header for Model Registry")

    return headers
