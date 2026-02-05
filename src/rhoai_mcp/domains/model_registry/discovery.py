"""Model Registry auto-discovery service.

This module provides functionality to discover the Model Registry service
in an OpenShift AI cluster by querying the ModelRegistry component CRD
and falling back to common namespace/service patterns.

When running outside the cluster, it uses `oc port-forward` to tunnel
directly to the Model Registry service, bypassing the need for external
Routes and OAuth authentication.

It also supports detecting whether the discovered service is a standard
Kubeflow Model Registry or a Red Hat AI Model Catalog.
"""

from __future__ import annotations

import logging
import ssl
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx
from kubernetes.client import ApiException  # type: ignore[import-untyped]

from rhoai_mcp.domains.model_registry.auth import (
    _is_running_in_cluster,
    build_auth_headers,
)
from rhoai_mcp.domains.model_registry.crds import ModelRegistryCRDs
from rhoai_mcp.utils.port_forward import PortForwardConnection, PortForwardError, PortForwardManager

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient
    from rhoai_mcp.config import RHOAIConfig

logger = logging.getLogger(__name__)

# Common namespace patterns where Model Registry is deployed
COMMON_NAMESPACES = [
    "rhoai-model-registries",
    "odh-model-registries",
    "model-registries",
]

# Service name patterns to match (in priority order)
SERVICE_NAME_PATTERNS = [
    "model-catalog",
    "model-registry",
    "modelregistry",
]

# Port preference (lower index = higher priority)
# 8080: Direct REST API (no auth overhead)
# 8443: kube-rbac-proxy (requires service account auth)
# 443: HTTPS endpoint
PREFERRED_PORTS = [8080, 8443, 443]


@dataclass
class DiscoveredModelRegistry:
    """Result of Model Registry discovery."""

    url: str
    namespace: str
    service_name: str
    port: int
    source: str  # "crd", "namespace_scan", "fallback", or "*_port_forward" variants
    requires_auth: bool = False
    is_external: bool = field(default=False)  # True when using port-forward
    api_type: str = field(default="unknown")  # "model_catalog", "model_registry", or "unknown"
    port_forward_connection: PortForwardConnection | None = field(default=None)

    def __str__(self) -> str:
        return f"{self.url} (discovered via {self.source}, api_type={self.api_type})"


class ModelRegistryDiscovery:
    """Discovers Model Registry service in the cluster.

    Discovery strategy:
    1. Query ModelRegistry component CRD for spec.registriesNamespace
    2. Fall back to common namespace patterns
    3. Find services matching known Model Registry patterns
    4. Prefer port 8080 (direct REST) over 8443 (kube-rbac-proxy)
    """

    def __init__(self, k8s: K8sClient) -> None:
        self._k8s = k8s

    def discover(self, fallback_url: str | None = None) -> DiscoveredModelRegistry | None:
        """Discover the Model Registry service.

        Args:
            fallback_url: URL to use if discovery fails

        Returns:
            DiscoveredModelRegistry if found, None otherwise
        """
        # Try to discover from CRD first
        result = self._discover_from_crd()
        if result:
            logger.info(f"Discovered Model Registry from CRD: {result}")
            return result

        # Fall back to scanning common namespaces
        result = self._discover_from_namespaces()
        if result:
            logger.info(f"Discovered Model Registry from namespace scan: {result}")
            return result

        # Use fallback URL if provided
        if fallback_url:
            logger.info(f"Using fallback Model Registry URL: {fallback_url}")
            return DiscoveredModelRegistry(
                url=fallback_url,
                namespace="unknown",
                service_name="unknown",
                port=8080,
                source="fallback",
                requires_auth=False,
            )

        logger.warning("Model Registry discovery failed and no fallback URL provided")
        return None

    def _discover_from_crd(self) -> DiscoveredModelRegistry | None:
        """Discover Model Registry from the component CRD."""
        try:
            resources = self._k8s.list_resources(ModelRegistryCRDs.MODEL_REGISTRY_COMPONENT)
            if not resources:
                logger.debug("No ModelRegistry component CRD found")
                return None

            # Get the first (and typically only) ModelRegistry component
            component = resources[0]
            spec = getattr(component, "spec", None)
            if not spec:
                logger.debug("ModelRegistry component has no spec")
                return None

            # Get the registries namespace from the component spec
            registries_namespace = getattr(spec, "registriesNamespace", None)
            if not registries_namespace:
                logger.debug("ModelRegistry component has no registriesNamespace")
                return None

            logger.debug(f"Found registries namespace from CRD: {registries_namespace}")

            # Find services in the namespace
            return self._find_service_in_namespace(registries_namespace, source="crd")

        except ApiException as e:
            if e.status == 404:
                logger.debug("ModelRegistry CRD not installed in cluster")
            else:
                logger.debug(f"Error querying ModelRegistry CRD: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error during CRD discovery: {e}")
            return None

    def _discover_from_namespaces(self) -> DiscoveredModelRegistry | None:
        """Discover Model Registry by scanning common namespaces."""
        for namespace in COMMON_NAMESPACES:
            try:
                result = self._find_service_in_namespace(namespace, source="namespace_scan")
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Error scanning namespace {namespace}: {e}")
                continue

        return None

    def _find_service_in_namespace(
        self, namespace: str, source: str
    ) -> DiscoveredModelRegistry | None:
        """Find Model Registry service in a namespace."""
        try:
            services = self._k8s.core_v1.list_namespaced_service(namespace=namespace)
        except ApiException as e:
            if e.status == 404 or e.status == 403:
                logger.debug(f"Cannot access namespace {namespace}: {e.status}")
            else:
                logger.debug(f"Error listing services in {namespace}: {e}")
            return None

        # Find matching services
        matching_services = []
        for svc in services.items:
            svc_name = svc.metadata.name.lower()
            for pattern in SERVICE_NAME_PATTERNS:
                if pattern in svc_name:
                    matching_services.append(svc)
                    break

        if not matching_services:
            logger.debug(f"No matching Model Registry services in {namespace}")
            return None

        # Find the best service/port combination
        best_service = None
        best_port = None
        best_port_priority = len(PREFERRED_PORTS)  # Lower is better

        for svc in matching_services:
            for port_spec in svc.spec.ports or []:
                port_num = port_spec.port
                try:
                    priority = PREFERRED_PORTS.index(port_num)
                    if priority < best_port_priority:
                        best_port_priority = priority
                        best_port = port_num
                        best_service = svc
                except ValueError:
                    # Port not in preferred list, use if nothing else found
                    if best_service is None:
                        best_service = svc
                        best_port = port_num

        if best_service and best_port:
            service_name = best_service.metadata.name

            # Use internal service URL
            # Determine if auth is required (8443 uses kube-rbac-proxy)
            requires_auth = best_port == 8443

            # Use HTTP for 8080, HTTPS for other ports
            protocol = "http" if best_port == 8080 else "https"
            url = f"{protocol}://{service_name}.{namespace}.svc:{best_port}"

            return DiscoveredModelRegistry(
                url=url,
                namespace=namespace,
                service_name=service_name,
                port=best_port,
                source=source,
                requires_auth=requires_auth,
            )

        return None

    async def discover_with_port_forward(
        self, fallback_url: str | None = None
    ) -> DiscoveredModelRegistry | None:
        """Discover the Model Registry service, using port-forward when outside cluster.

        This method first discovers the service using the standard discovery flow,
        then sets up port-forwarding if running outside the cluster.

        Args:
            fallback_url: URL to use if discovery fails

        Returns:
            DiscoveredModelRegistry with accessible URL (port-forwarded if outside cluster)
        """
        # First, use standard discovery to find the service
        result = self.discover(fallback_url)
        if not result:
            return None

        # If we're running in-cluster, use the internal URL directly
        if _is_running_in_cluster():
            logger.debug("Running in-cluster, using internal service URL")
            return result

        # If this is a fallback URL (not a discovered service), use it directly
        if result.source == "fallback":
            logger.debug("Using fallback URL, no port-forward needed")
            return result

        # Running outside cluster - set up port-forward
        try:
            manager = PortForwardManager.get_instance()
            conn = await manager.forward(
                namespace=result.namespace,
                service_name=result.service_name,
                remote_port=result.port,
            )

            # Return discovery result with port-forwarded URL
            return DiscoveredModelRegistry(
                url=conn.local_url,
                namespace=result.namespace,
                service_name=result.service_name,
                port=result.port,
                source=f"{result.source}_port_forward",
                requires_auth=False,  # Port-forward bypasses auth
                is_external=True,
                port_forward_connection=conn,
            )
        except PortForwardError as e:
            logger.error(f"Failed to set up port-forward: {e}")
            logger.warning(
                f"Running outside cluster but port-forward failed for "
                f"{result.service_name}.{result.namespace}. "
                f"Internal URL will not be accessible."
            )
            return None


async def probe_api_type(
    url: str,
    config: RHOAIConfig,
    requires_auth: bool = False,
) -> str:
    """Probe which API type is available at the given URL.

    Tries Model Catalog API first, then falls back to Model Registry API.
    This detection allows seamless use of either API type.

    Args:
        url: Base URL of the service.
        config: RHOAI configuration for auth and TLS settings.
        requires_auth: Whether the endpoint requires authentication.

    Returns:
        "model_catalog" if Model Catalog API is available,
        "model_registry" if standard Model Registry API is available,
        "unknown" if neither responds successfully.
    """
    # Build auth headers using shared utility
    headers = build_auth_headers(config, requires_auth_override=requires_auth)

    # Configure SSL
    verify: bool | ssl.SSLContext = True
    if config.model_registry_skip_tls_verify:
        verify = False

    async with httpx.AsyncClient(
        base_url=url,
        timeout=10,  # Short timeout for probing
        headers=headers,
        verify=verify,
    ) as client:
        # Try Model Catalog API first
        try:
            response = await client.get(
                "/api/model_catalog/v1alpha1/models",
                params={"pageSize": 1},
            )
            if response.status_code == 200:
                logger.info(f"Detected Model Catalog API at {url}")
                return "model_catalog"
        except Exception as e:
            logger.debug(f"Model Catalog probe failed: {e}")

        # Try standard Model Registry API
        try:
            response = await client.get(
                "/api/model_registry/v1alpha3/registered_models",
                params={"pageSize": 1},
            )
            if response.status_code == 200:
                logger.info(f"Detected Model Registry API at {url}")
                return "model_registry"
        except Exception as e:
            logger.debug(f"Model Registry probe failed: {e}")

    logger.warning(f"Could not detect API type at {url}")
    return "unknown"
