"""Model Registry auto-discovery service.

This module provides functionality to discover the Model Registry service
in an OpenShift AI cluster by querying the ModelRegistry component CRD
and falling back to common namespace/service patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kubernetes.client import ApiException  # type: ignore[import-untyped]

from rhoai_mcp.domains.model_registry.crds import ModelRegistryCRDs

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

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
    source: str  # "crd", "namespace_scan", or "fallback"
    requires_auth: bool = False

    def __str__(self) -> str:
        return f"{self.url} (discovered via {self.source})"


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
            # Determine if auth is required (8443 uses kube-rbac-proxy)
            requires_auth = best_port == 8443

            # Use HTTP for 8080, HTTPS for other ports
            protocol = "http" if best_port == 8080 else "https"
            url = f"{protocol}://{best_service.metadata.name}.{namespace}.svc:{best_port}"

            return DiscoveredModelRegistry(
                url=url,
                namespace=namespace,
                service_name=best_service.metadata.name,
                port=best_port,
                source=source,
                requires_auth=requires_auth,
            )

        return None
