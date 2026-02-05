"""Model Registry auto-discovery service.

This module provides functionality to discover the Model Registry service
in an OpenShift AI cluster by querying the ModelRegistry component CRD
and falling back to common namespace/service patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kubernetes.client import ApiException  # type: ignore[import-untyped]

from rhoai_mcp.domains.model_registry.client import _is_running_in_cluster
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
    source: str  # "crd", "namespace_scan", "fallback", or "*_route" variants
    requires_auth: bool = False
    is_external: bool = field(default=False)  # True when using external Route
    route_name: str | None = field(default=None)  # Name of the Route if discovered

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

    def _find_route_for_service(self, service_name: str, namespace: str) -> tuple[str, str] | None:
        """Find an OpenShift Route that exposes the given service.

        Args:
            service_name: Name of the Kubernetes service to find a Route for.
            namespace: Namespace where the service and Route are located.

        Returns:
            Tuple of (route_name, external_url) if found, None otherwise.
        """
        try:
            routes = self._k8s.list_resources(ModelRegistryCRDs.ROUTE, namespace)
        except ApiException as e:
            logger.debug(f"Error listing Routes in {namespace}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error during Route discovery: {e}")
            return None

        if not routes:
            logger.debug(f"No Routes found in {namespace}")
            return None

        for route in routes:
            route_obj: Any = route
            spec = getattr(route_obj, "spec", None)
            if not spec:
                continue

            # Check if this Route targets our service
            to_ref = getattr(spec, "to", None)
            if not to_ref:
                continue

            to_kind = getattr(to_ref, "kind", None)
            to_name = getattr(to_ref, "name", None)

            if to_kind != "Service" or to_name != service_name:
                continue

            # Check if Route is admitted (ready to serve traffic)
            status = getattr(route_obj, "status", None)
            if not status:
                logger.debug(
                    f"Route {getattr(route_obj.metadata, 'name', 'unknown')} has no status"
                )
                continue

            ingress_list = getattr(status, "ingress", None) or []
            is_admitted = False
            host: str | None = None

            for ingress in ingress_list:
                conditions = getattr(ingress, "conditions", None) or []
                for condition in conditions:
                    cond_type = getattr(condition, "type", None)
                    cond_status = getattr(condition, "status", None)
                    if cond_type == "Admitted" and cond_status == "True":
                        is_admitted = True
                        host = getattr(ingress, "host", None)
                        break
                if is_admitted:
                    break

            if not is_admitted or not host:
                route_name = getattr(route_obj.metadata, "name", "unknown")
                logger.debug(f"Route {route_name} is not admitted or has no host")
                continue

            # Determine protocol based on TLS configuration
            tls = getattr(spec, "tls", None)
            protocol = "https" if tls else "http"

            route_name = getattr(route_obj.metadata, "name", service_name)
            external_url = f"{protocol}://{host}"

            logger.debug(f"Found Route {route_name} exposing {service_name} at {external_url}")
            return (route_name, external_url)

        logger.debug(f"No Route found exposing service {service_name} in {namespace}")
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

            # If running outside cluster, try to find an external Route
            if not _is_running_in_cluster():
                route_result = self._find_route_for_service(service_name, namespace)
                if route_result:
                    route_name, external_url = route_result
                    return DiscoveredModelRegistry(
                        url=external_url,
                        namespace=namespace,
                        service_name=service_name,
                        port=443,  # Routes typically use standard HTTPS port
                        source=f"{source}_route",
                        requires_auth=True,  # Routes typically use OAuth proxy
                        is_external=True,
                        route_name=route_name,
                    )
                else:
                    logger.warning(
                        f"Running outside cluster but no Route found for {service_name}. "
                        f"Internal URL will not be accessible."
                    )

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
