"""FastMCP server definition for RHOAI with domain modules and plugin discovery."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp_core.clients.base import K8sClient
from rhoai_mcp_core.config import RHOAIConfig, get_config

if TYPE_CHECKING:
    from rhoai_mcp_core.domains.registry import DomainModule
    from rhoai_mcp_core.plugin import RHOAIMCPPlugin

logger = logging.getLogger(__name__)

# Entry point group name for external plugin discovery
PLUGIN_ENTRY_POINT_GROUP = "rhoai_mcp.plugins"


class RHOAIServer:
    """RHOAI MCP Server with domain modules and external plugin discovery."""

    def __init__(self, config: RHOAIConfig | None = None) -> None:
        self._config = config or get_config()
        self._k8s_client: K8sClient | None = None
        self._mcp: FastMCP | None = None
        self._domains: list[DomainModule] = []
        self._plugins: dict[str, RHOAIMCPPlugin] = {}
        self._healthy_plugins: dict[str, RHOAIMCPPlugin] = {}
        self._healthy_domains: list[str] = []

    @property
    def config(self) -> RHOAIConfig:
        """Get server configuration."""
        return self._config

    @property
    def k8s(self) -> K8sClient:
        """Get the Kubernetes client.

        Raises:
            RuntimeError: If server is not running.
        """
        if self._k8s_client is None:
            raise RuntimeError("Server not running. K8s client not available.")
        return self._k8s_client

    @property
    def mcp(self) -> FastMCP:
        """Get the MCP server instance.

        Raises:
            RuntimeError: If server is not initialized.
        """
        if self._mcp is None:
            raise RuntimeError("Server not initialized.")
        return self._mcp

    @property
    def plugins(self) -> dict[str, RHOAIMCPPlugin]:
        """Get all discovered external plugins."""
        return self._plugins

    @property
    def healthy_plugins(self) -> dict[str, RHOAIMCPPlugin]:
        """Get external plugins that passed health checks."""
        return self._healthy_plugins

    def _discover_plugins(self) -> dict[str, RHOAIMCPPlugin]:
        """Discover and load external plugins from entry points.

        Note: Core domains are registered directly, not via entry points.
        This method only discovers external plugins like 'training'.

        Returns:
            Dictionary mapping plugin names to plugin instances.
        """
        plugins: dict[str, RHOAIMCPPlugin] = {}

        eps = entry_points(group=PLUGIN_ENTRY_POINT_GROUP)

        for ep in eps:
            try:
                logger.debug(f"Loading plugin from entry point: {ep.name}")
                factory = ep.load()
                plugin = factory()

                # Verify plugin has required interface
                if not hasattr(plugin, "metadata"):
                    logger.warning(f"Plugin {ep.name} does not have metadata, skipping")
                    continue

                plugins[plugin.metadata.name] = plugin
                logger.info(
                    f"Discovered external plugin: {plugin.metadata.name} v{plugin.metadata.version}"
                )
            except Exception as e:
                logger.error(f"Failed to load plugin {ep.name}: {e}")

        return plugins

    def _check_plugin_health(self) -> dict[str, RHOAIMCPPlugin]:
        """Check health of all discovered external plugins.

        Returns:
            Dictionary of plugins that passed health checks.
        """
        healthy: dict[str, RHOAIMCPPlugin] = {}

        for name, plugin in self._plugins.items():
            try:
                is_healthy, message = plugin.health_check(self)
                if is_healthy:
                    healthy[name] = plugin
                    logger.info(f"Plugin {name} health check passed: {message}")
                else:
                    logger.warning(f"Plugin {name} unavailable: {message}")
            except Exception as e:
                logger.warning(f"Plugin {name} health check failed with error: {e}")

        return healthy

    def _check_domain_health(self) -> list[str]:
        """Check health of core domain modules.

        Returns:
            List of domain names that passed health checks.
        """
        healthy: list[str] = []

        for domain in self._domains:
            try:
                if domain.health_check:
                    is_healthy, message = domain.health_check(self)
                    if is_healthy:
                        healthy.append(domain.name)
                        logger.info(f"Domain {domain.name} health check passed: {message}")
                    else:
                        logger.warning(f"Domain {domain.name} unavailable: {message}")
                else:
                    # Domains without required CRDs are always healthy
                    if not domain.required_crds:
                        healthy.append(domain.name)
                        logger.info(f"Domain {domain.name} active (no CRD requirements)")
                    else:
                        # Check if required CRDs are available
                        # For now, assume healthy and let tools fail gracefully
                        healthy.append(domain.name)
                        logger.info(f"Domain {domain.name} active")
            except Exception as e:
                logger.warning(f"Domain {domain.name} health check failed with error: {e}")

        return healthy

    def _create_lifespan(self) -> Callable[[Any], AbstractAsyncContextManager[None]]:
        """Create the lifespan context manager for the MCP server."""
        server_self = self

        @asynccontextmanager
        async def lifespan(_app: Any) -> AsyncIterator[None]:
            """Manage server lifecycle - connect K8s on startup, disconnect on shutdown."""
            logger.info("Starting RHOAI MCP server...")

            # Connect to Kubernetes
            server_self._k8s_client = K8sClient(server_self._config)
            try:
                server_self._k8s_client.connect()

                # Check domain and plugin health after K8s connection is established
                server_self._healthy_domains = server_self._check_domain_health()
                server_self._healthy_plugins = server_self._check_plugin_health()

                logger.info(
                    f"RHOAI MCP server started with {len(server_self._healthy_domains)}/{len(server_self._domains)} "
                    f"domains and {len(server_self._healthy_plugins)}/{len(server_self._plugins)} external plugins active"
                )
                yield
            finally:
                logger.info("Shutting down RHOAI MCP server...")
                if server_self._k8s_client:
                    server_self._k8s_client.disconnect()
                server_self._k8s_client = None
                logger.info("RHOAI MCP server shut down")

        return lifespan

    def create_mcp(self) -> FastMCP:
        """Create and configure the FastMCP server."""
        # Load core domain modules
        from rhoai_mcp_core.domains.registry import get_core_domains

        self._domains = get_core_domains()
        logger.info(f"Loaded {len(self._domains)} core domain modules")

        # Discover external plugins (like training)
        self._plugins = self._discover_plugins()
        logger.info(f"Discovered {len(self._plugins)} external plugins")

        # Create MCP server with lifespan
        # Host/port configured for container networking (0.0.0.0 allows external access)
        mcp = FastMCP(
            name="rhoai-mcp",
            instructions="MCP server for Red Hat OpenShift AI - enables AI agents to "
            "interact with RHOAI environments including workbenches, "
            "model serving, pipelines, and data connections.",
            lifespan=self._create_lifespan(),
            host=self._config.host,
            port=self._config.port,
        )

        # Store reference
        self._mcp = mcp

        # Register core domain modules directly (not via entry points)
        self._register_domain_modules(mcp)

        # Register tools and resources from external plugins
        self._register_plugin_tools(mcp)
        self._register_plugin_resources(mcp)

        # Register core resources (cluster status, etc.)
        self._register_core_resources(mcp)

        return mcp

    def _register_domain_modules(self, mcp: FastMCP) -> None:
        """Register MCP tools and resources from core domain modules."""
        for domain in self._domains:
            try:
                if domain.register_tools:
                    domain.register_tools(mcp, self)
                    logger.debug(f"Registered tools from domain: {domain.name}")
                if domain.register_resources:
                    domain.register_resources(mcp, self)
                    logger.debug(f"Registered resources from domain: {domain.name}")
            except Exception as e:
                logger.error(f"Failed to register domain {domain.name}: {e}")

        logger.info(f"Registered {len(self._domains)} core domain modules")

    def _register_plugin_tools(self, mcp: FastMCP) -> None:
        """Register MCP tools from all discovered external plugins."""
        for name, plugin in self._plugins.items():
            try:
                plugin.register_tools(mcp, self)
                logger.debug(f"Registered tools from plugin: {name}")
            except Exception as e:
                logger.error(f"Failed to register tools from plugin {name}: {e}")

        if self._plugins:
            logger.info(f"Registered tools from {len(self._plugins)} external plugins")

    def _register_plugin_resources(self, mcp: FastMCP) -> None:
        """Register MCP resources from all discovered external plugins."""
        for name, plugin in self._plugins.items():
            try:
                plugin.register_resources(mcp, self)
                logger.debug(f"Registered resources from plugin: {name}")
            except Exception as e:
                logger.error(f"Failed to register resources from plugin {name}: {e}")

        if self._plugins:
            logger.info(f"Registered resources from {len(self._plugins)} external plugins")

    def _register_core_resources(self, mcp: FastMCP) -> None:
        """Register core MCP resources for cluster information."""
        from rhoai_mcp_core.clients.base import CRDs

        @mcp.resource("rhoai://cluster/status")
        def cluster_status() -> dict:
            """Get RHOAI cluster status and health.

            Returns overall cluster status including RHOAI operator status,
            available components, and loaded domains/plugins.
            """
            k8s = self.k8s

            result: dict = {
                "connected": k8s.is_connected,
                "rhoai_available": False,
                "components": {},
                "domains": {
                    "total": len(self._domains),
                    "active": self._healthy_domains,
                },
                "plugins": {
                    "discovered": list(self._plugins.keys()),
                    "active": list(self._healthy_plugins.keys()),
                },
                "accelerators": [],
            }

            # Check for DataScienceCluster
            try:
                dsc_list = k8s.list_resources(CRDs.DATA_SCIENCE_CLUSTER)
                if dsc_list:
                    result["rhoai_available"] = True
                    dsc = dsc_list[0]
                    status = getattr(dsc, "status", None)
                    if status:
                        # Extract component status
                        installed = getattr(status, "installedComponents", {}) or {}
                        for component, state in installed.items():
                            result["components"][component] = state
            except Exception:
                pass

            # Check for accelerator profiles
            try:
                accelerators = k8s.list_resources(CRDs.ACCELERATOR_PROFILE)
                result["accelerators"] = [
                    {
                        "name": acc.metadata.name,
                        "display_name": (acc.metadata.annotations or {}).get(
                            "openshift.io/display-name", acc.metadata.name
                        ),
                        "enabled": getattr(acc.spec, "enabled", True)
                        if hasattr(acc, "spec")
                        else True,
                    }
                    for acc in accelerators
                ]
            except Exception:
                pass

            return result

        @mcp.resource("rhoai://cluster/plugins")
        def cluster_plugins() -> dict:
            """Get information about loaded domains and external plugins.

            Returns details about all core domains and external plugins
            with their health status.
            """
            domain_info = {}
            for domain in self._domains:
                domain_info[domain.name] = {
                    "description": domain.description,
                    "requires_crds": domain.required_crds,
                    "healthy": domain.name in self._healthy_domains,
                    "type": "core",
                }

            plugin_info = {}
            for name, plugin in self._plugins.items():
                meta = plugin.metadata
                is_healthy = name in self._healthy_plugins
                plugin_info[name] = {
                    "version": meta.version,
                    "description": meta.description,
                    "maintainer": meta.maintainer,
                    "requires_crds": meta.requires_crds,
                    "healthy": is_healthy,
                    "type": "external",
                }

            return {
                "total_domains": len(self._domains),
                "active_domains": len(self._healthy_domains),
                "total_plugins": len(self._plugins),
                "active_plugins": len(self._healthy_plugins),
                "domains": domain_info,
                "plugins": plugin_info,
            }

        @mcp.resource("rhoai://cluster/accelerators")
        def cluster_accelerators() -> list[dict]:
            """Get available accelerator profiles (GPUs).

            Returns the list of AcceleratorProfile resources that define
            available GPU types and configurations.
            """
            k8s = self.k8s

            try:
                accelerators = k8s.list_resources(CRDs.ACCELERATOR_PROFILE)
                return [
                    {
                        "name": acc.metadata.name,
                        "display_name": (acc.metadata.annotations or {}).get(
                            "openshift.io/display-name", acc.metadata.name
                        ),
                        "description": (acc.metadata.annotations or {}).get(
                            "openshift.io/description", ""
                        ),
                        "enabled": getattr(acc.spec, "enabled", True)
                        if hasattr(acc, "spec")
                        else True,
                        "identifier": getattr(acc.spec, "identifier", "nvidia.com/gpu")
                        if hasattr(acc, "spec")
                        else "nvidia.com/gpu",
                        "tolerations": getattr(acc.spec, "tolerations", [])
                        if hasattr(acc, "spec")
                        else [],
                    }
                    for acc in accelerators
                ]
            except Exception as e:
                return [{"error": str(e)}]

        logger.info("Registered core MCP resources")


# Global server instance
_server: RHOAIServer | None = None


def get_server() -> RHOAIServer:
    """Get the global server instance."""
    global _server
    if _server is None:
        _server = RHOAIServer()
    return _server


def create_server(config: RHOAIConfig | None = None) -> FastMCP:
    """Create and return the MCP server instance.

    This is the main entry point for creating the server.
    """
    global _server
    _server = RHOAIServer(config)
    return _server.create_mcp()
