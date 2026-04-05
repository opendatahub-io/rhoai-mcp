"""FastMCP server definition for RHOAI with pluggy-based plugin system."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from rhoai_mcp.clients.base import K8sClient
from rhoai_mcp.config import RHOAIConfig, get_config
from rhoai_mcp.plugin_manager import PluginManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RHOAIServer:
    """RHOAI MCP Server with pluggy-based plugin system."""

    def __init__(self, config: RHOAIConfig | None = None) -> None:
        self._config = config or get_config()
        self._k8s_client: K8sClient | None = None
        self._mcp: FastMCP | None = None
        self._plugin_manager: PluginManager | None = None

    @property
    def config(self) -> RHOAIConfig:
        """Get server configuration."""
        return self._config

    @property
    def k8s(self) -> K8sClient:
        """Get the Kubernetes client.

        When OIDC is enabled, returns an impersonating client for the
        current user. Otherwise returns the shared SA client.

        Raises:
            RuntimeError: If server is not running.
        """
        if self._k8s_client is None:
            raise RuntimeError("Server not running. K8s client not available.")

        if self._config.oidc_enabled:
            from rhoai_mcp.auth.user_context import UserContext

            ctx = UserContext.current()
            if ctx is None:
                raise RuntimeError(
                    "OIDC is enabled but no UserContext is set. "
                    "Refusing to fall back to service-account client."
                )
            return self._k8s_client.create_impersonating_client(ctx.username, ctx.groups)

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
    def plugin_manager(self) -> PluginManager:
        """Get the plugin manager.

        Raises:
            RuntimeError: If server is not initialized.
        """
        if self._plugin_manager is None:
            raise RuntimeError("Server not initialized.")
        return self._plugin_manager

    @property
    def plugins(self) -> dict[str, Any]:
        """Get all registered plugins."""
        if self._plugin_manager is None:
            return {}
        return self._plugin_manager.registered_plugins

    @property
    def healthy_plugins(self) -> dict[str, Any]:
        """Get plugins that passed health checks."""
        if self._plugin_manager is None:
            return {}
        return self._plugin_manager.healthy_plugins

    def get_allowed_tools(self) -> set[str] | None:
        """Get the set of tool names the current user can access.

        Returns None when OIDC is disabled (all tools allowed).
        Returns a set of tool names when OIDC is enabled.
        """
        if not self._config.oidc_enabled:
            return None

        from rhoai_mcp.auth.rbac import RBACChecker, ToolPermission
        from rhoai_mcp.auth.user_context import UserContext

        ctx = UserContext.current()
        if ctx is None:
            return set()  # No user context = no tools

        # Collect tool permission mappings from plugins
        if not self._plugin_manager:
            raise RuntimeError("OIDC enabled but plugin_manager not initialized")

        raw_perms = self._plugin_manager.collect_tool_permissions()
        if not raw_perms:
            logger.warning("get_allowed_tools: no permission mappings, allowing all tools")
            return None

        # Convert to ToolPermission objects
        tool_perms: dict[str, list[ToolPermission]] = {}
        for tool_name, perm_dicts in raw_perms.items():
            tool_perms[tool_name] = [ToolPermission.from_dict(p) for p in perm_dicts]

        # Create RBAC checker using the SA's API client (not impersonating)
        from kubernetes import client as k8s_client  # type: ignore[import-untyped]

        assert self._k8s_client is not None  # guaranteed by startup()
        authz_api = k8s_client.AuthorizationV1Api(self._k8s_client._api_client)
        checker = RBACChecker(authz_api)

        return checker.filter_tools(ctx.username, ctx.groups, tool_perms)

    def _init_k8s_client(self) -> None:
        """Initialize and connect the Kubernetes client.

        Idempotent: if a K8s client is already connected (e.g. a mock
        injected for testing), it is preserved.
        """
        if self._k8s_client is not None and self._k8s_client.is_connected:
            return  # Already initialized and connected

        if self._config.mock_cluster:
            from rhoai_mcp.mock_k8s import MockK8sClient, create_default_cluster_state

            state = create_default_cluster_state()
            mock_client = MockK8sClient(config_obj=self._config, state=state)
            mock_client.connect()
            self._k8s_client = mock_client
            logger.info("Using mock K8s client with pre-populated cluster state")
        else:
            self._k8s_client = K8sClient(self._config)
            self._k8s_client.connect()

    def _create_lifespan(self) -> Callable[[Any], AbstractAsyncContextManager[None]]:
        """Create the lifespan context manager for MCP sessions.

        Note: K8s connection is established eagerly in create_mcp(), not here.
        FastMCP's lifespan runs per MCP client session (inside Server.run()),
        not at ASGI app startup. Putting K8s connect here would block the
        /health endpoint until the first MCP client connects.
        """

        @asynccontextmanager
        async def lifespan(_app: Any) -> AsyncIterator[None]:
            """Per-session lifespan - K8s is already connected."""
            logger.debug("MCP session started")
            try:
                yield
            finally:
                logger.debug("MCP session ended")
                # Close any port-forward connections opened during this session
                try:
                    from rhoai_mcp.utils.port_forward import PortForwardManager

                    await PortForwardManager.get_instance().close_all()
                except Exception as e:
                    logger.warning(f"Error closing port-forward connections: {e}")

        return lifespan

    def startup(self) -> None:
        """Connect to Kubernetes and run plugin health checks.

        Called eagerly during create_mcp() so the /health endpoint works
        immediately, before any MCP client connects.

        Idempotent: if a K8s client is already connected (e.g. a mock
        injected for testing), it is preserved. Health checks run if a plugin
        manager has been initialised.
        """
        self._init_k8s_client()

        if self._plugin_manager:
            self._plugin_manager.run_health_checks(self)

        pm = self._plugin_manager
        total = len(pm.registered_plugins) if pm else 0
        healthy = len(pm.healthy_plugins) if pm else 0
        logger.info(f"RHOAI MCP server ready with {healthy}/{total} plugins active")

    def shutdown(self) -> None:
        """Disconnect from Kubernetes and clean up resources."""
        logger.info("Shutting down RHOAI MCP server...")
        if self._k8s_client:
            self._k8s_client.disconnect()
            self._k8s_client = None
        logger.info("RHOAI MCP server shut down")

    def create_mcp(self) -> FastMCP:
        """Create and configure the FastMCP server."""
        # Create plugin manager
        self._plugin_manager = PluginManager()

        # Load core domain plugins (filtered by config if set)
        core_count = self._plugin_manager.load_core_plugins(
            enabled_plugins=self._config.enabled_plugins,
        )
        logger.info(f"Loaded {core_count} core domain plugins")

        # Discover and load external plugins
        external_count = self._plugin_manager.load_entrypoint_plugins()
        logger.info(f"Discovered {external_count} external plugins")

        # Connect to Kubernetes eagerly so /health works before any MCP session
        self.startup()

        # Create MCP server with lifespan
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

        # Register tools, resources, and prompts from all plugins
        self._plugin_manager.register_all_tools(mcp, self)
        self._plugin_manager.register_all_resources(mcp, self)
        self._plugin_manager.register_all_prompts(mcp, self)

        # Register core resources (cluster status, etc.)
        self._register_core_resources(mcp)

        # Register health endpoint for Kubernetes probes
        self._register_health_endpoint(mcp)

        # Register OIDC auth components if enabled
        if self._config.oidc_enabled:
            self._setup_oidc_auth(mcp)

        return mcp

    def _setup_oidc_auth(self, mcp: FastMCP) -> None:
        """Configure OIDC authentication middleware and endpoints."""
        from rhoai_mcp.auth.metadata import build_protected_resource_metadata
        from rhoai_mcp.auth.oidc import OIDCValidator

        self._config.validate_oidc_config()
        # issuer_url is guaranteed non-None here by validate_oidc_config()
        assert self._config.oidc_issuer_url is not None
        issuer_url = self._config.oidc_issuer_url

        # Create OIDC validator
        validator = OIDCValidator(
            issuer_url=issuer_url,
            audience=self._config.oidc_audience,
            username_claim=self._config.oidc_username_claim,
            groups_claim=self._config.oidc_groups_claim,
            jwks_cache_ttl=self._config.oidc_jwks_cache_ttl,
        )

        # Build resource metadata URL
        resource_url = f"https://{self._config.host}:{self._config.port}"
        metadata_path = "/.well-known/oauth-protected-resource"

        # Register Protected Resource Metadata endpoint
        @mcp.custom_route(metadata_path, methods=["GET"])
        async def protected_resource_metadata(request: Request) -> JSONResponse:  # noqa: ARG001
            meta = build_protected_resource_metadata(
                resource_url=resource_url,
                issuer_url=issuer_url,
                scopes=self._config.oidc_required_scopes,
            )
            return JSONResponse(meta)

        # Add auth middleware
        exclude_paths = ["/health", metadata_path]
        self._oidc_resource_metadata_url = f"{resource_url}{metadata_path}"

        from rhoai_mcp.auth.middleware import OIDCAuthMiddleware

        starlette_app = getattr(mcp, "_app", None) or getattr(mcp, "app", None)
        if starlette_app and hasattr(starlette_app, "add_middleware"):
            starlette_app.add_middleware(
                OIDCAuthMiddleware,
                validator=validator,
                exclude_paths=exclude_paths,
                resource_metadata_url=self._oidc_resource_metadata_url,
            )
        else:
            logger.warning(
                "Could not attach OIDC middleware to FastMCP app. "
                "Check FastMCP version for ASGI app access."
            )

        logger.info("OIDC authentication enabled")

    def _register_core_resources(self, mcp: FastMCP) -> None:
        """Register core MCP resources for cluster information."""
        from rhoai_mcp.clients.base import CRDs

        @mcp.resource("rhoai://cluster/status")
        def cluster_status() -> dict:
            """Get RHOAI cluster status and health.

            Returns overall cluster status including RHOAI operator status,
            available components, and loaded plugins.
            """
            k8s = self.k8s
            pm = self._plugin_manager

            result: dict = {
                "connected": k8s.is_connected,
                "rhoai_available": False,
                "components": {},
                "plugins": {
                    "total": len(pm.registered_plugins) if pm else 0,
                    "active": list(pm.healthy_plugins.keys()) if pm else [],
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
            """Get information about loaded plugins.

            Returns details about all plugins with their health status.
            """
            pm = self._plugin_manager
            if not pm:
                return {"plugins": {}}

            plugin_info = {}
            for name, plugin in pm.registered_plugins.items():
                is_healthy = name in pm.healthy_plugins

                # Get metadata if available
                meta = None
                if hasattr(plugin, "rhoai_get_plugin_metadata"):
                    meta = plugin.rhoai_get_plugin_metadata()

                plugin_info[name] = {
                    "version": meta.version if meta else "unknown",
                    "description": meta.description if meta else "No description",
                    "maintainer": meta.maintainer if meta else "unknown",
                    "requires_crds": meta.requires_crds if meta else [],
                    "healthy": is_healthy,
                }

            return {
                "total": len(pm.registered_plugins),
                "active": len(pm.healthy_plugins),
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

    def _register_health_endpoint(self, mcp: FastMCP) -> None:
        """Register /health endpoint for Kubernetes liveness/readiness probes."""

        @mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> JSONResponse:  # noqa: ARG001
            """Health check endpoint for Kubernetes probes.

            Returns 200 when the server is ready (K8s connected), or 503 when
            not yet ready. This allows Kubernetes to use the endpoint for both
            liveness (process is running) and readiness (able to serve requests).
            """
            kubernetes_client = self._k8s_client
            plugin_manager = self._plugin_manager

            is_connected = kubernetes_client.is_connected if kubernetes_client else False
            total = len(plugin_manager.registered_plugins) if plugin_manager else 0
            healthy = len(plugin_manager.healthy_plugins) if plugin_manager else 0
            is_ready = is_connected

            status_code = HTTPStatus.OK if is_ready else HTTPStatus.SERVICE_UNAVAILABLE  # 200 / 503
            return JSONResponse(
                {
                    "status": "healthy" if is_ready else "unhealthy",
                    "connected": is_connected,
                    "plugins": {
                        "total": total,
                        "healthy": healthy,
                    },
                },
                status_code=status_code,
            )

        logger.info("Registered /health endpoint for Kubernetes probes")


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
