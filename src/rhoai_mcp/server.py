"""FastMCP server definition for RHOAI with pluggy-based plugin system."""

from __future__ import annotations

import asyncio
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

    def get_allowed_tools(self) -> tuple[set[str], set[str]] | None:
        """Get the set of tool names the current user can access.

        Returns None when OIDC is disabled (all tools allowed).
        Returns (allowed_tools, governed_tools) when OIDC is enabled.
        ``governed_tools`` is the set of tool names that have RBAC permission
        mappings.  Tools not in this set have no mapping and should be allowed
        by default.
        """
        if not self._config.oidc_enabled:
            logger.debug("get_allowed_tools: OIDC disabled, returning None")
            return None

        from rhoai_mcp.auth.rbac import RBACChecker, ToolPermission
        from rhoai_mcp.auth.user_context import UserContext

        ctx = UserContext.current()
        if ctx is None:
            logger.warning("get_allowed_tools: no UserContext, returning empty set")
            return set(), set()  # No user context = no tools

        logger.debug("get_allowed_tools: user=%s groups=%s", ctx.username, ctx.groups)

        # Collect tool permission mappings from plugins
        if not self._plugin_manager:
            raise RuntimeError("OIDC enabled but plugin_manager not initialized")

        raw_perms = self._plugin_manager.collect_tool_permissions()
        if not raw_perms:
            logger.warning("get_allowed_tools: no permission mappings, all tools ungoverned")
            return set(), set()

        # Convert to ToolPermission objects
        tool_perms: dict[str, list[ToolPermission]] = {}
        for tool_name, perm_dicts in raw_perms.items():
            tool_perms[tool_name] = [ToolPermission.from_dict(p) for p in perm_dicts]

        governed_tools = set(tool_perms.keys())

        # Create RBAC checker using the SA's API client (not impersonating)
        from kubernetes import client as k8s_client  # type: ignore[import-untyped]

        assert self._k8s_client is not None  # guaranteed by startup()
        authz_api = k8s_client.AuthorizationV1Api(self._k8s_client._api_client)
        checker = RBACChecker(authz_api)

        allowed = checker.filter_tools(ctx.username, ctx.groups, tool_perms)
        return allowed, governed_tools

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
            self._setup_auth(mcp)

        return mcp

    def _setup_auth(self, mcp: FastMCP) -> None:
        """Configure token validation middleware and metadata endpoint."""
        from rhoai_mcp.auth.metadata import build_protected_resource_metadata
        from rhoai_mcp.auth.oidc import OIDCValidator
        from rhoai_mcp.auth.token_review import TokenReviewValidator
        from rhoai_mcp.config import OIDCTokenMode

        self._config.validate_oidc_config()

        # Ensure K8s client is connected
        if not self._k8s_client or not self._k8s_client.is_connected:
            raise RuntimeError("K8s client not connected. Call startup() first.")

        # Create the appropriate validator and determine issuer URL for metadata
        validator: OIDCValidator | TokenReviewValidator
        if self._config.oidc_token_mode == OIDCTokenMode.TOKEN_REVIEW:
            api_client = self._k8s_client._api_client
            if not api_client:
                raise RuntimeError("K8s client API client not available")
            validator = TokenReviewValidator(api_client)
            # Issuer URL for metadata: explicit config, or auto-detect from K8s client
            issuer_url = self._config.oidc_ocp_api_url or api_client.configuration.host
            logger.info("Auth mode: TokenReview (OCP OAuth)")
        else:
            # JWT mode: issuer_url is guaranteed non-None by validate_oidc_config()
            assert self._config.oidc_issuer_url is not None
            issuer_url = self._config.oidc_issuer_url
            validator = OIDCValidator(
                issuer_url=issuer_url,
                audience=self._config.oidc_audience,
                username_claim=self._config.oidc_username_claim,
                groups_claim=self._config.oidc_groups_claim,
                jwks_cache_ttl=self._config.oidc_jwks_cache_ttl,
            )
            logger.info("Auth mode: OIDC JWT")

        # Build resource metadata URL (external Route URL, or fallback to listen address)
        resource_url = (
            self._config.oidc_resource_url or f"https://{self._config.host}:{self._config.port}"
        )
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

        # Wrap sse_app/streamable_http_app with auth middleware.
        # Uses pure ASGI wrapping (not add_middleware) because
        # BaseHTTPMiddleware is incompatible with SSE streaming.
        exclude_paths = ["/health", metadata_path]
        self._oidc_resource_metadata_url = f"{resource_url}{metadata_path}"

        from rhoai_mcp.auth.middleware import OIDCAuthMiddleware

        original_sse_app = mcp.sse_app

        def patched_sse_app(mount_path: str | None = None) -> Any:
            app = original_sse_app(mount_path)
            return OIDCAuthMiddleware(
                app,
                validator=validator,
                exclude_paths=exclude_paths,
                resource_metadata_url=self._oidc_resource_metadata_url,
            )

        mcp.sse_app = patched_sse_app  # type: ignore[method-assign]

        if hasattr(mcp, "streamable_http_app"):
            original_http_app = mcp.streamable_http_app

            def patched_http_app() -> Any:
                app = original_http_app()
                return OIDCAuthMiddleware(
                    app,
                    validator=validator,
                    exclude_paths=exclude_paths,
                    resource_metadata_url=self._oidc_resource_metadata_url,
                )

            mcp.streamable_http_app = patched_http_app  # type: ignore[method-assign]

        logger.info("Auth middleware will be attached when transport app is created")

        # Install tool-level RBAC filtering
        self._install_tool_filtering(mcp)

        logger.info("Authentication enabled")

    def _install_tool_filtering(self, mcp: FastMCP) -> None:
        """Patch lowlevel request handlers to enforce per-user RBAC filtering.

        FastMCP registers protocol handlers via closures at setup time.
        Monkey-patching methods on the FastMCP instance has no effect because
        the lowlevel server's request_handlers dict still holds the original
        closures. We must wrap the handlers in that dict directly.
        """
        from mcp import types as mcp_types

        server = self
        lowlevel = mcp._mcp_server  # noqa: SLF001

        # Wrap list_tools handler
        original_list_handler = lowlevel.request_handlers.get(mcp_types.ListToolsRequest)
        if original_list_handler is None:
            logger.warning("No ListToolsRequest handler registered, skipping tool filtering")
            return

        async def filtered_list_handler(req: mcp_types.ListToolsRequest) -> mcp_types.ServerResult:
            result = await original_list_handler(req)
            try:
                check = await asyncio.to_thread(server.get_allowed_tools)
            except Exception:
                logger.error("RBAC check failed, denying all tools", exc_info=True)
                return mcp_types.ServerResult(mcp_types.ListToolsResult(tools=[]))
            if check is None:
                return result
            allowed, governed = check
            list_result = result.root
            assert isinstance(list_result, mcp_types.ListToolsResult)
            all_tools = list_result.tools
            # Tools with permission mappings must pass RBAC; unmapped tools are allowed
            filtered = [t for t in all_tools if t.name in allowed or t.name not in governed]
            logger.info("Tool filtering: %d/%d tools allowed", len(filtered), len(all_tools))
            return mcp_types.ServerResult(mcp_types.ListToolsResult(tools=filtered))

        lowlevel.request_handlers[mcp_types.ListToolsRequest] = filtered_list_handler

        # Wrap call_tool handler
        original_call_handler = lowlevel.request_handlers.get(mcp_types.CallToolRequest)
        if original_call_handler is None:
            logger.warning("No CallToolRequest handler registered, skipping call filtering")
            return

        async def filtered_call_handler(req: mcp_types.CallToolRequest) -> mcp_types.ServerResult:
            tool_name = req.params.name
            try:
                check = await asyncio.to_thread(server.get_allowed_tools)
            except Exception:
                logger.error("RBAC check failed for tool '%s'", tool_name, exc_info=True)
                return mcp_types.ServerResult(
                    mcp_types.CallToolResult(
                        content=[
                            mcp_types.TextContent(
                                type="text",
                                text=f"Tool '{tool_name}' is not permitted: RBAC check failed",
                            )
                        ],
                        isError=True,
                    )
                )
            if check is not None:
                allowed, governed = check
                # Only enforce for tools with permission mappings
                if tool_name in governed and tool_name not in allowed:
                    return mcp_types.ServerResult(
                        mcp_types.CallToolResult(
                            content=[
                                mcp_types.TextContent(
                                    type="text",
                                    text=f"Tool '{tool_name}' is not permitted for the current user",
                                )
                            ],
                            isError=True,
                        )
                    )
            return await original_call_handler(req)

        lowlevel.request_handlers[mcp_types.CallToolRequest] = filtered_call_handler
        logger.info("Tool-level RBAC filtering installed (request_handlers patched)")

    def _register_core_resources(self, mcp: FastMCP) -> None:
        """Register core MCP resources for cluster information."""
        from rhoai_mcp.core_resources import register_core_resources

        register_core_resources(mcp, self)

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
