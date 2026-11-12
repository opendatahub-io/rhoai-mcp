"""MCP server definition for RHOAI with pluggy-based plugin system."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from mcp.server.mcpserver import Context, MCPServer
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from rhoai_mcp.auth.rbac_middleware import RBACMiddleware
from rhoai_mcp.clients.base import K8sClient
from rhoai_mcp.config import RHOAIConfig, TransportMode, get_config
from rhoai_mcp.plugin_manager import PluginManager
from rhoai_mcp.utils.errors import RHOAIError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RHOAIServer:
    """RHOAI MCP Server with pluggy-based plugin system."""

    def __init__(self, config: RHOAIConfig | None = None) -> None:
        self._config = config or get_config()
        self._k8s_client: K8sClient | None = None
        self._mcp: MCPServer | None = None
        self._plugin_manager: PluginManager | None = None

    @property
    def config(self) -> RHOAIConfig:
        """Get server configuration."""
        return self._config

    def get_k8s(self, ctx: Context | None = None) -> K8sClient:
        """Per-request Kubernetes client for the calling user (RHAI-14631).

        Token precedence per the 1.6 design: the per-message request
        headers when the transport carries them (SSE attaches each POST's
        Request to the message metadata — ``ctx.request_context.meta``),
        else ``UserContext.token`` (per-POST fresh under Streamable HTTP;
        connection-time under SSE, bounded by the SDK's session-owner
        pinning).

        Args:
            ctx: the tool call's Context. None falls back to UserContext
                only (used by the deprecated ``k8s`` property).

        Returns:
            The shared SA client when OIDC is disabled; otherwise a client
            authenticated as the current user (user-token or impersonation
            strategy).

        Raises:
            RHOAIError: If server is not running, no UserContext is set
                under OIDC, or no token is available for the user-token
                strategy.
        """
        if self._k8s_client is None:
            raise RHOAIError("Server not running. K8s client not available.")

        if self._config.oidc_enabled:
            from rhoai_mcp.auth.user_context import UserContext
            from rhoai_mcp.config import OIDCKubeAuthStrategy

            user_context = UserContext.current()
            if user_context is None:
                raise RHOAIError(
                    "OIDC is enabled but no UserContext is set. "
                    "Refusing to fall back to service-account client."
                )

            if self._config.oidc_kube_auth_strategy == OIDCKubeAuthStrategy.USER_TOKEN:
                token = self._per_message_token(ctx) or user_context.token
                if not token:
                    raise RHOAIError(
                        "OIDC user-token strategy is active but no token is available "
                        "(neither the per-message request nor UserContext.token)."
                    )
                return self._k8s_client.create_user_token_client(token)
            else:
                return self._k8s_client.create_impersonating_client(
                    user_context.username, user_context.groups
                )

        return self._k8s_client

    @staticmethod
    def _per_message_token(ctx: Context | None) -> SecretStr | None:
        """Bearer token from the per-message request, when the transport carries it.

        SSE attaches each POST's Starlette Request to the message metadata
        (``ctx.request_context.meta.request_context``); Streamable HTTP and
        stdio leave ``meta`` None — the per-POST ``UserContext`` covers
        streamable, and stdio has no per-message requests. Uses only public
        attribute access; no SDK private imports (replaces the v1
        ``request_ctx`` ContextVar mechanism, which no longer exists in v2).
        """
        if ctx is None or ctx.request_context is None:
            return None
        meta = getattr(ctx.request_context, "meta", None)
        request = getattr(meta, "request_context", None) if meta is not None else None
        headers = getattr(request, "headers", None)
        if not headers:
            return None
        auth_header = headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return None
        return SecretStr(auth_header[7:])

    def get_shared_k8s(self) -> K8sClient:
        """The shared service-account Kubernetes client.

        For server-level operations that are not on behalf of a calling
        user: startup health checks, CRD discovery. Per-request tool code
        must use :meth:`get_k8s` instead.

        Raises:
            RHOAIError: If server is not running.
        """
        if self._k8s_client is None:
            raise RHOAIError("Server not running. K8s client not available.")
        return self._k8s_client

    @property
    def mcp(self) -> MCPServer:
        """Get the MCP server instance.

        Raises:
            RHOAIError: If server is not initialized.
        """
        if self._mcp is None:
            raise RHOAIError("Server not initialized.")
        return self._mcp

    @property
    def plugin_manager(self) -> PluginManager:
        """Get the plugin manager.

        Raises:
            RHOAIError: If server is not initialized.
        """
        if self._plugin_manager is None:
            raise RHOAIError("Server not initialized.")
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
        MCPServer's lifespan runs per MCP client session (inside Server.run()),
        not at ASGI app startup. Putting K8s connect here would block the
        /health endpoint until the first MCP client connects.
        """

        @asynccontextmanager
        async def lifespan(_app: Any) -> AsyncGenerator[None]:
            """Per-session lifespan - K8s is already connected."""
            logger.debug("MCP session started")
            try:
                yield
            finally:
                logger.debug("MCP session ended")
                # HTTP sessions share the port-forward manager; ending one must
                # not close a forward still used by another session.
                if self._config.transport == TransportMode.STDIO:
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
        """Disconnect from Kubernetes."""
        logger.info("Shutting down RHOAI MCP server...")
        if self._k8s_client:
            self._k8s_client.disconnect()
            self._k8s_client = None
        logger.info("RHOAI MCP server shut down")

    @staticmethod
    def _with_port_forward_shutdown(app: Starlette) -> Starlette:
        """Close shared port forwards when the HTTP app lifespan ends."""
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncGenerator[Any]:
            try:
                async with original_lifespan(app) as state:
                    yield state
            finally:
                try:
                    from rhoai_mcp.utils.port_forward import PortForwardManager

                    await PortForwardManager.get_instance().close_all()
                except Exception as e:
                    logger.warning(f"Error closing port-forward connections: {e}")

        app.router.lifespan_context = lifespan
        return app

    def _install_http_shutdown(self, mcp: MCPServer) -> None:
        """Attach process cleanup to HTTP app shutdown, not client sessions."""
        if self._config.transport == TransportMode.SSE:
            original_sse_app = mcp.sse_app

            def sse_app(*args: Any, **kwargs: Any) -> Starlette:
                return self._with_port_forward_shutdown(original_sse_app(*args, **kwargs))

            mcp.sse_app = sse_app  # type: ignore[method-assign]
        elif self._config.transport == TransportMode.STREAMABLE_HTTP:
            original_http_app = mcp.streamable_http_app

            def streamable_http_app(*args: Any, **kwargs: Any) -> Starlette:
                return self._with_port_forward_shutdown(original_http_app(*args, **kwargs))

            mcp.streamable_http_app = streamable_http_app  # type: ignore[method-assign]

    def create_mcp(self) -> MCPServer:
        """Create and configure the MCPServer."""
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
        # v2: host/port are per-run arguments (passed to run()/app builders),
        # not MCPServer constructor args
        mcp = MCPServer(
            name="rhoai-mcp",
            instructions="MCP server for Red Hat OpenShift AI - enables AI agents to "
            "interact with RHOAI environments including workbenches, "
            "model serving, pipelines, and data connections.",
            lifespan=self._create_lifespan(),
            # Dual-point RBAC enforcement via the public middleware parameter
            # (RHAI-14627 / the ratified 1.6 design) — replaces the v1
            # request_handlers monkey-patch. No-ops when OIDC is disabled.
            middleware=[RBACMiddleware(self)],
        )

        # Store reference
        self._mcp = mcp

        self._install_http_shutdown(mcp)

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

    def _setup_auth(self, mcp: MCPServer) -> None:
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

        def patched_sse_app(*args: Any, **kwargs: Any) -> Any:
            app = original_sse_app(*args, **kwargs)
            return OIDCAuthMiddleware(
                app,
                validator=validator,
                exclude_paths=exclude_paths,
                resource_metadata_url=self._oidc_resource_metadata_url,
            )

        mcp.sse_app = patched_sse_app  # type: ignore[method-assign]

        if hasattr(mcp, "streamable_http_app"):
            original_http_app = mcp.streamable_http_app

            def patched_http_app(*args: Any, **kwargs: Any) -> Any:
                app = original_http_app(*args, **kwargs)
                return OIDCAuthMiddleware(
                    app,
                    validator=validator,
                    exclude_paths=exclude_paths,
                    resource_metadata_url=self._oidc_resource_metadata_url,
                )

            mcp.streamable_http_app = patched_http_app  # type: ignore[method-assign]

        logger.info("Auth middleware will be attached when transport app is created")

        # Tool-level RBAC enforcement is installed at construction time via
        # the RBACMiddleware (see create_mcp) — nothing to patch here.

        logger.info("Authentication enabled")

    def _register_core_resources(self, mcp: MCPServer) -> None:
        """Register core MCP resources for cluster information."""
        from rhoai_mcp.core_resources import register_core_resources

        register_core_resources(mcp, self)

    def _register_health_endpoint(self, mcp: MCPServer) -> None:
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


def create_server(config: RHOAIConfig | None = None) -> MCPServer:
    """Create and return the MCP server instance.

    This is the main entry point for creating the server.
    """
    global _server
    _server = RHOAIServer(config)
    return _server.create_mcp()
