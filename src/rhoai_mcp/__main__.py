"""Entry point for RHOAI MCP server."""

import argparse
import logging
import sys
from typing import Any

from rhoai_mcp import __version__
from rhoai_mcp.config import (
    AuthMode,
    LogLevel,
    RHOAIConfig,
    TransportMode,
)
from rhoai_mcp.utils.errors import AuthenticationError


def _has_auth_error(exc: BaseException) -> bool:
    """Check if an exception is or contains an AuthenticationError.

    Handles both direct AuthenticationError and ExceptionGroup wrappers
    (from anyio task groups) on Python 3.10+.
    """
    if isinstance(exc, AuthenticationError):
        return True
    # ExceptionGroup (Python 3.11+) or exceptiongroup backport
    inner: tuple[BaseException, ...] = getattr(exc, "exceptions", ())
    if inner:
        return any(_has_auth_error(e) for e in inner)
    return False


def setup_logging(level: LogLevel) -> None:
    """Configure logging for the server."""
    logging.basicConfig(
        level=level.value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="rhoai-mcp",
        description="MCP server for Red Hat OpenShift AI",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Transport options
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=None,
        help="Transport mode (default: from config or stdio)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind HTTP server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind HTTP server to (default: 8000)",
    )

    # Auth options
    parser.add_argument(
        "--auth-mode",
        choices=["auto", "kubeconfig", "token"],
        default=None,
        help="Authentication mode (default: auto)",
    )
    parser.add_argument(
        "--kubeconfig",
        default=None,
        help="Path to kubeconfig file",
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Kubeconfig context to use",
    )

    # Safety options
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Run in read-only mode (disable all write operations)",
    )
    parser.add_argument(
        "--enable-dangerous",
        action="store_true",
        help="Enable dangerous operations like delete",
    )

    # Mock cluster mode
    parser.add_argument(
        "--mock-cluster",
        action="store_true",
        help="Use mock K8s client with pre-populated cluster state (for testing)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Build config from args, falling back to environment/defaults
    config_kwargs: dict[str, Any] = {}

    if args.transport:
        transport_map = {
            "stdio": TransportMode.STDIO,
            "sse": TransportMode.SSE,
            "streamable-http": TransportMode.STREAMABLE_HTTP,
        }
        config_kwargs["transport"] = transport_map[args.transport]

    if args.host:
        config_kwargs["host"] = args.host

    if args.port:
        config_kwargs["port"] = args.port

    if args.auth_mode:
        auth_map = {
            "auto": AuthMode.AUTO,
            "kubeconfig": AuthMode.KUBECONFIG,
            "token": AuthMode.TOKEN,
        }
        config_kwargs["auth_mode"] = auth_map[args.auth_mode]

    if args.kubeconfig:
        config_kwargs["kubeconfig_path"] = args.kubeconfig

    if args.context:
        config_kwargs["kubeconfig_context"] = args.context

    if args.read_only:
        config_kwargs["read_only_mode"] = True

    if args.enable_dangerous:
        config_kwargs["enable_dangerous_operations"] = True

    if args.mock_cluster:
        config_kwargs["mock_cluster"] = True

    if args.log_level:
        config_kwargs["log_level"] = LogLevel(args.log_level)

    # Create config
    config = RHOAIConfig(**config_kwargs)

    # Setup logging
    setup_logging(config.log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting RHOAI MCP server v{__version__}")

    # Validate auth config (skip for mock cluster mode)
    if not config.mock_cluster:
        try:
            warnings = config.validate_auth_config()
            for warning in warnings:
                logger.warning(warning)
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return 1

    # Create and run server
    # Run with appropriate transport
    # Note: Host/port are set via RHOAI_MCP_HOST/PORT env vars which FastMCP reads
    import os

    from rhoai_mcp.server import create_server, get_server

    os.environ.setdefault("UVICORN_HOST", config.host)
    os.environ.setdefault("UVICORN_PORT", str(config.port))

    transport_name: str = config.transport.value
    if config.transport != TransportMode.STDIO:
        logger.info(f"Running with {transport_name} transport on {config.host}:{config.port}")
    else:
        logger.info(f"Running with {transport_name} transport")

    try:
        mcp = create_server(config)
        mcp.run(transport=transport_name)  # type: ignore[arg-type]
    except BaseException as exc:  # BaseException to catch anyio's BaseExceptionGroup
        if _has_auth_error(exc):
            logger.error(
                "Kubernetes authentication failed. Your credentials may be expired. "
                "Try re-authenticating with: oc login / kubectl config set-credentials"
            )
            return 1
        raise
    finally:
        server = get_server()
        if server:
            server.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
