"""Entry point for RHOAI MCP server."""

import argparse
import logging
import sys
from typing import Any

from rhoai_mcp_core import __version__
from rhoai_mcp_core.config import (
    AuthMode,
    LogLevel,
    RHOAIConfig,
    TransportMode,
)


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

    if args.log_level:
        config_kwargs["log_level"] = LogLevel(args.log_level)

    # Create config
    config = RHOAIConfig(**config_kwargs)

    # Setup logging
    setup_logging(config.log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting RHOAI MCP server v{__version__}")

    # Validate auth config
    try:
        warnings = config.validate_auth_config()
        for warning in warnings:
            logger.warning(warning)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    # Create and run server
    from rhoai_mcp_core.server import create_server

    mcp = create_server(config)

    # Run with appropriate transport
    # Note: Host/port are set via RHOAI_MCP_HOST/PORT env vars which FastMCP reads
    import os

    os.environ.setdefault("UVICORN_HOST", config.host)
    os.environ.setdefault("UVICORN_PORT", str(config.port))

    if config.transport == TransportMode.STDIO:
        logger.info("Running with stdio transport")
        mcp.run(transport="stdio")
    elif config.transport == TransportMode.SSE:
        logger.info(f"Running with SSE transport on {config.host}:{config.port}")
        mcp.run(transport="sse")
    elif config.transport == TransportMode.STREAMABLE_HTTP:
        logger.info(f"Running with streamable-http transport on {config.host}:{config.port}")
        mcp.run(transport="streamable-http")

    return 0


if __name__ == "__main__":
    sys.exit(main())
