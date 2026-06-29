"""MCP Resources for {{DOMAIN_DESCRIPTION}}.

Optional — only needed if your domain exposes MCP resources
(data endpoints that agents can read). Delete this file if not needed.
"""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.{{DOMAIN_NAME}}.client import {{DOMAIN_CLASS}}Client

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_resources(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register {{DOMAIN_NAME}} MCP resources."""

    @mcp.resource("rhoai://{{DOMAIN_NAME}}/{namespace}/status")
    def {{DOMAIN_NAME}}_status(namespace: str) -> dict[str, Any]:
        """Get {{DOMAIN_NAME}} status in a namespace."""
        client = {{DOMAIN_CLASS}}Client(server.k8s)
        items = client.list_{{RESOURCE_NAME}}s(namespace)
        return {
            "namespace": namespace,
            "count": len(items),
            "items": items,
        }
