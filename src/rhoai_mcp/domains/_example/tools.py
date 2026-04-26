"""MCP Tools for the example domain."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains._example.client import ExampleClient
from rhoai_mcp.utils.response import PaginatedResponse, paginate

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register example domain tools with the MCP server."""

    @mcp.tool()
    def list_example_items(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List example items (ConfigMaps labelled rhoai.io/example=true).

        This is a demonstration tool from the contributor blueprint.

        Args:
            namespace: The project (namespace) name.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).

        Returns:
            Paginated list of example items.
        """
        client = ExampleClient(server.k8s)
        all_items = client.list_items(namespace)

        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        paginated, total = paginate(all_items, offset, effective_limit)
        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_example_item(
        name: str,
        namespace: str,
    ) -> dict[str, Any]:
        """Get a single example item by name.

        Args:
            name: The ConfigMap name.
            namespace: The project (namespace) name.

        Returns:
            Example item details.
        """
        client = ExampleClient(server.k8s)
        item = client.get_item(name, namespace)

        return {
            "name": item.metadata.name,
            "namespace": item.metadata.namespace,
            "display_name": item.display_name,
            "data": item.data,
            "status": item.status.value,
            "_source": item.metadata.to_source_dict(),
        }
