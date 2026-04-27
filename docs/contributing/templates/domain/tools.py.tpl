"""MCP Tools for {{DOMAIN_DESCRIPTION}} operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.{{DOMAIN_NAME}}.client import {{DOMAIN_CLASS}}Client
from rhoai_mcp.utils.response import PaginatedResponse, paginate

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register {{DOMAIN_NAME}} tools with the MCP server."""

    @mcp.tool()
    def list_{{RESOURCE_NAME}}s(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List {{RESOURCE_NAME}} resources in a namespace.

        Args:
            namespace: The project (namespace) name.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            Paginated list of {{RESOURCE_NAME}} resources.
        """
        client = {{DOMAIN_CLASS}}Client(server.k8s)
        all_items = client.list_{{RESOURCE_NAME}}s(namespace)

        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        paginated, total = paginate(all_items, offset, effective_limit)

        # TODO: Apply verbosity filtering if needed.
        # from rhoai_mcp.utils.response import Verbosity
        # v = Verbosity.from_str(verbosity)
        # items = [ResponseBuilder.your_list_item(item, v) for item in paginated]

        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_{{RESOURCE_NAME}}(
        name: str,
        namespace: str,
    ) -> dict[str, Any]:
        """Get a single {{RESOURCE_NAME}} by name.

        Args:
            name: Resource name.
            namespace: The project (namespace) name.

        Returns:
            {{RESOURCE_CLASS}} details.
        """
        client = {{DOMAIN_CLASS}}Client(server.k8s)
        resource = client.get_{{RESOURCE_NAME}}(name, namespace)

        return {
            "name": resource.metadata.name,
            "namespace": resource.metadata.namespace,
            "display_name": resource.display_name,
            "status": resource.status.value,
            "_source": resource.metadata.to_source_dict(),
        }

    # --- Optional: write operations ---
    # Uncomment and implement as needed. Always check is_operation_allowed().
    #
    # @mcp.tool()
    # def create_{{RESOURCE_NAME}}(name: str, namespace: str, ...) -> dict[str, Any]:
    #     """Create a {{RESOURCE_NAME}}."""
    #     allowed, reason = server.config.is_operation_allowed("create")
    #     if not allowed:
    #         return {"error": reason}
    #     ...
    #
    # @mcp.tool()
    # def delete_{{RESOURCE_NAME}}(name: str, namespace: str, confirm: bool = False) -> dict[str, Any]:
    #     """Delete a {{RESOURCE_NAME}}."""
    #     allowed, reason = server.config.is_operation_allowed("delete")
    #     if not allowed:
    #         return {"error": reason}
    #     if not confirm:
    #         return {"error": "Deletion not confirmed", "message": "Set confirm=True to delete."}
    #     ...
