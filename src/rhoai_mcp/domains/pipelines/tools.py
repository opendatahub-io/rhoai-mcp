"""MCP Tools for Pipeline operations."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.pipelines.client import PipelineClient
from rhoai_mcp.domains.pipelines.models import PipelineServerCreate
from rhoai_mcp.utils.errors import NotManagedByMCPError

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register pipeline tools with the MCP server."""

    @mcp.tool()
    def get_pipeline_server(namespace: str) -> dict[str, Any]:
        """Get the pipeline server status for a Data Science Project.

        Each project can have one Data Science Pipelines Application (DSPA)
        that provides pipeline execution capabilities.

        Args:
            namespace: The project (namespace) name.

        Returns:
            Pipeline server status, or indication that none exists.
        """
        client = PipelineClient(server.k8s)
        result = client.get_pipeline_server(namespace)

        if result is None:
            return {
                "exists": False,
                "message": (
                    f"No pipeline server configured in project '{namespace}'. "
                    "Use create_pipeline_server to set one up."
                ),
            }

        return {
            "exists": True,
            **result,
        }

    @mcp.tool()
    def create_pipeline_server(
        namespace: str,
        object_storage_secret: str,
        object_storage_bucket: str,
        object_storage_endpoint: str,
        object_storage_region: str = "us-east-1",
    ) -> dict[str, Any]:
        """Create a pipeline server for a Data Science Project.

        Creates a Data Science Pipelines Application (DSPA) that enables
        running ML pipelines in the project. Requires an S3-compatible
        object storage for storing pipeline artifacts.

        Args:
            namespace: Project (namespace) name.
            object_storage_secret: Name of secret containing S3 credentials
                (must have AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY).
            object_storage_bucket: S3 bucket name for pipeline artifacts.
            object_storage_endpoint: S3 endpoint URL.
            object_storage_region: S3 region (default: us-east-1).

        Returns:
            Created pipeline server information.
        """
        # Check if operation is allowed
        allowed, reason = server.config.is_operation_allowed("create")
        if not allowed:
            return {"error": reason}

        client = PipelineClient(server.k8s)
        request = PipelineServerCreate(
            namespace=namespace,
            object_storage_secret=object_storage_secret,
            object_storage_bucket=object_storage_bucket,
            object_storage_endpoint=object_storage_endpoint,
            object_storage_region=object_storage_region,
        )
        dspa = client.create_pipeline_server(request)

        return {
            "name": dspa.metadata.name,
            "namespace": dspa.metadata.namespace,
            "status": dspa.status.value,
            "message": ("Pipeline server created. It may take several minutes to become ready."),
        }

    @mcp.tool()
    def delete_pipeline_server(
        namespace: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete the pipeline server from a Data Science Project.

        WARNING: This will delete the DSPA and all associated pipeline
        infrastructure. Running pipelines will be terminated.

        Args:
            namespace: The project (namespace) name.
            confirm: Must be True to actually delete.

        Returns:
            Confirmation of deletion.
        """
        if server.config.read_only_mode:
            return {"error": "Read-only mode is enabled"}

        if not confirm:
            return {
                "error": "Deletion not confirmed",
                "message": (
                    "To delete the pipeline server, set confirm=True. "
                    "WARNING: All pipeline infrastructure will be removed."
                ),
            }

        client = PipelineClient(server.k8s)
        # Get the DSPA name first
        existing = client.get_pipeline_server(namespace)
        if not existing:
            return {
                "error": f"No pipeline server exists in namespace '{namespace}'",
            }

        try:
            client.delete_pipeline_server(existing["name"], namespace)
        except NotManagedByMCPError as e:
            return {"error": str(e)}

        return {
            "namespace": namespace,
            "deleted": True,
            "message": "Pipeline server deleted",
        }
