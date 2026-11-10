"""MCP Tools for training job discovery."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.training.client import TrainingClient
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    ResponseBuilder,
    Verbosity,
    paginate,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register training discovery tools with the MCP server."""

    @mcp.tool()
    def list_training_jobs(
        namespace: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List training jobs in a namespace with pagination.

        Returns information about TrainJob resources in the specified
        namespace, including their status and progress.

        Args:
            namespace: The namespace to list training jobs from.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Paginated list of training jobs with metadata.
        """
        client = TrainingClient(server.k8s)
        jobs = client.list_training_jobs(namespace)

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(jobs, offset, effective_limit)

        # Format with verbosity
        v = Verbosity.from_str(verbosity)
        items = [ResponseBuilder.training_job_list_item(job, v) for job in paginated]

        result = PaginatedResponse.build(items, total, offset, effective_limit)
        result["namespace"] = namespace
        return result

    @mcp.tool()
    def get_training_job(
        namespace: str,
        name: str,
        verbosity: str = "full",
    ) -> dict[str, Any]:
        """Get detailed information about a specific training job.

        Returns comprehensive information about a TrainJob including its
        configuration, current status, and training progress.

        Args:
            namespace: The namespace of the training job.
            name: The name of the training job.
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Training job information at the requested verbosity level.
        """
        client = TrainingClient(server.k8s)
        job = client.get_training_job(namespace, name)

        v = Verbosity.from_str(verbosity)
        return ResponseBuilder.training_job_detail(job, v)

    from rhoai_mcp.composites.cluster.resources import register_cluster_resources

    register_cluster_resources(mcp, server)

    @mcp.tool()
    def list_training_runtimes(namespace: str | None = None) -> dict[str, Any]:
        """List available training runtimes.

        Training runtimes define the container images, frameworks, and
        configurations used for training jobs. This includes both
        cluster-scoped and namespace-scoped runtimes.

        Args:
            namespace: Optional namespace to include namespace-scoped runtimes.

        Returns:
            List of available training runtimes.
        """
        client = TrainingClient(server.k8s)

        # Always get cluster-scoped runtimes
        runtimes = client.list_cluster_training_runtimes()

        # Optionally include namespace-scoped runtimes
        if namespace:
            ns_runtimes = client.list_training_runtimes(namespace)
            runtimes.extend(ns_runtimes)

        runtime_list = []
        for runtime in runtimes:
            runtime_list.append(
                {
                    "name": runtime.name,
                    "namespace": runtime.namespace,
                    "framework": runtime.framework,
                    "has_model_initializer": runtime.has_model_initializer,
                    "has_dataset_initializer": runtime.has_dataset_initializer,
                    "scope": "cluster" if runtime.namespace is None else "namespace",
                }
            )

        return {
            "count": len(runtime_list),
            "runtimes": runtime_list,
        }
