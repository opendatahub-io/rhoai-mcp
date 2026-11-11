"""Cluster resource inventory tool."""

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.cluster.client import ClusterClient

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_cluster_resources(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register the cluster resource inventory tool."""

    @mcp.tool()
    def get_cluster_resources() -> dict[str, Any]:
        """Get cluster-wide compute resources available for workloads.

        Returns information about CPU, memory, and GPU resources across
        all nodes in the cluster. Useful for planning workloads and
        understanding cluster capacity.

        Returns:
            Cluster resource summary including CPU, memory, and GPU info.
        """
        client = ClusterClient(server.k8s)
        resources = client.get_cluster_resources()

        result: dict[str, Any] = {
            "cpu_total": resources.cpu_total,
            "cpu_allocatable": resources.cpu_allocatable,
            "memory_total_gb": round(resources.memory_total_gb, 1),
            "memory_allocatable_gb": round(resources.memory_allocatable_gb, 1),
            "node_count": resources.node_count,
            "has_gpus": resources.has_gpus,
        }

        if resources.gpu_info:
            result["gpu_info"] = {
                "type": resources.gpu_info.type,
                "product": resources.gpu_info.product,
                "products": resources.gpu_info.products,
                "total": resources.gpu_info.total,
                "available": resources.gpu_info.available,
                "nodes_with_gpu": resources.gpu_info.nodes_with_gpu,
            }

        # Include per-node details
        result["nodes"] = [
            {
                "name": node.name,
                "cpu": node.cpu_allocatable,
                "memory_gb": round(node.memory_allocatable_gb, 1),
                "gpus": node.gpu_count,
                "gpu_product": node.gpu_product,
            }
            for node in resources.nodes
        ]

        return result
