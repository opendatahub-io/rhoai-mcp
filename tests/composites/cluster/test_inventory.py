"""Tests for the cluster resource inventory tool."""

from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from rhoai_mcp.composites.cluster.inventory import register_cluster_resources


@pytest.mark.asyncio
async def test_get_cluster_resources() -> None:
    """The cluster tool returns the established resource inventory shape."""
    server = MagicMock()
    node = MagicMock()
    node.metadata.name = "worker-1"
    node.metadata.labels = {"nvidia.com/gpu.product": "Tesla-T4"}
    node.status.capacity = {"cpu": "32", "memory": "128Gi", "nvidia.com/gpu": "4"}
    node.status.allocatable = {"cpu": "30", "memory": "120Gi", "nvidia.com/gpu": "4"}
    server.k8s.core_v1.list_node.return_value.items = [node]

    mcp = FastMCP("test")
    register_cluster_resources(mcp, server)
    _, result = await mcp.call_tool("get_cluster_resources", {})

    assert result == {
        "cpu_total": 32,
        "cpu_allocatable": 30,
        "memory_total_gb": 128.0,
        "memory_allocatable_gb": 120.0,
        "node_count": 1,
        "has_gpus": True,
        "gpu_info": {
            "type": "nvidia.com/gpu",
            "product": "Tesla-T4",
            "products": ["Tesla-T4"],
            "total": 4,
            "available": 4,
            "nodes_with_gpu": 1,
        },
        "nodes": [
            {
                "name": "worker-1",
                "cpu": 30,
                "memory_gb": 120.0,
                "gpus": 4,
                "gpu_product": "Tesla-T4",
            }
        ],
    }
