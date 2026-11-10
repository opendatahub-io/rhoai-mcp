"""Cluster-wide Kubernetes resource inventory.

these were originally in src/rhoai_mcp/domains/training/client.py
moved from Training domain, to Cluster domain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rhoai_mcp.domains.cluster.models import ClusterResources, GPUInfo, NodeResources

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient


# GPU product label keys by GPU resource type
GPU_PRODUCT_LABELS: dict[str, list[str]] = {
    "nvidia.com/gpu": ["nvidia.com/gpu.product", "nvidia.com/gpu-product"],
    "amd.com/gpu": ["amd.com/gpu.product", "amd.com/gpu-product"],
}


def _get_gpu_product(labels: dict[str, str] | None, gpu_resource_key: str) -> str | None:
    """Extract GPU product name from node labels.

    Args:
        labels: Node labels dictionary.
        gpu_resource_key: GPU resource key (e.g., "nvidia.com/gpu").

    Returns:
        GPU product name (e.g., "Tesla-T4") or None if not found.
    """
    if not labels:
        return None
    for key in GPU_PRODUCT_LABELS.get(gpu_resource_key, []):
        if key in labels:
            return labels[key]
    return None


class ClusterClient:
    """Client for cluster-wide resource inventory."""

    def __init__(self, k8s: K8sClient) -> None:
        self._k8s = k8s

    def get_cluster_resources(self) -> ClusterResources:
        """Get cluster-wide resource information.

        Returns:
            ClusterResources with CPU, memory, and GPU info.
        """
        nodes = self._k8s.core_v1.list_node()

        total_cpu = 0
        allocatable_cpu = 0
        total_memory_gb = 0.0
        allocatable_memory_gb = 0.0
        gpu_total = 0
        gpu_nodes = 0
        gpu_type = None
        gpu_products: list[str] = []
        node_resources = []

        for node in nodes.items:
            capacity = node.status.capacity or {}
            allocatable = node.status.allocatable or {}
            node_labels = node.metadata.labels or {}

            cpu = _parse_cpu(capacity.get("cpu", "0"))
            cpu_alloc = _parse_cpu(allocatable.get("cpu", "0"))
            memory = _parse_memory_gb(capacity.get("memory", "0"))
            memory_alloc = _parse_memory_gb(allocatable.get("memory", "0"))

            total_cpu += cpu
            allocatable_cpu += cpu_alloc
            total_memory_gb += memory
            allocatable_memory_gb += memory_alloc

            # Check for GPUs
            node_gpus = 0
            node_gpu_type = None
            node_gpu_product = None
            for key in ["nvidia.com/gpu", "amd.com/gpu"]:
                if key in capacity:
                    node_gpus = int(capacity[key])
                    if node_gpus > 0:
                        node_gpu_type = key
                        gpu_type = key
                        gpu_total += node_gpus
                        gpu_nodes += 1
                        # Extract GPU product from node labels
                        node_gpu_product = _get_gpu_product(node_labels, key)
                        if node_gpu_product and node_gpu_product not in gpu_products:
                            gpu_products.append(node_gpu_product)
                        break

            node_resources.append(
                NodeResources(
                    name=node.metadata.name,
                    cpu_total=cpu,
                    cpu_allocatable=cpu_alloc,
                    memory_total_gb=memory,
                    memory_allocatable_gb=memory_alloc,
                    gpu_count=node_gpus,
                    gpu_type=node_gpu_type,
                    gpu_product=node_gpu_product,
                )
            )

        gpu_info = None
        if gpu_total > 0 and gpu_type:
            gpu_info = GPUInfo(
                type=gpu_type,
                product=gpu_products[0] if gpu_products else None,
                products=gpu_products,
                total=gpu_total,
                available=gpu_total,  # Simplified; real impl would check allocations
                nodes_with_gpu=gpu_nodes,
            )

        return ClusterResources(
            cpu_total=total_cpu,
            cpu_allocatable=allocatable_cpu,
            memory_total_gb=total_memory_gb,
            memory_allocatable_gb=allocatable_memory_gb,
            gpu_info=gpu_info,
            node_count=len(nodes.items),
            nodes=node_resources,
        )


def _parse_cpu(value: str) -> int:
    """Parse CPU value to cores."""
    if not value:
        return 0
    value = str(value)
    if value.endswith("m"):
        return int(value[:-1]) // 1000
    return int(value)


def _parse_memory_gb(value: str) -> float:
    """Parse memory value to GB."""
    if not value:
        return 0.0
    value = str(value)
    multipliers = {
        "Ki": 1 / (1024 * 1024),
        "Mi": 1 / 1024,
        "Gi": 1,
        "Ti": 1024,
        "K": 1 / (1000 * 1000),
        "M": 1 / 1000,
        "G": 1,
        "T": 1000,
    }
    for suffix, mult in multipliers.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * mult
    return float(value) / (1024 * 1024 * 1024)  # Assume bytes
