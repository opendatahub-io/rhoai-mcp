"""Models for cluster resource inventory.

originally these where in src/rhoai_mcp/domains/training/models.py
moved from Trainer domain, to Cluster domain
"""

from pydantic import BaseModel, Field


class NodeResources(BaseModel):
    """Node-level resource information."""

    name: str = Field(..., description="Node name")
    cpu_total: int = Field(0, description="Total CPU cores")
    cpu_allocatable: int = Field(0, description="Allocatable CPU cores")
    memory_total_gb: float = Field(0.0, description="Total memory in GB")
    memory_allocatable_gb: float = Field(0.0, description="Allocatable memory in GB")
    gpu_count: int = Field(0, description="Number of GPUs")
    gpu_type: str | None = Field(None, description="GPU resource key (e.g., nvidia.com/gpu)")
    gpu_product: str | None = Field(None, description="GPU product name (e.g., Tesla-T4)")


class GPUInfo(BaseModel):
    """GPU availability information."""

    type: str = Field(..., description="GPU resource key (e.g., nvidia.com/gpu)")
    product: str | None = Field(None, description="GPU product name (e.g., Tesla-T4)")
    products: list[str] = Field(default_factory=list, description="All GPU products in cluster")
    total: int = Field(0, description="Total GPUs in cluster")
    available: int = Field(0, description="Available GPUs")
    nodes_with_gpu: int = Field(0, description="Number of nodes with GPUs")


class ClusterResources(BaseModel):
    """Cluster-wide resource information."""

    cpu_total: int = Field(0, description="Total CPU cores")
    cpu_allocatable: int = Field(0, description="Allocatable CPU cores")
    memory_total_gb: float = Field(0.0, description="Total memory in GB")
    memory_allocatable_gb: float = Field(0.0, description="Allocatable memory in GB")
    gpu_info: GPUInfo | None = Field(None, description="GPU information")
    node_count: int = Field(0, description="Number of nodes")
    nodes: list[NodeResources] = Field(default_factory=list, description="Per-node resources")

    @property
    def has_gpus(self) -> bool:
        """Check if cluster has GPUs available."""
        return self.gpu_info is not None and self.gpu_info.total > 0


