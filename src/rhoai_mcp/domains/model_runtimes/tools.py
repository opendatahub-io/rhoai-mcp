"""MCP tools for CUDA compatibility queries."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)


def register_tools(mcp: "FastMCP", server: "RHOAIServer") -> None:
    """Register Model Runtimes CUDA compatibility tools with the MCP server.

    Args:
        mcp: FastMCP server instance
        server: RHOAI server instance
    """
    from rhoai_mcp.domains.model_runtimes.client import CudaCompatibilityClient

    def get_client() -> CudaCompatibilityClient:
        """Create a new client instance."""
        return CudaCompatibilityClient(server.k8s)

    @mcp.tool()
    async def get_cuda_version_for_runtime(image: str) -> dict[str, str | list[str]]:
        """Get CUDA toolkit version(s) for a RHOAI serving runtime image.

        Use this tool to find out which CUDA version is required by a specific
        RHOAI serving runtime container image.

        Args:
            image: RHOAI serving runtime image name (e.g., "rhaiis/vllm-cuda-rhel9:3.0")

        Returns:
            Dictionary with image name and list of CUDA versions

        Raises:
            ValueError: If image not found in compatibility matrix
        """
        cuda_versions = await get_client().get_cuda_for_runtime(image)
        return {
            "image": image,
            "cuda_versions": cuda_versions,
        }

    @mcp.tool()
    async def get_min_driver_for_cuda_version(cuda_version: str) -> dict[str, str | list[str]]:
        """Get minimum NVIDIA driver version for a CUDA toolkit version.

        Use this tool to find out the minimum GPU driver version required
        to run a specific CUDA toolkit version.

        Args:
            cuda_version: CUDA toolkit version (e.g., "12.4", "13.0")

        Returns:
            Dictionary with CUDA version and minimum driver version(s)

        Raises:
            ValueError: If CUDA version not found in compatibility matrix
        """
        min_driver = await get_client().get_min_driver_for_cuda(cuda_version)
        return {
            "cuda_version": cuda_version,
            "min_driver_versions": min_driver,
        }

    @mcp.tool()
    async def get_supported_cuda_for_gpu(compute_capability: str) -> dict[str, str | list[str]]:
        """Get supported CUDA versions for a GPU compute capability.

        Use this tool to find out which CUDA versions are compatible with
        a specific GPU architecture (identified by its compute capability).

        Args:
            compute_capability: GPU compute capability (e.g., "8.0" for A100, "9.0" for H100)

        Returns:
            Dictionary with compute capability and list of supported CUDA versions

        Raises:
            ValueError: If compute capability not found in compatibility matrix
        """
        supported_cuda = await get_client().get_supported_cuda_for_compute(compute_capability)
        return {
            "compute_capability": compute_capability,
            "supported_cuda_versions": supported_cuda,
        }
