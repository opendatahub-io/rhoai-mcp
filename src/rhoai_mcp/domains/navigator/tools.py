"""MCP tools for CUDA compatibility queries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register Navigator CUDA compatibility tools with the MCP server.

    Args:
        mcp: FastMCP server instance
        server: RHOAI server instance
    """
    from rhoai_mcp.domains.navigator.client import CudaCompatibilityClient

    @mcp.tool()
    def get_cuda_version_for_runtime(image: str) -> dict[str, str | list[str]]:
        """Get CUDA toolkit version(s) for a RHOAI serving runtime image.

        Use this tool to find out which CUDA version is required by a specific
        RHOAI serving runtime container image.

        Args:
            image: RHOAI serving runtime image name (e.g., "rhaiis/vllm-cuda-rhel9:3.0")

        Returns:
            Dictionary with image name and list of CUDA versions
        """
        cuda_client = CudaCompatibilityClient(server.k8s)
        try:
            cuda_versions = cuda_client.get_cuda_for_runtime(image)
            return {
                "image": image,
                "cuda_versions": cuda_versions,
                "status": "found",
            }
        except ValueError as e:
            logger.warning(f"Runtime image not found: {image}")
            try:
                available_images = cuda_client.list_all_runtime_images()[:10]
            except Exception:
                available_images = []
            return {
                "image": image,
                "cuda_versions": [],
                "status": "not_found",
                "error": str(e),
                "available_images": available_images,  # First 10 as hint
            }
        except Exception as e:
            logger.error(f"Backend error querying runtime image {image}: {e}")
            return {
                "image": image,
                "cuda_versions": [],
                "status": "backend_error",
                "error": str(e),
            }

    @mcp.tool()
    def get_min_driver_for_cuda_version(cuda_version: str) -> dict[str, str | list[str]]:
        """Get minimum NVIDIA driver version for a CUDA toolkit version.

        Use this tool to find out the minimum GPU driver version required
        to run a specific CUDA toolkit version.

        Args:
            cuda_version: CUDA toolkit version (e.g., "12.4", "13.0")

        Returns:
            Dictionary with CUDA version and minimum driver version(s)
        """
        cuda_client = CudaCompatibilityClient(server.k8s)
        try:
            min_driver = cuda_client.get_min_driver_for_cuda(cuda_version)
            return {
                "cuda_version": cuda_version,
                "min_driver_versions": min_driver,
                "status": "found",
            }
        except ValueError as e:
            logger.warning(f"CUDA version not found: {cuda_version}")
            try:
                available_versions = cuda_client.list_all_cuda_versions()
            except Exception:
                available_versions = []
            return {
                "cuda_version": cuda_version,
                "min_driver_versions": [],
                "status": "not_found",
                "error": str(e),
                "available_cuda_versions": available_versions,
            }
        except Exception as e:
            logger.error(f"Backend error querying CUDA version {cuda_version}: {e}")
            return {
                "cuda_version": cuda_version,
                "min_driver_versions": [],
                "status": "backend_error",
                "error": str(e),
            }

    @mcp.tool()
    def get_supported_cuda_for_gpu(compute_capability: str) -> dict[str, str | list[str]]:
        """Get supported CUDA versions for a GPU compute capability.

        Use this tool to find out which CUDA versions are compatible with
        a specific GPU architecture (identified by its compute capability).

        Args:
            compute_capability: GPU compute capability (e.g., "8.0" for A100, "9.0" for H100)

        Returns:
            Dictionary with compute capability and list of supported CUDA versions
        """
        cuda_client = CudaCompatibilityClient(server.k8s)
        try:
            supported_cuda = cuda_client.get_supported_cuda_for_compute(compute_capability)
            return {
                "compute_capability": compute_capability,
                "supported_cuda_versions": supported_cuda,
                "status": "found",
            }
        except ValueError as e:
            logger.warning(f"Compute capability not found: {compute_capability}")
            try:
                available_capabilities = cuda_client.list_all_compute_capabilities()
            except Exception:
                available_capabilities = []
            return {
                "compute_capability": compute_capability,
                "supported_cuda_versions": [],
                "status": "not_found",
                "error": str(e),
                "available_compute_capabilities": available_capabilities,
            }
        except Exception as e:
            logger.error(f"Backend error querying compute capability {compute_capability}: {e}")
            return {
                "compute_capability": compute_capability,
                "supported_cuda_versions": [],
                "status": "backend_error",
                "error": str(e),
            }

    logger.info("Registered 3 CUDA compatibility tools")
