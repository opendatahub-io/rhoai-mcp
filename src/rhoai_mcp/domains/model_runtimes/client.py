"""ConfigMap loader for CUDA compatibility matrix."""

import json
import logging
from typing import TYPE_CHECKING

from kubernetes.client.exceptions import ApiException  # type: ignore[import-untyped]
from packaging.version import parse

from rhoai_mcp.domains.model_runtimes.models import CudaCompatibilityMatrix

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

logger = logging.getLogger(__name__)


class CudaCompatibilityClient:
    """Client for loading CUDA compatibility matrix from ConfigMap."""

    CONFIGMAP_NAME = "cuda-compatibility-matrix"
    CONFIGMAP_DATA_KEY = "cuda_compat.json"

    # Platform namespaces to search (RHOAI first, then ODH)
    PLATFORM_NAMESPACES = ["redhat-ods-applications", "opendatahub"]

    def __init__(self, k8s_client: "K8sClient", namespace: str | None = None) -> None:
        """Initialize the CUDA compatibility client.

        Args:
            k8s_client: Kubernetes client for ConfigMap access
            namespace: Namespace where the ConfigMap is located (auto-detected if not specified)
        """
        self.k8s = k8s_client
        self._explicit_namespace = namespace

    async def load_matrix(self) -> CudaCompatibilityMatrix:
        """Load CUDA compatibility matrix from ConfigMap.

        Tries to find the ConfigMap in platform namespaces (RHOAI, then ODH)
        unless a specific namespace was provided.

        Returns:
            CudaCompatibilityMatrix: The loaded compatibility matrix

        Raises:
            ValueError: If the ConfigMap is not found or data is invalid
        """
        # If explicit namespace provided, use it
        if self._explicit_namespace:
            namespaces = [self._explicit_namespace]
        else:
            # Auto-detect: try RHOAI first, then ODH
            namespaces = self.PLATFORM_NAMESPACES

        last_error = None
        for namespace in namespaces:
            try:
                configmap = self.k8s.core_v1.read_namespaced_config_map(
                    name=self.CONFIGMAP_NAME, namespace=namespace
                )

                if not configmap.data or self.CONFIGMAP_DATA_KEY not in configmap.data:
                    raise ValueError(
                        f"ConfigMap {self.CONFIGMAP_NAME} exists but missing key "
                        f"{self.CONFIGMAP_DATA_KEY}"
                    )

                json_data = configmap.data[self.CONFIGMAP_DATA_KEY]
                data = json.loads(json_data)
                return CudaCompatibilityMatrix.model_validate(data)

            except ApiException as e:
                if e.status == 404:
                    last_error = e
                    continue  # Try next namespace
                raise  # Other errors should propagate immediately

        # If we get here, ConfigMap not found in any namespace
        tried_namespaces = ", ".join(namespaces)
        raise ValueError(
            f"ConfigMap {self.CONFIGMAP_NAME} not found in namespaces: {tried_namespaces}"
        ) from last_error

    def get_namespaces_to_try(self) -> list[str]:
        """Get list of namespaces that will be searched for the ConfigMap.

        Returns:
            List of namespace names to search
        """
        if self._explicit_namespace:
            return [self._explicit_namespace]
        return list(self.PLATFORM_NAMESPACES)

    async def get_cuda_for_runtime(self, image: str) -> list[str]:
        """Get CUDA versions for a given RHOAI runtime image.

        Args:
            image: Container image reference (can be full registry ref or short name)

        Returns:
            List of CUDA versions supported by this runtime image

        Raises:
            ValueError: If the image is not found in the matrix
        """
        matrix = await self.load_matrix()

        # Try exact match or full registry prefix match
        # e.g., "registry.redhat.io/rhaiis/vllm:3.0" matches "rhaiis/vllm:3.0"
        # Require "/" separator to avoid false matches like ":3.0"
        for mapping in matrix.runtime_images:
            if mapping.image == image or image.endswith("/" + mapping.image):
                return mapping.cuda_version

        raise ValueError(f"Runtime image not found in compatibility matrix: {image}")

    async def get_min_driver_for_cuda(self, cuda_version: str) -> list[str]:
        """Get minimum driver version for a given CUDA toolkit version.

        Args:
            cuda_version: CUDA toolkit version (e.g., "12.4")

        Returns:
            List of minimum driver versions (typically one element)

        Raises:
            ValueError: If the CUDA version is not found in the matrix
        """
        matrix = await self.load_matrix()

        for mapping in matrix.cuda_drivers:
            if cuda_version in mapping.cuda_version:
                return mapping.min_driver_version

        raise ValueError(f"CUDA version not found in compatibility matrix: {cuda_version}")

    async def get_supported_cuda_for_compute(self, compute_capability: str) -> list[str]:
        """Get supported CUDA versions for a given GPU compute capability.

        Args:
            compute_capability: GPU compute capability (e.g., "8.0")

        Returns:
            List of supported CUDA versions

        Raises:
            ValueError: If the compute capability is not found in the matrix
        """
        matrix = await self.load_matrix()

        for mapping in matrix.gpu_compute:
            if mapping.compute_capability == compute_capability:
                return mapping.supported_cuda_versions

        raise ValueError(
            f"GPU compute capability not found in compatibility matrix: {compute_capability}"
        )

    async def list_all_runtime_images(self) -> list[str]:
        """Get list of all RHOAI runtime images in the matrix.

        Returns:
            List of runtime image names
        """
        matrix = await self.load_matrix()
        return [mapping.image for mapping in matrix.runtime_images]

    async def list_all_cuda_versions(self) -> list[str]:
        """Get list of all CUDA versions in the matrix.

        Returns:
            List of CUDA versions (sorted semantically)
        """
        matrix = await self.load_matrix()
        versions = set()
        for mapping in matrix.cuda_drivers:
            versions.update(mapping.cuda_version)

        # Sort by semantic version
        return sorted(versions, key=parse)

    async def list_all_compute_capabilities(self) -> list[str]:
        """Get list of all GPU compute capabilities in the matrix.

        Returns:
            List of compute capabilities
        """
        matrix = await self.load_matrix()
        return [mapping.compute_capability for mapping in matrix.gpu_compute]
