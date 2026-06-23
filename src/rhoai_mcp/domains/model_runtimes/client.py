"""ConfigMap loader for CUDA compatibility matrix."""

import json
import logging
from typing import TYPE_CHECKING

from kubernetes.client.exceptions import ApiException  # type: ignore[import-untyped]
from packaging.version import Version, parse

from rhoai_mcp.domains.model_runtimes.models import CudaCompatibilityMatrix

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import K8sClient

logger = logging.getLogger(__name__)


class CudaCompatibilityClient:
    """Client for loading CUDA compatibility matrix from ConfigMap."""

    CONFIGMAP_NAME = "cuda-compatibility-matrix"
    CONFIGMAP_DATA_KEY = "cuda_compat.json"

    def __init__(self, k8s_client: "K8sClient", namespace: str | None = None) -> None:
        """Initialize the CUDA compatibility client.

        Args:
            k8s_client: Kubernetes client for ConfigMap access
            namespace: Namespace where the ConfigMap is located (defaults to redhat-ods-applications)
        """
        self.k8s = k8s_client
        self.namespace = namespace or "redhat-ods-applications"
        self._matrix: CudaCompatibilityMatrix | None = None

    async def load_matrix(self) -> CudaCompatibilityMatrix:
        """Load CUDA compatibility matrix from ConfigMap.

        Returns:
            CudaCompatibilityMatrix: The loaded compatibility matrix

        Raises:
            ValueError: If the ConfigMap is not found or data is invalid
        """
        if self._matrix is not None:
            return self._matrix

        try:
            logger.debug(
                f"Loading CUDA compatibility matrix from ConfigMap "
                f"{self.namespace}/{self.CONFIGMAP_NAME}"
            )
            configmap = self.k8s.core_v1.read_namespaced_config_map(
                name=self.CONFIGMAP_NAME, namespace=self.namespace
            )

            if not configmap.data or self.CONFIGMAP_DATA_KEY not in configmap.data:
                raise ValueError(
                    f"ConfigMap {self.CONFIGMAP_NAME} exists but missing key "
                    f"{self.CONFIGMAP_DATA_KEY}"
                )

            json_data = configmap.data[self.CONFIGMAP_DATA_KEY]
            data = json.loads(json_data)
            self._matrix = CudaCompatibilityMatrix.model_validate(data)
            logger.debug("Successfully loaded CUDA compatibility matrix from ConfigMap")
            return self._matrix

        except ApiException as e:
            if e.status == 404:
                raise ValueError(
                    f"ConfigMap {self.CONFIGMAP_NAME} not found in namespace {self.namespace}"
                ) from e
            raise

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

        # First try exact match
        for mapping in matrix.runtime_images:
            if mapping.image == image:
                return mapping.cuda_version

        # If no exact match, try matching by suffix (handles full registry refs)
        # e.g., "registry.redhat.io/rhaiis/vllm:3.0" matches "rhaiis/vllm:3.0"
        for mapping in matrix.runtime_images:
            if image.endswith(mapping.image) or mapping.image.endswith(image):
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

        # Sort by semantic version using packaging.version
        def safe_parse(v: str) -> Version:
            try:
                return parse(v)
            except Exception:
                # Fallback for non-standard versions
                return Version("0.0.0")

        return sorted(versions, key=safe_parse)

    async def list_all_compute_capabilities(self) -> list[str]:
        """Get list of all GPU compute capabilities in the matrix.

        Returns:
            List of compute capabilities
        """
        matrix = await self.load_matrix()
        return [mapping.compute_capability for mapping in matrix.gpu_compute]
