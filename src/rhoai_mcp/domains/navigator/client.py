"""ConfigMap loader for CUDA compatibility matrix."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kubernetes.client.exceptions import ApiException

from rhoai_mcp.domains.navigator.models import CudaCompatibilityMatrix

if TYPE_CHECKING:
    from rhoai_mcp.clients.base import KubernetesClient

logger = logging.getLogger(__name__)


class CudaCompatibilityClient:
    """Client for loading CUDA compatibility matrix from ConfigMap or static file."""

    CONFIGMAP_NAME = "cuda-compatibility-matrix"
    CONFIGMAP_NAMESPACE = "redhat-ods-applications"
    CONFIGMAP_DATA_KEY = "cuda_compat.json"

    # Path to the static fallback JSON file (relative to this file)
    STATIC_DATA_PATH = Path(__file__).parent / "data" / "cuda_compat_default.json"

    def __init__(self, k8s_client: KubernetesClient) -> None:
        """Initialize the CUDA compatibility client.

        Args:
            k8s_client: Kubernetes client for ConfigMap access
        """
        self.k8s = k8s_client
        self._matrix: CudaCompatibilityMatrix | None = None

    def load_matrix(self) -> CudaCompatibilityMatrix:
        """Load CUDA compatibility matrix from ConfigMap or fallback to static file.

        Returns:
            CudaCompatibilityMatrix: The loaded compatibility matrix

        Raises:
            ValueError: If the matrix data is invalid or missing
        """
        if self._matrix is not None:
            return self._matrix

        # Try loading from ConfigMap first
        try:
            logger.info(
                f"Attempting to load CUDA compatibility matrix from ConfigMap "
                f"{self.CONFIGMAP_NAMESPACE}/{self.CONFIGMAP_NAME}"
            )
            matrix = self._load_from_configmap()
            if matrix:
                logger.info("Successfully loaded CUDA compatibility matrix from ConfigMap")
                self._matrix = matrix
                return self._matrix
        except Exception as e:
            logger.warning(f"Failed to load from ConfigMap: {e}")

        # Fallback to static file
        logger.info("Falling back to static CUDA compatibility matrix")
        self._matrix = self._load_from_static_file()
        return self._matrix

    def _load_from_configmap(self) -> CudaCompatibilityMatrix | None:
        """Load matrix from Kubernetes ConfigMap.

        Returns:
            CudaCompatibilityMatrix if found, None otherwise

        Raises:
            ApiException: If there's a Kubernetes API error
        """
        try:
            configmap = self.k8s.core_v1.read_namespaced_config_map(
                name=self.CONFIGMAP_NAME, namespace=self.CONFIGMAP_NAMESPACE
            )

            if not configmap.data or self.CONFIGMAP_DATA_KEY not in configmap.data:
                logger.warning(
                    f"ConfigMap {self.CONFIGMAP_NAME} exists but missing key "
                    f"{self.CONFIGMAP_DATA_KEY}"
                )
                return None

            json_data = configmap.data[self.CONFIGMAP_DATA_KEY]
            data = json.loads(json_data)
            return CudaCompatibilityMatrix.model_validate(data)

        except ApiException as e:
            if e.status == 404:
                logger.info(f"ConfigMap {self.CONFIGMAP_NAME} not found in cluster")
                return None
            raise

    def _load_from_static_file(self) -> CudaCompatibilityMatrix:
        """Load matrix from static JSON file.

        Returns:
            CudaCompatibilityMatrix: The loaded matrix

        Raises:
            ValueError: If the static file is missing or invalid
        """
        if not self.STATIC_DATA_PATH.exists():
            raise ValueError(
                f"Static CUDA compatibility data not found at {self.STATIC_DATA_PATH}"
            )

        with open(self.STATIC_DATA_PATH) as f:
            data = json.load(f)

        return CudaCompatibilityMatrix.model_validate(data)

    def get_cuda_for_runtime(self, image: str) -> list[str]:
        """Get CUDA versions for a given RHOAI runtime image.

        Args:
            image: Container image reference

        Returns:
            List of CUDA versions supported by this runtime image

        Raises:
            ValueError: If the image is not found in the matrix
        """
        matrix = self.load_matrix()

        for mapping in matrix.runtime_images:
            if mapping.image == image:
                return mapping.cuda_version

        raise ValueError(f"Runtime image not found in compatibility matrix: {image}")

    def get_min_driver_for_cuda(self, cuda_version: str) -> list[str]:
        """Get minimum driver version for a given CUDA toolkit version.

        Args:
            cuda_version: CUDA toolkit version (e.g., "12.4")

        Returns:
            List of minimum driver versions (typically one element)

        Raises:
            ValueError: If the CUDA version is not found in the matrix
        """
        matrix = self.load_matrix()

        for mapping in matrix.cuda_drivers:
            if cuda_version in mapping.cuda_version:
                return mapping.min_driver_version

        raise ValueError(f"CUDA version not found in compatibility matrix: {cuda_version}")

    def get_supported_cuda_for_compute(self, compute_capability: str) -> list[str]:
        """Get supported CUDA versions for a given GPU compute capability.

        Args:
            compute_capability: GPU compute capability (e.g., "8.0")

        Returns:
            List of supported CUDA versions

        Raises:
            ValueError: If the compute capability is not found in the matrix
        """
        matrix = self.load_matrix()

        for mapping in matrix.gpu_compute:
            if mapping.compute_capability == compute_capability:
                return mapping.supported_cuda_versions

        raise ValueError(
            f"GPU compute capability not found in compatibility matrix: {compute_capability}"
        )

    def list_all_runtime_images(self) -> list[str]:
        """Get list of all RHOAI runtime images in the matrix.

        Returns:
            List of runtime image names
        """
        matrix = self.load_matrix()
        return [mapping.image for mapping in matrix.runtime_images]

    def list_all_cuda_versions(self) -> list[str]:
        """Get list of all CUDA versions in the matrix.

        Returns:
            List of CUDA versions
        """
        matrix = self.load_matrix()
        versions = set()
        for mapping in matrix.cuda_drivers:
            versions.update(mapping.cuda_version)
        return sorted(versions)

    def list_all_compute_capabilities(self) -> list[str]:
        """Get list of all GPU compute capabilities in the matrix.

        Returns:
            List of compute capabilities
        """
        matrix = self.load_matrix()
        return [mapping.compute_capability for mapping in matrix.gpu_compute]
