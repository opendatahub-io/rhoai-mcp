"""Tests for CudaCompatibilityClient."""

import json
from unittest.mock import MagicMock

import pytest
from kubernetes.client.exceptions import ApiException

from rhoai_mcp.domains.model_runtimes.client import CudaCompatibilityClient
from rhoai_mcp.domains.model_runtimes.models import CudaCompatibilityMatrix


class TestCudaCompatibilityClient:
    """Test CudaCompatibilityClient methods."""

    @pytest.mark.asyncio
    async def test_load_matrix_from_configmap(
        self, mock_k8s_client: MagicMock
    ) -> None:
        """Test loading matrix from ConfigMap."""
        client = CudaCompatibilityClient(mock_k8s_client, namespace="test-namespace")

        matrix = await client.load_matrix()

        assert isinstance(matrix, CudaCompatibilityMatrix)
        assert len(matrix.runtime_images) == 1
        assert len(matrix.cuda_drivers) == 1
        assert len(matrix.gpu_compute) == 1
        mock_k8s_client.core_v1.read_namespaced_config_map.assert_called_once_with(
            name="cuda-compatibility-matrix", namespace="test-namespace"
        )

    @pytest.mark.asyncio
    async def test_load_matrix_caching(self, mock_k8s_client: MagicMock) -> None:
        """Test that matrix is cached after first load."""
        client = CudaCompatibilityClient(mock_k8s_client)

        # Load twice
        matrix1 = await client.load_matrix()
        matrix2 = await client.load_matrix()

        # Matrix should be cached (same instance)
        assert matrix1 is matrix2
        mock_k8s_client.core_v1.read_namespaced_config_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_matrix_configmap_not_found(self) -> None:
        """Test error when ConfigMap doesn't exist."""
        mock_k8s = MagicMock()
        mock_k8s.core_v1.read_namespaced_config_map.side_effect = ApiException(status=404)

        client = CudaCompatibilityClient(mock_k8s)

        with pytest.raises(ValueError, match="ConfigMap .* not found"):
            await client.load_matrix()

    @pytest.mark.asyncio
    async def test_load_matrix_missing_data_key(self) -> None:
        """Test error when ConfigMap exists but missing data key."""
        mock_k8s = MagicMock()
        configmap = MagicMock()
        configmap.data = {"wrong_key": "data"}
        mock_k8s.core_v1.read_namespaced_config_map.return_value = configmap

        client = CudaCompatibilityClient(mock_k8s)

        with pytest.raises(ValueError, match="missing key"):
            await client.load_matrix()

    @pytest.mark.asyncio
    async def test_get_cuda_for_runtime_exact_match(self, mock_k8s_client: MagicMock) -> None:
        """Test getting CUDA version for runtime with exact match."""
        client = CudaCompatibilityClient(mock_k8s_client)

        cuda_versions = await client.get_cuda_for_runtime("rhaiis/vllm-cuda-rhel9:3.0")

        assert cuda_versions == ["12.4"]

    @pytest.mark.asyncio
    async def test_get_cuda_for_runtime_full_registry_ref(
        self, mock_k8s_client: MagicMock
    ) -> None:
        """Test getting CUDA version with full registry reference."""
        client = CudaCompatibilityClient(mock_k8s_client)

        # Full registry ref should match short name
        cuda_versions = await client.get_cuda_for_runtime(
            "registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.0"
        )

        assert cuda_versions == ["12.4"]

    @pytest.mark.asyncio
    async def test_get_cuda_for_runtime_not_found(self, mock_k8s_client: MagicMock) -> None:
        """Test error when runtime image not found."""
        client = CudaCompatibilityClient(mock_k8s_client)

        with pytest.raises(ValueError, match="Runtime image not found"):
            await client.get_cuda_for_runtime("nonexistent/image:1.0")

    @pytest.mark.asyncio
    async def test_get_min_driver_for_cuda(self, mock_k8s_client: MagicMock) -> None:
        """Test getting minimum driver version for CUDA."""
        client = CudaCompatibilityClient(mock_k8s_client)

        driver_versions = await client.get_min_driver_for_cuda("12.4")

        assert driver_versions == ["550.54.14"]

    @pytest.mark.asyncio
    async def test_get_min_driver_for_cuda_not_found(self, mock_k8s_client: MagicMock) -> None:
        """Test error when CUDA version not found."""
        client = CudaCompatibilityClient(mock_k8s_client)

        with pytest.raises(ValueError, match="CUDA version not found"):
            await client.get_min_driver_for_cuda("99.9")

    @pytest.mark.asyncio
    async def test_get_supported_cuda_for_compute(self, mock_k8s_client: MagicMock) -> None:
        """Test getting supported CUDA versions for GPU compute capability."""
        client = CudaCompatibilityClient(mock_k8s_client)

        cuda_versions = await client.get_supported_cuda_for_compute("8.0")

        assert cuda_versions == ["12.4"]

    @pytest.mark.asyncio
    async def test_get_supported_cuda_for_compute_not_found(
        self, mock_k8s_client: MagicMock
    ) -> None:
        """Test error when compute capability not found."""
        client = CudaCompatibilityClient(mock_k8s_client)

        with pytest.raises(ValueError, match="GPU compute capability not found"):
            await client.get_supported_cuda_for_compute("99.9")

    @pytest.mark.asyncio
    async def test_list_all_runtime_images(self, mock_k8s_client: MagicMock) -> None:
        """Test listing all runtime images."""
        client = CudaCompatibilityClient(mock_k8s_client)

        images = await client.list_all_runtime_images()

        assert len(images) == 1
        assert "rhaiis/vllm-cuda-rhel9:3.0" in images

    @pytest.mark.asyncio
    async def test_list_all_cuda_versions(self, mock_k8s_client: MagicMock) -> None:
        """Test listing all CUDA versions with semantic sorting."""
        client = CudaCompatibilityClient(mock_k8s_client)

        versions = await client.list_all_cuda_versions()

        # Should be sorted semantically
        assert versions == ["12.4"]

    @pytest.mark.asyncio
    async def test_list_all_compute_capabilities(self, mock_k8s_client: MagicMock) -> None:
        """Test listing all GPU compute capabilities."""
        client = CudaCompatibilityClient(mock_k8s_client)

        capabilities = await client.list_all_compute_capabilities()

        assert len(capabilities) == 1
        assert "8.0" in capabilities

    @pytest.mark.asyncio
    async def test_default_namespace(self, mock_k8s_client: MagicMock) -> None:
        """Test that default namespace is used when not specified."""
        client = CudaCompatibilityClient(mock_k8s_client)

        await client.load_matrix()

        # Should use default namespace
        mock_k8s_client.core_v1.read_namespaced_config_map.assert_called_once_with(
            name="cuda-compatibility-matrix", namespace="redhat-ods-applications"
        )

    @pytest.mark.asyncio
    async def test_semantic_version_sorting(self, mock_k8s_client: MagicMock) -> None:
        """Test that versions are sorted semantically, not lexicographically."""
        # Add version data that would sort incorrectly lexicographically
        matrix_data = {
            "RHOAI serving runtime image": [],
            "CUDA toolkit version": [
                {"cuda_version": ["9.0"], "min_driver_version": ["384.81"]},
                {"cuda_version": ["10.0"], "min_driver_version": ["410.48"]},
                {"cuda_version": ["11.0"], "min_driver_version": ["450.80"]},
                {"cuda_version": ["12.0"], "min_driver_version": ["525.60"]},
            ],
            "GPU compute capability": [],
        }

        configmap = MagicMock()
        configmap.data = {"cuda_compat.json": json.dumps(matrix_data)}
        mock_k8s_client.core_v1.read_namespaced_config_map.return_value = configmap

        client = CudaCompatibilityClient(mock_k8s_client)
        versions = await client.list_all_cuda_versions()

        # Should be 9.0, 10.0, 11.0, 12.0 (not 10.0, 11.0, 12.0, 9.0)
        assert versions == ["9.0", "10.0", "11.0", "12.0"]
