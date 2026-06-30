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
    async def test_load_matrix_configmap_not_found(self) -> None:
        """Test error when ConfigMap doesn't exist in any namespace."""
        mock_k8s = MagicMock()
        mock_k8s.core_v1.read_namespaced_config_map.side_effect = ApiException(status=404)

        client = CudaCompatibilityClient(mock_k8s)

        with pytest.raises(ValueError, match="unavailable"):
            await client.load_matrix()

    @pytest.mark.asyncio
    async def test_load_matrix_namespace_fallback(self) -> None:
        """Test namespace fallback - 404 in first namespace, success in second."""
        mock_k8s = MagicMock()
        sample_data = {
            "RHOAI serving runtime image": [],
            "CUDA toolkit version": [{"cuda_version": ["12.4"], "min_driver_version": ["550.54.14"]}],
            "GPU compute capability": [],
        }

        # First call (redhat-ods-applications) returns 404, second (opendatahub) succeeds
        configmap = MagicMock()
        configmap.data = {"cuda_compat.json": json.dumps(sample_data)}
        mock_k8s.core_v1.read_namespaced_config_map.side_effect = [
            ApiException(status=404),  # First namespace
            configmap,  # Second namespace
        ]

        client = CudaCompatibilityClient(mock_k8s)
        matrix = await client.load_matrix()

        assert len(matrix.cuda_drivers) == 1
        assert mock_k8s.core_v1.read_namespaced_config_map.call_count == 2

    @pytest.mark.asyncio
    async def test_load_matrix_missing_data_key(self) -> None:
        """Test error when ConfigMap exists but missing data key."""
        mock_k8s = MagicMock()
        configmap = MagicMock()
        configmap.data = {"wrong_key": "data"}
        mock_k8s.core_v1.read_namespaced_config_map.return_value = configmap

        client = CudaCompatibilityClient(mock_k8s)

        with pytest.raises(ValueError, match="unavailable"):
            await client.load_matrix()

    @pytest.mark.asyncio
    async def test_load_matrix_invalid_json(self) -> None:
        """Test error when ConfigMap contains invalid JSON."""
        mock_k8s = MagicMock()
        configmap = MagicMock()
        configmap.data = {"cuda_compat.json": "not valid json{"}
        mock_k8s.core_v1.read_namespaced_config_map.return_value = configmap

        client = CudaCompatibilityClient(mock_k8s)

        with pytest.raises(ValueError, match="invalid"):
            await client.load_matrix()

    @pytest.mark.asyncio
    async def test_load_matrix_invalid_schema(self) -> None:
        """Test error when ConfigMap JSON doesn't match schema."""
        mock_k8s = MagicMock()
        configmap = MagicMock()
        # Missing required fields
        configmap.data = {"cuda_compat.json": json.dumps({"invalid": "schema"})}
        mock_k8s.core_v1.read_namespaced_config_map.return_value = configmap

        client = CudaCompatibilityClient(mock_k8s)

        with pytest.raises(ValueError, match="invalid"):
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
    async def test_semantic_version_sorting(self, mock_k8s_client: MagicMock) -> None:
        """Test that versions are sorted semantically, not lexicographically."""
        # Add version data that would sort incorrectly lexicographically
        # Also include -rc version to test release candidate handling
        matrix_data = {
            "RHOAI serving runtime image": [],
            "CUDA toolkit version": [
                {"cuda_version": ["9.0"], "min_driver_version": ["384.81"]},
                {"cuda_version": ["10.0"], "min_driver_version": ["410.48"]},
                {"cuda_version": ["11.0"], "min_driver_version": ["450.80"]},
                {"cuda_version": ["12.0"], "min_driver_version": ["525.60"]},
                {"cuda_version": ["12.1-rc1"], "min_driver_version": ["530.00"]},
            ],
            "GPU compute capability": [],
        }

        configmap = MagicMock()
        configmap.data = {"cuda_compat.json": json.dumps(matrix_data)}
        mock_k8s_client.core_v1.read_namespaced_config_map.return_value = configmap

        client = CudaCompatibilityClient(mock_k8s_client)
        versions = await client.list_all_cuda_versions()

        # Should be 9.0, 10.0, 11.0, 12.0, 12.1-rc1 (rc comes after release)
        assert versions == ["9.0", "10.0", "11.0", "12.0", "12.1-rc1"]

    @pytest.mark.asyncio
    async def test_unparsable_version_handling(self, mock_k8s_client: MagicMock) -> None:
        """Test that unparsable version strings are handled gracefully."""
        matrix_data = {
            "RHOAI serving runtime image": [],
            "CUDA toolkit version": [
                {"cuda_version": ["12.0"], "min_driver_version": ["525.60"]},
                {"cuda_version": ["latest"], "min_driver_version": ["999.99"]},
                {"cuda_version": ["vendor-1"], "min_driver_version": ["888.88"]},
            ],
            "GPU compute capability": [],
        }

        configmap = MagicMock()
        configmap.data = {"cuda_compat.json": json.dumps(matrix_data)}
        mock_k8s_client.core_v1.read_namespaced_config_map.return_value = configmap

        client = CudaCompatibilityClient(mock_k8s_client)
        versions = await client.list_all_cuda_versions()

        # Valid versions should come first, invalid ones last (sorted alphabetically)
        assert versions[0] == "12.0"
        assert "latest" in versions
        assert "vendor-1" in versions
