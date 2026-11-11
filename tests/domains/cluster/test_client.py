"""Tests for the cluster resource inventory client."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.cluster.client import ClusterClient, _get_gpu_product


class TestGetGpuProduct:
    """Test _get_gpu_product helper function."""

    def test_nvidia_gpu_product_label(self) -> None:
        """Test extracting NVIDIA GPU product from labels."""
        labels = {"nvidia.com/gpu.product": "Tesla-T4"}
        result = _get_gpu_product(labels, "nvidia.com/gpu")
        assert result == "Tesla-T4"

    def test_nvidia_gpu_product_alternate_label(self) -> None:
        """Test extracting NVIDIA GPU product from alternate label format."""
        labels = {"nvidia.com/gpu-product": "NVIDIA-A100-SXM4-40GB"}
        result = _get_gpu_product(labels, "nvidia.com/gpu")
        assert result == "NVIDIA-A100-SXM4-40GB"

    def test_amd_gpu_product_label(self) -> None:
        """Test extracting AMD GPU product from labels."""
        labels = {"amd.com/gpu.product": "MI250X"}
        result = _get_gpu_product(labels, "amd.com/gpu")
        assert result == "MI250X"

    def test_no_gpu_labels(self) -> None:
        """Test returns None when no GPU labels present."""
        labels = {"kubernetes.io/os": "linux"}
        result = _get_gpu_product(labels, "nvidia.com/gpu")
        assert result is None

    def test_none_labels(self) -> None:
        """Test returns None when labels is None."""
        result = _get_gpu_product(None, "nvidia.com/gpu")
        assert result is None

    def test_empty_labels(self) -> None:
        """Test returns None when labels is empty."""
        result = _get_gpu_product({}, "nvidia.com/gpu")
        assert result is None

    def test_unknown_gpu_type(self) -> None:
        """Test returns None for unknown GPU resource type."""
        labels = {"nvidia.com/gpu.product": "Tesla-T4"}
        result = _get_gpu_product(labels, "unknown.com/gpu")
        assert result is None

    def test_prefers_first_label_format(self) -> None:
        """Test prefers .product format over -product format."""
        labels = {
            "nvidia.com/gpu.product": "Tesla-T4",
            "nvidia.com/gpu-product": "Other-GPU",
        }
        result = _get_gpu_product(labels, "nvidia.com/gpu")
        assert result == "Tesla-T4"


class TestGetClusterResourcesGpu:
    """Test get_cluster_resources GPU product extraction."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        mock = MagicMock()
        mock.core_v1 = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> ClusterClient:
        """Create a ClusterClient with mocked K8sClient."""
        return ClusterClient(mock_k8s)

    def test_extracts_gpu_product_from_node_labels(
        self, client: ClusterClient, mock_k8s: MagicMock
    ) -> None:
        """Test GPU product is extracted from node labels."""
        node = _make_mock_node(
            "gpu-node-1",
            capacity={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"},
            labels={"nvidia.com/gpu.product": "Tesla-T4"},
        )
        mock_k8s.core_v1.list_node.return_value = MagicMock(items=[node])

        resources = client.get_cluster_resources()

        assert resources.gpu_info is not None
        assert resources.gpu_info.product == "Tesla-T4"
        assert resources.gpu_info.products == ["Tesla-T4"]
        assert resources.nodes[0].gpu_product == "Tesla-T4"

    def test_handles_mixed_gpu_types(self, client: ClusterClient, mock_k8s: MagicMock) -> None:
        """Test tracking multiple GPU products in cluster."""
        node1 = _make_mock_node(
            "gpu-node-1",
            capacity={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "2"},
            labels={"nvidia.com/gpu.product": "Tesla-T4"},
        )
        node2 = _make_mock_node(
            "gpu-node-2",
            capacity={"cpu": "8", "memory": "64Gi", "nvidia.com/gpu": "4"},
            labels={"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-40GB"},
        )
        mock_k8s.core_v1.list_node.return_value = MagicMock(items=[node1, node2])

        resources = client.get_cluster_resources()

        assert resources.gpu_info is not None
        assert resources.gpu_info.product == "Tesla-T4"  # First GPU product
        assert len(resources.gpu_info.products) == 2
        assert "Tesla-T4" in resources.gpu_info.products
        assert "NVIDIA-A100-SXM4-40GB" in resources.gpu_info.products
        assert resources.nodes[0].gpu_product == "Tesla-T4"
        assert resources.nodes[1].gpu_product == "NVIDIA-A100-SXM4-40GB"

    def test_handles_node_without_gpu_labels(
        self, client: ClusterClient, mock_k8s: MagicMock
    ) -> None:
        """Test handles GPU nodes without product labels."""
        node = _make_mock_node(
            "gpu-node-1",
            capacity={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"},
            labels={},  # No GPU product label
        )
        mock_k8s.core_v1.list_node.return_value = MagicMock(items=[node])

        resources = client.get_cluster_resources()

        assert resources.gpu_info is not None
        assert resources.gpu_info.product is None
        assert resources.gpu_info.products == []
        assert resources.nodes[0].gpu_product is None
        assert resources.nodes[0].gpu_count == 1

    def test_handles_non_gpu_nodes(self, client: ClusterClient, mock_k8s: MagicMock) -> None:
        """Test handles nodes without GPUs."""
        node = _make_mock_node(
            "cpu-node",
            capacity={"cpu": "16", "memory": "64Gi"},
            labels={},
        )
        mock_k8s.core_v1.list_node.return_value = MagicMock(items=[node])

        resources = client.get_cluster_resources()

        assert resources.gpu_info is None
        assert resources.nodes[0].gpu_product is None
        assert resources.nodes[0].gpu_count == 0

    def test_deduplicates_gpu_products(self, client: ClusterClient, mock_k8s: MagicMock) -> None:
        """Test same GPU product on multiple nodes is not duplicated."""
        node1 = _make_mock_node(
            "gpu-node-1",
            capacity={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "2"},
            labels={"nvidia.com/gpu.product": "Tesla-T4"},
        )
        node2 = _make_mock_node(
            "gpu-node-2",
            capacity={"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "2"},
            labels={"nvidia.com/gpu.product": "Tesla-T4"},
        )
        mock_k8s.core_v1.list_node.return_value = MagicMock(items=[node1, node2])

        resources = client.get_cluster_resources()

        assert resources.gpu_info is not None
        assert resources.gpu_info.products == ["Tesla-T4"]
        assert resources.gpu_info.total == 4


def _make_mock_node(
    name: str,
    capacity: dict[str, str] | None = None,
    allocatable: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock Kubernetes Node."""
    mock = MagicMock()
    mock.metadata.name = name
    mock.metadata.labels = labels or {}
    mock.status.capacity = capacity or {}
    mock.status.allocatable = allocatable or capacity or {}
    return mock
