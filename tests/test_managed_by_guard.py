"""Tests for the managed-by label guard on K8sClient delete methods.

Covers:
- K8sClient.delete (CRD-based)
- K8sClient.delete_namespace
- K8sClient.delete_secret
- K8sClient.delete_pvc

Each method should:
- Raise ReadOnlyError when read_only_mode=True
- Raise NotManagedByMCPError when label absent and dangerous_ops=False
- Succeed when label absent and dangerous_ops=True
- Succeed when label present (regardless of dangerous_ops)
"""

from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.clients.base import CRDDefinition, K8sClient
from rhoai_mcp.config import AuthMode, RHOAIConfig
from rhoai_mcp.utils.errors import NotManagedByMCPError, ReadOnlyError
from rhoai_mcp.utils.labels import RHOAILabels

MANAGED_LABELS = RHOAILabels.managed_by_mcp_labels()
UNMANAGED_LABELS: dict[str, str] = {"some-label": "value"}

TEST_CRD = CRDDefinition(
    group="test.example.io",
    version="v1",
    plural="widgets",
    kind="Widget",
)


def _make_config(
    *, read_only: bool = False, dangerous_ops: bool = False
) -> RHOAIConfig:
    return RHOAIConfig(
        auth_mode=AuthMode.KUBECONFIG,
        read_only_mode=read_only,
        enable_dangerous_operations=dangerous_ops,
    )


def _mock_resource(labels: dict[str, str] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.metadata.labels = labels
    return mock


class TestDeleteCRD:
    """Test K8sClient.delete() managed-by guard."""

    def _make_client(self, config: RHOAIConfig) -> K8sClient:
        k8s = K8sClient(config)
        k8s._dynamic_client = MagicMock()
        k8s._core_v1 = MagicMock()
        return k8s

    def test_read_only_raises(self) -> None:
        k8s = self._make_client(_make_config(read_only=True))
        with pytest.raises(ReadOnlyError):
            k8s.delete(TEST_CRD, "w1", namespace="ns")

    def test_unmanaged_no_dangerous_ops_raises(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get", return_value=_mock_resource(UNMANAGED_LABELS)):
            with pytest.raises(NotManagedByMCPError):
                k8s.delete(TEST_CRD, "w1", namespace="ns")

    def test_unmanaged_with_dangerous_ops_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=True))
        with patch.object(k8s, "get", return_value=_mock_resource(UNMANAGED_LABELS)):
            with patch.object(k8s, "get_resource") as mock_res:
                k8s.delete(TEST_CRD, "w1", namespace="ns")
                mock_res.return_value.delete.assert_called_once()

    def test_managed_without_dangerous_ops_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get", return_value=_mock_resource(MANAGED_LABELS)):
            with patch.object(k8s, "get_resource") as mock_res:
                k8s.delete(TEST_CRD, "w1", namespace="ns")
                mock_res.return_value.delete.assert_called_once()

    def test_managed_with_dangerous_ops_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=True))
        with patch.object(k8s, "get", return_value=_mock_resource(MANAGED_LABELS)):
            with patch.object(k8s, "get_resource") as mock_res:
                k8s.delete(TEST_CRD, "w1", namespace="ns")
                mock_res.return_value.delete.assert_called_once()

    def test_none_labels_treated_as_unmanaged(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get", return_value=_mock_resource(None)):
            with pytest.raises(NotManagedByMCPError):
                k8s.delete(TEST_CRD, "w1", namespace="ns")


class TestDeleteNamespace:
    """Test K8sClient.delete_namespace() managed-by guard."""

    def _make_client(self, config: RHOAIConfig) -> K8sClient:
        k8s = K8sClient(config)
        k8s._dynamic_client = MagicMock()
        k8s._core_v1 = MagicMock()
        return k8s

    def test_read_only_raises(self) -> None:
        k8s = self._make_client(_make_config(read_only=True))
        with pytest.raises(ReadOnlyError):
            k8s.delete_namespace("my-ns")

    def test_unmanaged_no_dangerous_ops_raises(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get_namespace", return_value=_mock_resource(UNMANAGED_LABELS)):
            with pytest.raises(NotManagedByMCPError):
                k8s.delete_namespace("my-ns")

    def test_unmanaged_with_dangerous_ops_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=True))
        with patch.object(k8s, "get_namespace", return_value=_mock_resource(UNMANAGED_LABELS)):
            k8s.delete_namespace("my-ns")
            k8s._core_v1.delete_namespace.assert_called_once_with(name="my-ns")

    def test_managed_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get_namespace", return_value=_mock_resource(MANAGED_LABELS)):
            k8s.delete_namespace("my-ns")
            k8s._core_v1.delete_namespace.assert_called_once_with(name="my-ns")


class TestDeleteSecret:
    """Test K8sClient.delete_secret() managed-by guard."""

    def _make_client(self, config: RHOAIConfig) -> K8sClient:
        k8s = K8sClient(config)
        k8s._dynamic_client = MagicMock()
        k8s._core_v1 = MagicMock()
        return k8s

    def test_read_only_raises(self) -> None:
        k8s = self._make_client(_make_config(read_only=True))
        with pytest.raises(ReadOnlyError):
            k8s.delete_secret("s1", "ns")

    def test_unmanaged_no_dangerous_ops_raises(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get_secret", return_value=_mock_resource(UNMANAGED_LABELS)):
            with pytest.raises(NotManagedByMCPError):
                k8s.delete_secret("s1", "ns")

    def test_unmanaged_with_dangerous_ops_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=True))
        with patch.object(k8s, "get_secret", return_value=_mock_resource(UNMANAGED_LABELS)):
            k8s.delete_secret("s1", "ns")
            k8s._core_v1.delete_namespaced_secret.assert_called_once()

    def test_managed_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get_secret", return_value=_mock_resource(MANAGED_LABELS)):
            k8s.delete_secret("s1", "ns")
            k8s._core_v1.delete_namespaced_secret.assert_called_once()


class TestDeletePVC:
    """Test K8sClient.delete_pvc() managed-by guard."""

    def _make_client(self, config: RHOAIConfig) -> K8sClient:
        k8s = K8sClient(config)
        k8s._dynamic_client = MagicMock()
        k8s._core_v1 = MagicMock()
        return k8s

    def test_read_only_raises(self) -> None:
        k8s = self._make_client(_make_config(read_only=True))
        with pytest.raises(ReadOnlyError):
            k8s.delete_pvc("pvc1", "ns")

    def test_unmanaged_no_dangerous_ops_raises(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get_pvc", return_value=_mock_resource(UNMANAGED_LABELS)):
            with pytest.raises(NotManagedByMCPError):
                k8s.delete_pvc("pvc1", "ns")

    def test_unmanaged_with_dangerous_ops_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=True))
        with patch.object(k8s, "get_pvc", return_value=_mock_resource(UNMANAGED_LABELS)):
            k8s.delete_pvc("pvc1", "ns")
            k8s._core_v1.delete_namespaced_persistent_volume_claim.assert_called_once()

    def test_managed_succeeds(self) -> None:
        k8s = self._make_client(_make_config(dangerous_ops=False))
        with patch.object(k8s, "get_pvc", return_value=_mock_resource(MANAGED_LABELS)):
            k8s.delete_pvc("pvc1", "ns")
            k8s._core_v1.delete_namespaced_persistent_volume_claim.assert_called_once()
