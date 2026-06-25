"""Tests for StorageClient operations."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.storage.client import StorageClient
from rhoai_mcp.domains.storage.models import StorageStatus


class TestListStorage:
    """Tests for StorageClient.list_storage."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> StorageClient:
        return StorageClient(mock_k8s)

    def test_list_storage_returns_pvcs(
        self,
        client: StorageClient,
        mock_k8s: MagicMock,
        sample_pvc: MagicMock,
    ) -> None:
        """List storage returns dicts with expected keys."""
        mock_k8s.list_pvcs.return_value = [sample_pvc]

        result = client.list_storage("test-project")

        mock_k8s.list_pvcs.assert_called_once_with(namespace="test-project")
        assert len(result) == 1
        item = result[0]
        assert item["name"] == "test-storage"
        assert item["display_name"] == "Test Storage"
        assert item["size"] == "10Gi"
        assert item["access_modes"] == ["ReadWriteOnce"]
        assert item["storage_class"] == "gp3"
        assert item["status"] == "Bound"
        assert "_source" in item
        assert item["_source"]["kind"] == "PersistentVolumeClaim"

    def test_list_storage_empty(
        self,
        client: StorageClient,
        mock_k8s: MagicMock,
    ) -> None:
        """List storage returns empty list when no PVCs exist."""
        mock_k8s.list_pvcs.return_value = []

        result = client.list_storage("empty-project")

        mock_k8s.list_pvcs.assert_called_once_with(namespace="empty-project")
        assert result == []


class TestGetStorage:
    """Tests for StorageClient.get_storage."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> StorageClient:
        return StorageClient(mock_k8s)

    def test_get_storage_success(
        self,
        client: StorageClient,
        mock_k8s: MagicMock,
        sample_pvc: MagicMock,
    ) -> None:
        """Get storage returns a Storage model from PVC."""
        mock_k8s.get_pvc.return_value = sample_pvc

        storage = client.get_storage("test-storage", "test-project")

        mock_k8s.get_pvc.assert_called_once_with("test-storage", "test-project")
        assert storage.metadata.name == "test-storage"
        assert storage.metadata.namespace == "test-project"
        assert storage.metadata.uid == "pvc-uid"
        assert storage.metadata.kind == "PersistentVolumeClaim"
        assert storage.metadata.api_version == "v1"
        assert storage.display_name == "Test Storage"
        assert storage.size == "10Gi"
        assert storage.access_modes == ["ReadWriteOnce"]
        assert storage.storage_class == "gp3"
        assert storage.status == StorageStatus.BOUND
        assert storage.volume_name is None

    @pytest.mark.parametrize(
        ("phase", "expected_status"),
        [
            ("Bound", StorageStatus.BOUND),
            ("Pending", StorageStatus.PENDING),
            ("Lost", StorageStatus.LOST),
            ("SomethingElse", StorageStatus.UNKNOWN),
            (None, StorageStatus.UNKNOWN),
        ],
    )
    def test_get_storage_status_mapping(
        self,
        client: StorageClient,
        mock_k8s: MagicMock,
        sample_pvc: MagicMock,
        phase: str | None,
        expected_status: StorageStatus,
    ) -> None:
        """Status phase is correctly mapped to StorageStatus enum."""
        sample_pvc.status.phase = phase
        mock_k8s.get_pvc.return_value = sample_pvc

        storage = client.get_storage("test-storage", "test-project")

        assert storage.status == expected_status


class TestCreateStorage:
    """Tests for StorageClient.create_storage."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> StorageClient:
        return StorageClient(mock_k8s)

    def test_create_storage_success(
        self,
        client: StorageClient,
        mock_k8s: MagicMock,
        sample_pvc: MagicMock,
    ) -> None:
        """Create storage calls k8s.create_pvc with correct args."""
        from rhoai_mcp.domains.storage.models import StorageCreate

        mock_k8s.create_pvc.return_value = sample_pvc

        request = StorageCreate(
            name="new-storage",
            namespace="test-project",
            size="20Gi",
            access_mode="ReadWriteOnce",
            storage_class="gp3",
        )
        storage = client.create_storage(request)

        mock_k8s.create_pvc.assert_called_once_with(
            name="new-storage",
            namespace="test-project",
            size="20Gi",
            access_modes=["ReadWriteOnce"],
            storage_class="gp3",
            labels={"opendatahub.io/dashboard": "true"},
            annotations=None,
        )
        assert storage.metadata.name == "test-storage"
        assert storage.status == StorageStatus.BOUND

    def test_create_storage_with_display_name(
        self,
        client: StorageClient,
        mock_k8s: MagicMock,
        sample_pvc: MagicMock,
    ) -> None:
        """Create storage with display_name sets annotation."""
        from rhoai_mcp.domains.storage.models import StorageCreate

        mock_k8s.create_pvc.return_value = sample_pvc

        request = StorageCreate(
            name="new-storage",
            namespace="test-project",
            display_name="My Storage Volume",
            size="10Gi",
            access_mode="ReadWriteOnce",
        )
        client.create_storage(request)

        call_kwargs = mock_k8s.create_pvc.call_args[1]
        assert call_kwargs["annotations"] == {
            "openshift.io/display-name": "My Storage Volume",
        }


class TestDeleteStorage:
    """Tests for StorageClient.delete_storage."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> StorageClient:
        return StorageClient(mock_k8s)

    def test_delete_storage(
        self,
        client: StorageClient,
        mock_k8s: MagicMock,
    ) -> None:
        """Delete storage calls k8s.delete_pvc with correct args."""
        client.delete_storage("test-storage", "test-project")

        mock_k8s.delete_pvc.assert_called_once_with("test-storage", "test-project")
