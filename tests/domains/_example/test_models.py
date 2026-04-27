"""Tests for _example domain models."""

from unittest.mock import MagicMock

from rhoai_mcp.domains._example.models import ExampleItem, ExampleStatus


class TestExampleStatus:
    """Test ExampleStatus enum."""

    def test_active_value(self) -> None:
        assert ExampleStatus.ACTIVE.value == "Active"

    def test_inactive_value(self) -> None:
        assert ExampleStatus.INACTIVE.value == "Inactive"


class TestExampleItem:
    """Test ExampleItem model."""

    def test_from_configmap_basic(self) -> None:
        """Test conversion from a K8s ConfigMap object."""
        cm = MagicMock()
        cm.metadata.name = "my-item"
        cm.metadata.namespace = "test-ns"
        cm.metadata.uid = "uid-123"
        cm.metadata.creation_timestamp = None
        cm.metadata.labels = {"rhoai.io/example": "true"}
        cm.metadata.annotations = {"openshift.io/display-name": "My Item"}
        cm.data = {"key1": "value1", "key2": "value2"}

        item = ExampleItem.from_configmap(cm)

        assert item.metadata.name == "my-item"
        assert item.metadata.namespace == "test-ns"
        assert item.display_name == "My Item"
        assert item.data == {"key1": "value1", "key2": "value2"}
        assert item.status == ExampleStatus.ACTIVE

    def test_from_configmap_no_data(self) -> None:
        """ConfigMap with no data section defaults to empty dict."""
        cm = MagicMock()
        cm.metadata.name = "empty-item"
        cm.metadata.namespace = "test-ns"
        cm.metadata.uid = "uid-456"
        cm.metadata.creation_timestamp = None
        cm.metadata.labels = {}
        cm.metadata.annotations = None
        cm.data = None

        item = ExampleItem.from_configmap(cm)

        assert item.data == {}
        assert item.display_name is None

    def test_from_configmap_inactive_label(self) -> None:
        """ConfigMap with enabled=false label is INACTIVE."""
        cm = MagicMock()
        cm.metadata.name = "disabled"
        cm.metadata.namespace = "test-ns"
        cm.metadata.uid = "uid-789"
        cm.metadata.creation_timestamp = None
        cm.metadata.labels = {"rhoai.io/example-enabled": "false"}
        cm.metadata.annotations = None
        cm.data = {}

        item = ExampleItem.from_configmap(cm)

        assert item.status == ExampleStatus.INACTIVE
