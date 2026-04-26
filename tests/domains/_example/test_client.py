"""Tests for _example domain client."""

from unittest.mock import MagicMock

from rhoai_mcp.domains._example.client import ExampleClient

EXAMPLE_LABEL_SELECTOR = "rhoai.io/example=true"


class TestListItems:
    """Test ExampleClient.list_items."""

    def test_returns_formatted_list(self) -> None:
        """Returns list of dicts with name, display_name, data, status."""
        cm = MagicMock()
        cm.metadata.name = "item-1"
        cm.metadata.namespace = "ns"
        cm.metadata.uid = "uid-1"
        cm.metadata.creation_timestamp = None
        cm.metadata.labels = {"rhoai.io/example": "true"}
        cm.metadata.annotations = {"openshift.io/display-name": "Item One"}
        cm.data = {"foo": "bar"}

        k8s = MagicMock()
        k8s.core_v1.list_namespaced_config_map.return_value.items = [cm]

        client = ExampleClient(k8s)
        result = client.list_items("ns")

        assert len(result) == 1
        assert result[0]["name"] == "item-1"
        assert result[0]["display_name"] == "Item One"
        assert result[0]["data"] == {"foo": "bar"}
        assert result[0]["status"] == "Active"
        k8s.core_v1.list_namespaced_config_map.assert_called_once_with(
            namespace="ns", label_selector=EXAMPLE_LABEL_SELECTOR
        )

    def test_empty_namespace(self) -> None:
        """Returns empty list when no matching ConfigMaps."""
        k8s = MagicMock()
        k8s.core_v1.list_namespaced_config_map.return_value.items = []

        client = ExampleClient(k8s)
        result = client.list_items("ns")

        assert result == []


class TestGetItem:
    """Test ExampleClient.get_item."""

    def test_returns_model(self) -> None:
        """Returns ExampleItem from a ConfigMap."""
        cm = MagicMock()
        cm.metadata.name = "my-item"
        cm.metadata.namespace = "ns"
        cm.metadata.uid = "uid-1"
        cm.metadata.creation_timestamp = None
        cm.metadata.labels = {}
        cm.metadata.annotations = None
        cm.data = {"key": "val"}

        k8s = MagicMock()
        k8s.core_v1.read_namespaced_config_map.return_value = cm

        client = ExampleClient(k8s)
        item = client.get_item("my-item", "ns")

        assert item.metadata.name == "my-item"
        assert item.data == {"key": "val"}
        k8s.core_v1.read_namespaced_config_map.assert_called_once_with(
            name="my-item", namespace="ns"
        )
