"""Tests for {{DOMAIN_NAME}} domain client."""

from unittest.mock import MagicMock

from rhoai_mcp.domains.{{DOMAIN_NAME}}.client import {{DOMAIN_CLASS}}Client


class TestList{{RESOURCE_CLASS}}s:
    """Test {{DOMAIN_CLASS}}Client.list_{{RESOURCE_NAME}}s."""

    def test_returns_formatted_list(self) -> None:
        """Returns list of dicts with expected fields."""
        # TODO: Create a mock K8s object matching your resource type.
        obj = MagicMock()
        obj.metadata.name = "item-1"
        obj.metadata.namespace = "test-ns"
        obj.metadata.uid = "uid-1"
        obj.metadata.creation_timestamp = None
        obj.metadata.labels = {}
        obj.metadata.annotations = {}

        k8s = MagicMock()
        # TODO: Mock the correct K8s API call.
        # For CRDs: k8s.list_resources.return_value = [obj]
        # For core API: k8s.core_v1.list_namespaced_X.return_value.items = [obj]

        client = {{DOMAIN_CLASS}}Client(k8s)
        result = client.list_{{RESOURCE_NAME}}s("test-ns")

        assert len(result) == 1
        assert result[0]["name"] == "item-1"

    def test_empty_namespace(self) -> None:
        """Returns empty list when no resources exist."""
        k8s = MagicMock()
        # TODO: Mock empty response.

        client = {{DOMAIN_CLASS}}Client(k8s)
        result = client.list_{{RESOURCE_NAME}}s("test-ns")

        assert result == []


class TestGet{{RESOURCE_CLASS}}:
    """Test {{DOMAIN_CLASS}}Client.get_{{RESOURCE_NAME}}."""

    def test_returns_model(self) -> None:
        """Returns parsed {{RESOURCE_CLASS}} model."""
        obj = MagicMock()
        obj.metadata.name = "my-item"
        obj.metadata.namespace = "test-ns"
        obj.metadata.uid = "uid-1"
        obj.metadata.creation_timestamp = None
        obj.metadata.labels = {}
        obj.metadata.annotations = {}

        k8s = MagicMock()
        # TODO: Mock the correct K8s API call.

        client = {{DOMAIN_CLASS}}Client(k8s)
        item = client.get_{{RESOURCE_NAME}}("my-item", "test-ns")

        assert item.metadata.name == "my-item"
