"""Tests for {{DOMAIN_NAME}} domain models."""

from unittest.mock import MagicMock

from rhoai_mcp.domains.{{DOMAIN_NAME}}.models import (
    {{RESOURCE_CLASS}},
    {{RESOURCE_CLASS}}Status,
)


class Test{{RESOURCE_CLASS}}Status:
    """Test {{RESOURCE_CLASS}}Status enum."""

    def test_ready_value(self) -> None:
        assert {{RESOURCE_CLASS}}Status.READY.value == "Ready"

    def test_unknown_value(self) -> None:
        assert {{RESOURCE_CLASS}}Status.UNKNOWN.value == "Unknown"


class Test{{RESOURCE_CLASS}}:
    """Test {{RESOURCE_CLASS}} model."""

    def test_from_k8s(self) -> None:
        """Test conversion from Kubernetes object."""
        obj = MagicMock()
        obj.metadata.name = "my-resource"
        obj.metadata.namespace = "test-ns"
        obj.metadata.uid = "uid-123"
        obj.metadata.creation_timestamp = None
        obj.metadata.labels = {}
        obj.metadata.annotations = {"openshift.io/display-name": "My Resource"}

        item = {{RESOURCE_CLASS}}.from_k8s(obj)

        assert item.metadata.name == "my-resource"
        assert item.display_name == "My Resource"
