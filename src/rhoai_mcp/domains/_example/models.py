"""Pydantic models for the example domain."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import ResourceMetadata


class ExampleStatus(str, Enum):
    """Status values for example items."""

    ACTIVE = "Active"
    INACTIVE = "Inactive"


class ExampleItem(BaseModel):
    """An example resource backed by a Kubernetes ConfigMap."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Human-readable display name")
    data: dict[str, str] = Field(default_factory=dict, description="ConfigMap data entries")
    status: ExampleStatus = Field(ExampleStatus.ACTIVE, description="Item status")

    @classmethod
    def from_configmap(cls, cm: Any) -> "ExampleItem":
        """Create from a Kubernetes ConfigMap object."""
        meta = cm.metadata
        annotations = meta.annotations or {}
        labels = meta.labels or {}

        status = ExampleStatus.ACTIVE
        if labels.get("rhoai.io/example-enabled") == "false":
            status = ExampleStatus.INACTIVE

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                meta,
                kind="ConfigMap",
                api_version="v1",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            data=dict(cm.data) if cm.data else {},
            status=status,
        )
