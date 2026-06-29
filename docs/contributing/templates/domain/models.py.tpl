"""Pydantic models for {{DOMAIN_DESCRIPTION}}."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rhoai_mcp.models.common import ResourceMetadata


class {{RESOURCE_CLASS}}Status(str, Enum):
    """Status values for {{RESOURCE_NAME}} resources."""

    READY = "Ready"
    PENDING = "Pending"
    ERROR = "Error"
    UNKNOWN = "Unknown"


class {{RESOURCE_CLASS}}(BaseModel):
    """{{RESOURCE_CLASS}} resource representation."""

    metadata: ResourceMetadata
    display_name: str | None = Field(None, description="Human-readable display name")
    status: {{RESOURCE_CLASS}}Status = Field(
        {{RESOURCE_CLASS}}Status.UNKNOWN, description="Resource status"
    )

    # TODO: Add domain-specific fields here.

    @classmethod
    def from_k8s(cls, obj: Any) -> "{{RESOURCE_CLASS}}":
        """Create from a Kubernetes resource object.

        Args:
            obj: Raw Kubernetes API object (dict-like or ResourceField).

        Returns:
            Parsed {{RESOURCE_CLASS}} instance.
        """
        meta = obj.metadata
        annotations = meta.annotations or {}

        # TODO: Map Kubernetes status to {{RESOURCE_CLASS}}Status.
        status = {{RESOURCE_CLASS}}Status.UNKNOWN

        return cls(
            metadata=ResourceMetadata.from_k8s_metadata(
                meta,
                # For CRD-based domains, use kind="{{CRD_KIND}}" and
                # api_version="{{CRD_GROUP}}/{{CRD_VERSION}}".
                # For core API resources, use the appropriate kind/apiVersion.
                kind="{{CRD_KIND}}",
                api_version="{{CRD_GROUP}}/{{CRD_VERSION}}",
            ),
            display_name=annotations.get("openshift.io/display-name"),
            status=status,
            # TODO: Populate domain-specific fields from obj.
        )
