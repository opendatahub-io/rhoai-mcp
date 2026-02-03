"""Pydantic models for Model Registry entities."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CustomProperties(BaseModel):
    """Custom properties for model metadata.

    The Model Registry API allows arbitrary key-value properties to be
    attached to models, versions, and artifacts for metadata like metrics,
    labels, or experiment tracking information.
    """

    properties: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float property value."""
        value = self.properties.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an int property value."""
        value = self.properties.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default


class ModelArtifact(BaseModel):
    """Model artifact in the registry.

    Artifacts represent the actual model files (weights, configs) stored
    in object storage. Each model version can have multiple artifacts.
    """

    id: str = Field(..., description="Artifact ID")
    name: str = Field(..., description="Artifact name")
    uri: str = Field(..., description="Storage URI (e.g., s3://bucket/path)")
    description: str | None = Field(None, description="Artifact description")
    model_format_name: str | None = Field(None, description="Format name (e.g., onnx, pytorch)")
    model_format_version: str | None = Field(None, description="Format version")
    storage_key: str | None = Field(None, description="Storage key for S3")
    storage_path: str | None = Field(None, description="Path within storage")
    service_account_name: str | None = Field(None, description="Service account for access")
    custom_properties: CustomProperties = Field(default_factory=CustomProperties)
    create_time: datetime | None = Field(None, description="Creation timestamp")
    update_time: datetime | None = Field(None, description="Last update timestamp")


class ModelVersion(BaseModel):
    """Model version in the registry.

    Versions track different iterations of a registered model, each
    potentially with different artifacts, properties, and state.
    """

    id: str = Field(..., description="Version ID")
    name: str = Field(..., description="Version name (e.g., 'v1', '1.0.0')")
    registered_model_id: str = Field(..., description="Parent model ID")
    state: str = Field("LIVE", description="Version state (LIVE, ARCHIVED)")
    description: str | None = Field(None, description="Version description")
    author: str | None = Field(None, description="Version author")
    custom_properties: CustomProperties = Field(default_factory=CustomProperties)
    artifacts: list[ModelArtifact] = Field(default_factory=list, description="Version artifacts")
    create_time: datetime | None = Field(None, description="Creation timestamp")
    update_time: datetime | None = Field(None, description="Last update timestamp")


class RegisteredModel(BaseModel):
    """Registered model in the registry.

    A registered model is the top-level entity that groups related model
    versions together under a common name and ownership.
    """

    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    description: str | None = Field(None, description="Model description")
    owner: str | None = Field(None, description="Model owner")
    state: str = Field("LIVE", description="Model state (LIVE, ARCHIVED)")
    custom_properties: CustomProperties = Field(default_factory=CustomProperties)
    versions: list[ModelVersion] = Field(default_factory=list, description="Model versions")
    create_time: datetime | None = Field(None, description="Creation timestamp")
    update_time: datetime | None = Field(None, description="Last update timestamp")

    def get_latest_version(self) -> ModelVersion | None:
        """Get the most recent version by creation time."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.create_time or datetime.min)
