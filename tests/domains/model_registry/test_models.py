"""Tests for Model Registry models."""

import pytest

from rhoai_mcp.domains.model_registry.models import (
    CustomProperties,
    ModelArtifact,
    ModelVersion,
    RegisteredModel,
)


class TestCustomProperties:
    """Test CustomProperties model."""

    def test_get_property(self) -> None:
        """Test getting a property value."""
        props = CustomProperties(properties={"key": "value"})
        assert props.get("key") == "value"
        assert props.get("missing") is None
        assert props.get("missing", "default") == "default"

    def test_get_float(self) -> None:
        """Test getting a float property."""
        props = CustomProperties(properties={"accuracy": "0.95", "invalid": "not-a-number"})
        assert props.get_float("accuracy") == pytest.approx(0.95)
        assert props.get_float("invalid") == 0.0
        assert props.get_float("missing") == 0.0
        assert props.get_float("missing", 1.0) == 1.0

    def test_get_int(self) -> None:
        """Test getting an int property."""
        props = CustomProperties(properties={"count": "42", "invalid": "not-a-number"})
        assert props.get_int("count") == 42
        assert props.get_int("invalid") == 0
        assert props.get_int("missing") == 0
        assert props.get_int("missing", 10) == 10

    def test_empty_properties(self) -> None:
        """Test empty properties."""
        props = CustomProperties()
        assert props.properties == {}
        assert props.get("anything") is None


class TestModelArtifact:
    """Test ModelArtifact model."""

    def test_artifact_creation(self) -> None:
        """Test creating a model artifact."""
        artifact = ModelArtifact(
            id="artifact-123",
            name="model-weights",
            uri="s3://bucket/path/model.safetensors",
            model_format_name="safetensors",
        )

        assert artifact.id == "artifact-123"
        assert artifact.name == "model-weights"
        assert artifact.uri == "s3://bucket/path/model.safetensors"
        assert artifact.model_format_name == "safetensors"
        assert artifact.description is None
        assert artifact.custom_properties.properties == {}

    def test_artifact_with_all_fields(self) -> None:
        """Test artifact with all optional fields."""
        artifact = ModelArtifact(
            id="artifact-456",
            name="onnx-model",
            uri="s3://bucket/model.onnx",
            description="ONNX exported model",
            model_format_name="onnx",
            model_format_version="1.14",
            storage_key="models-bucket",
            storage_path="exported/model.onnx",
            service_account_name="model-deployer",
            custom_properties=CustomProperties(properties={"quantized": "true"}),
        )

        assert artifact.description == "ONNX exported model"
        assert artifact.storage_key == "models-bucket"
        assert artifact.custom_properties.get("quantized") == "true"


class TestModelVersion:
    """Test ModelVersion model."""

    def test_version_creation(self) -> None:
        """Test creating a model version."""
        version = ModelVersion(
            id="version-123",
            name="v1.0.0",
            registered_model_id="model-456",
        )

        assert version.id == "version-123"
        assert version.name == "v1.0.0"
        assert version.registered_model_id == "model-456"
        assert version.state == "LIVE"
        assert version.artifacts == []

    def test_version_with_artifacts(self) -> None:
        """Test version with artifacts."""
        artifact = ModelArtifact(
            id="artifact-1",
            name="weights",
            uri="s3://bucket/weights.bin",
        )
        version = ModelVersion(
            id="version-1",
            name="v2.0.0",
            registered_model_id="model-1",
            author="ml-team",
            artifacts=[artifact],
        )

        assert version.author == "ml-team"
        assert len(version.artifacts) == 1
        assert version.artifacts[0].name == "weights"


class TestRegisteredModel:
    """Test RegisteredModel model."""

    def test_model_creation(self) -> None:
        """Test creating a registered model."""
        model = RegisteredModel(
            id="model-123",
            name="llama-2-7b",
        )

        assert model.id == "model-123"
        assert model.name == "llama-2-7b"
        assert model.state == "LIVE"
        assert model.versions == []

    def test_model_with_versions(self) -> None:
        """Test model with versions."""
        v1 = ModelVersion(id="v1", name="1.0", registered_model_id="model-1")
        v2 = ModelVersion(id="v2", name="2.0", registered_model_id="model-1")

        model = RegisteredModel(
            id="model-1",
            name="test-model",
            description="A test model",
            owner="data-team",
            versions=[v1, v2],
        )

        assert model.description == "A test model"
        assert model.owner == "data-team"
        assert len(model.versions) == 2

    def test_get_latest_version_empty(self) -> None:
        """Test getting latest version when no versions exist."""
        model = RegisteredModel(id="model-1", name="empty-model")
        assert model.get_latest_version() is None

    def test_get_latest_version(self) -> None:
        """Test getting latest version by creation time."""
        from datetime import datetime, timezone

        v1 = ModelVersion(
            id="v1",
            name="1.0",
            registered_model_id="model-1",
            create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        v2 = ModelVersion(
            id="v2",
            name="2.0",
            registered_model_id="model-1",
            create_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        v3 = ModelVersion(
            id="v3",
            name="3.0",
            registered_model_id="model-1",
            create_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )

        model = RegisteredModel(
            id="model-1",
            name="versioned-model",
            versions=[v1, v3, v2],  # Out of order
        )

        latest = model.get_latest_version()
        assert latest is not None
        assert latest.name == "2.0"
