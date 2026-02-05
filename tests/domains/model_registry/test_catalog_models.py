"""Tests for Model Catalog models."""

import pytest

from rhoai_mcp.domains.model_registry.catalog_models import (
    CatalogListResponse,
    CatalogModel,
    CatalogModelArtifact,
    CatalogSource,
    CatalogSourcesResponse,
)


class TestCatalogModelArtifact:
    """Test CatalogModelArtifact model."""

    def test_artifact_creation_minimal(self) -> None:
        """Test creating an artifact with minimal fields."""
        artifact = CatalogModelArtifact(uri="s3://bucket/model.safetensors")

        assert artifact.uri == "s3://bucket/model.safetensors"
        assert artifact.format is None
        assert artifact.size is None
        assert artifact.quantization is None

    def test_artifact_creation_full(self) -> None:
        """Test creating an artifact with all fields."""
        artifact = CatalogModelArtifact(
            uri="oci://registry.example.com/models/llama-7b:v1",
            format="safetensors",
            size="7.5 GB",
            quantization="fp16",
        )

        assert artifact.uri == "oci://registry.example.com/models/llama-7b:v1"
        assert artifact.format == "safetensors"
        assert artifact.size == "7.5 GB"
        assert artifact.quantization == "fp16"


class TestCatalogModel:
    """Test CatalogModel model."""

    def test_model_creation_minimal(self) -> None:
        """Test creating a model with minimal required fields."""
        model = CatalogModel(
            name="granite-3b",
            source_label="Red Hat AI validated",
        )

        assert model.name == "granite-3b"
        assert model.source_label == "Red Hat AI validated"
        assert model.description is None
        assert model.provider is None
        assert model.task_type is None
        assert model.tags == []
        assert model.artifacts == []

    def test_model_creation_full(self) -> None:
        """Test creating a model with all fields."""
        artifact = CatalogModelArtifact(
            uri="oci://registry/models/mistral-7b:latest",
            format="safetensors",
            size="14.5 GB",
        )

        model = CatalogModel(
            name="mistral-7b-instruct-v0.3",
            description="Mistral 7B Instruct fine-tuned model",
            provider="Mistral AI",
            source_label="Red Hat AI validated",
            task_type="text-generation",
            tags=["llm", "instruct", "7b"],
            size="7B parameters",
            license="Apache 2.0",
            artifacts=[artifact],
            long_description="Extended description with more details...",
            readme="# Mistral 7B Instruct\n\nModel readme content...",
        )

        assert model.name == "mistral-7b-instruct-v0.3"
        assert model.description == "Mistral 7B Instruct fine-tuned model"
        assert model.provider == "Mistral AI"
        assert model.source_label == "Red Hat AI validated"
        assert model.task_type == "text-generation"
        assert model.tags == ["llm", "instruct", "7b"]
        assert model.size == "7B parameters"
        assert model.license == "Apache 2.0"
        assert len(model.artifacts) == 1
        assert model.artifacts[0].uri == "oci://registry/models/mistral-7b:latest"
        assert model.long_description is not None
        assert model.readme is not None

    def test_model_with_multiple_artifacts(self) -> None:
        """Test model with multiple artifacts."""
        artifacts = [
            CatalogModelArtifact(uri="s3://bucket/model-fp16.safetensors", format="safetensors"),
            CatalogModelArtifact(uri="s3://bucket/model-int8.gguf", format="gguf", quantization="int8"),
        ]

        model = CatalogModel(
            name="llama-2-7b",
            source_label="Community",
            artifacts=artifacts,
        )

        assert len(model.artifacts) == 2
        assert model.artifacts[0].format == "safetensors"
        assert model.artifacts[1].quantization == "int8"


class TestCatalogSource:
    """Test CatalogSource model."""

    def test_source_creation_minimal(self) -> None:
        """Test creating a source with required fields."""
        source = CatalogSource(
            name="rhoai",
            label="Red Hat AI validated",
        )

        assert source.name == "rhoai"
        assert source.label == "Red Hat AI validated"
        assert source.model_count == 0  # Default
        assert source.description is None

    def test_source_creation_full(self) -> None:
        """Test creating a source with all fields."""
        source = CatalogSource(
            name="community",
            label="Community",
            model_count=42,
            description="Community-contributed models",
        )

        assert source.name == "community"
        assert source.label == "Community"
        assert source.model_count == 42
        assert source.description == "Community-contributed models"


class TestCatalogListResponse:
    """Test CatalogListResponse model."""

    def test_empty_response(self) -> None:
        """Test empty response."""
        response = CatalogListResponse()

        assert response.models == []
        assert response.total_count == 0
        assert response.page_size == 50  # Default
        assert response.next_page_token is None

    def test_response_with_models(self) -> None:
        """Test response with models."""
        models = [
            CatalogModel(name="model-1", source_label="rhoai"),
            CatalogModel(name="model-2", source_label="rhoai"),
        ]

        response = CatalogListResponse(
            models=models,
            total_count=10,
            page_size=2,
            next_page_token="page2",
        )

        assert len(response.models) == 2
        assert response.total_count == 10
        assert response.page_size == 2
        assert response.next_page_token == "page2"


class TestCatalogSourcesResponse:
    """Test CatalogSourcesResponse model."""

    def test_empty_response(self) -> None:
        """Test empty sources response."""
        response = CatalogSourcesResponse()

        assert response.sources == []

    def test_response_with_sources(self) -> None:
        """Test response with sources."""
        sources = [
            CatalogSource(name="rhoai", label="Red Hat AI validated", model_count=5),
            CatalogSource(name="community", label="Community", model_count=15),
        ]

        response = CatalogSourcesResponse(sources=sources)

        assert len(response.sources) == 2
        assert response.sources[0].name == "rhoai"
        assert response.sources[1].model_count == 15
