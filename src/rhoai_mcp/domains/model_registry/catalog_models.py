"""Pydantic models for Red Hat AI Model Catalog entities.

The Model Catalog is a curated collection of validated AI models from
Red Hat, providing pre-tested models ready for deployment on OpenShift AI.
It uses a different API (/api/model_catalog/v1alpha1) than the standard
Kubeflow Model Registry (/api/model_registry/v1alpha3).
"""

from pydantic import BaseModel, Field


class CatalogModelArtifact(BaseModel):
    """Artifact information for a model in the catalog.

    Artifacts contain the storage location and format details for
    downloading or deploying a model from the catalog.
    """

    uri: str = Field(..., description="Storage URI for the model artifact")
    format: str | None = Field(None, description="Model format (e.g., safetensors, pytorch)")
    size: str | None = Field(None, description="Human-readable size (e.g., '7.5 GB')")
    quantization: str | None = Field(None, description="Quantization type if applicable")


class CatalogModel(BaseModel):
    """Model entry in the Red Hat AI Model Catalog.

    Represents a pre-validated model from the catalog with metadata
    about its source, task type, and available artifacts.
    """

    name: str = Field(..., description="Model name")
    description: str | None = Field(None, description="Model description")
    provider: str | None = Field(None, description="Model provider (e.g., 'Meta', 'Mistral AI')")
    source_id: str = Field("", description="Source ID for API calls (e.g., 'redhat_ai_validated_models')")
    source_label: str = Field("", description="Source label (e.g., 'Red Hat AI validated')")
    task_type: str | None = Field(None, description="Task type (e.g., 'text-generation')")
    tags: list[str] = Field(default_factory=list, description="Model tags")
    size: str | None = Field(None, description="Model size description")
    license: str | None = Field(None, description="Model license")
    artifacts: list[CatalogModelArtifact] = Field(
        default_factory=list, description="Available artifacts"
    )
    long_description: str | None = Field(None, description="Extended model description")
    readme: str | None = Field(None, description="README content if available")


class CatalogSource(BaseModel):
    """Source/category in the Model Catalog.

    Sources group models by validation status or provider, such as
    'Red Hat AI validated' or 'Community'.
    """

    id: str = Field(..., description="Source ID for API calls (e.g., 'redhat_ai_validated_models')")
    name: str = Field(..., description="Source name")
    label: str = Field(..., description="Source label/display name")
    model_count: int = Field(0, description="Number of models from this source")
    description: str | None = Field(None, description="Source description")


class CatalogListResponse(BaseModel):
    """Response from listing models in the catalog.

    Contains the list of models and pagination information.
    """

    models: list[CatalogModel] = Field(default_factory=list, description="List of catalog models")
    total_count: int = Field(0, description="Total number of models available")
    page_size: int = Field(50, description="Requested page size")
    next_page_token: str | None = Field(None, description="Token for next page if available")


class CatalogSourcesResponse(BaseModel):
    """Response from listing catalog sources.

    Contains the available sources/categories in the catalog.
    """

    sources: list[CatalogSource] = Field(default_factory=list, description="List of sources")
