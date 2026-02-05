"""Model Registry domain module.

This domain provides MCP tools for interacting with the OpenShift AI
Model Registry service or Red Hat AI Model Catalog. Unlike other domains
that use Kubernetes CRDs, these services use REST APIs.

The module auto-detects whether the cluster has a standard Kubeflow
Model Registry or a Red Hat AI Model Catalog and uses the appropriate client.

Exports:
    Models (Model Registry):
        - RegisteredModel: Top-level model entity
        - ModelVersion: Version of a registered model
        - ModelArtifact: Storage artifact for a version
        - CustomProperties: Arbitrary key-value metadata
        - ValidationMetrics: Benchmark/validation metrics
        - MetricHistoryPoint: Single point in metric history
        - MetricHistory: Metric history from experiment run
        - BenchmarkData: Model benchmark data for capacity planning

    Models (Model Catalog):
        - CatalogModel: Model entry in the catalog
        - CatalogModelArtifact: Artifact information for catalog models
        - CatalogSource: Source/category in the catalog

    Clients:
        - ModelRegistryClient: Async HTTP client for Model Registry API
        - ModelCatalogClient: Async HTTP client for Model Catalog API

    Benchmarks:
        - BenchmarkExtractor: Extracts benchmark data from model versions

    Discovery:
        - ModelRegistryDiscovery: Auto-discovers registry from cluster
        - DiscoveredModelRegistry: Discovery result dataclass
        - probe_api_type: Async function to detect API type

    Errors:
        - ModelRegistryError: Base exception
        - ModelNotFoundError: Model/version not found
        - ModelRegistryConnectionError: Connection failure

    Tools:
        - register_tools: Register MCP tools with server
"""

from rhoai_mcp.domains.model_registry.benchmarks import BenchmarkExtractor
from rhoai_mcp.domains.model_registry.catalog_client import ModelCatalogClient
from rhoai_mcp.domains.model_registry.catalog_models import (
    CatalogModel,
    CatalogModelArtifact,
    CatalogSource,
)
from rhoai_mcp.domains.model_registry.client import ModelRegistryClient
from rhoai_mcp.domains.model_registry.discovery import (
    DiscoveredModelRegistry,
    ModelRegistryDiscovery,
    probe_api_type,
)
from rhoai_mcp.domains.model_registry.errors import (
    ModelNotFoundError,
    ModelRegistryConnectionError,
    ModelRegistryError,
)
from rhoai_mcp.domains.model_registry.models import (
    BenchmarkData,
    CustomProperties,
    MetricHistory,
    MetricHistoryPoint,
    ModelArtifact,
    ModelVersion,
    RegisteredModel,
    ValidationMetrics,
)
from rhoai_mcp.domains.model_registry.tools import register_tools

__all__ = [
    # Models (Model Registry)
    "RegisteredModel",
    "ModelVersion",
    "ModelArtifact",
    "CustomProperties",
    "ValidationMetrics",
    "MetricHistoryPoint",
    "MetricHistory",
    "BenchmarkData",
    # Models (Model Catalog)
    "CatalogModel",
    "CatalogModelArtifact",
    "CatalogSource",
    # Clients
    "ModelRegistryClient",
    "ModelCatalogClient",
    # Benchmarks
    "BenchmarkExtractor",
    # Discovery
    "ModelRegistryDiscovery",
    "DiscoveredModelRegistry",
    "probe_api_type",
    # Errors
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelRegistryConnectionError",
    # Tools
    "register_tools",
]
