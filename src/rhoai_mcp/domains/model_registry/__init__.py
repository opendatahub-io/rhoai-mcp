"""Model Registry domain module.

This domain provides MCP tools for interacting with the OpenShift AI
Model Registry service. Unlike other domains that use Kubernetes CRDs,
the Model Registry uses a REST API.

Exports:
    Models:
        - RegisteredModel: Top-level model entity
        - ModelVersion: Version of a registered model
        - ModelArtifact: Storage artifact for a version
        - CustomProperties: Arbitrary key-value metadata
        - ValidationMetrics: Benchmark/validation metrics
        - MetricHistoryPoint: Single point in metric history
        - MetricHistory: Metric history from experiment run
        - BenchmarkData: Model benchmark data for capacity planning

    Client:
        - ModelRegistryClient: Async HTTP client for the registry API

    Benchmarks:
        - BenchmarkExtractor: Extracts benchmark data from model versions

    Discovery:
        - ModelRegistryDiscovery: Auto-discovers registry from cluster
        - DiscoveredModelRegistry: Discovery result dataclass

    Errors:
        - ModelRegistryError: Base exception
        - ModelNotFoundError: Model/version not found
        - ModelRegistryConnectionError: Connection failure

    Tools:
        - register_tools: Register MCP tools with server
"""

from rhoai_mcp.domains.model_registry.benchmarks import BenchmarkExtractor
from rhoai_mcp.domains.model_registry.client import ModelRegistryClient
from rhoai_mcp.domains.model_registry.discovery import (
    DiscoveredModelRegistry,
    ModelRegistryDiscovery,
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
    # Models
    "RegisteredModel",
    "ModelVersion",
    "ModelArtifact",
    "CustomProperties",
    "ValidationMetrics",
    "MetricHistoryPoint",
    "MetricHistory",
    "BenchmarkData",
    # Client
    "ModelRegistryClient",
    # Benchmarks
    "BenchmarkExtractor",
    # Discovery
    "ModelRegistryDiscovery",
    "DiscoveredModelRegistry",
    # Errors
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelRegistryConnectionError",
    # Tools
    "register_tools",
]
