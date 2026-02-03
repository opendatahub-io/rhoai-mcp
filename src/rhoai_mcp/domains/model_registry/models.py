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


class ValidationMetrics(BaseModel):
    """Validation/benchmark metrics from experiment runs.

    Stores performance and quality metrics captured during model
    evaluation or benchmark runs.
    """

    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    run_id: str | None = Field(None, description="Experiment run ID")

    # Latency metrics (milliseconds)
    p50_latency_ms: float | None = Field(None, description="50th percentile latency in ms")
    p95_latency_ms: float | None = Field(None, description="95th percentile latency in ms")
    p99_latency_ms: float | None = Field(None, description="99th percentile latency in ms")
    mean_latency_ms: float | None = Field(None, description="Mean latency in ms")

    # Throughput metrics
    tokens_per_second: float | None = Field(None, description="Token throughput")
    requests_per_second: float | None = Field(None, description="Request throughput")

    # Resource metrics
    gpu_memory_gb: float | None = Field(None, description="GPU memory usage in GB")
    gpu_utilization_percent: float | None = Field(None, description="GPU utilization percentage")
    peak_memory_gb: float | None = Field(None, description="Peak memory usage in GB")

    # Test conditions
    gpu_type: str | None = Field(None, description="GPU type (e.g., A100, H100)")
    gpu_count: int = Field(1, description="Number of GPUs used")
    input_tokens: int = Field(512, description="Input token count for benchmark")
    output_tokens: int = Field(256, description="Output token count for benchmark")
    batch_size: int = Field(1, description="Batch size used")
    concurrency: int = Field(1, description="Number of concurrent requests")
    tensor_parallel_size: int = Field(1, description="Tensor parallelism degree")

    # Quality metrics
    accuracy: float | None = Field(None, description="Model accuracy")
    perplexity: float | None = Field(None, description="Model perplexity")

    # Metadata
    benchmark_date: datetime | None = Field(None, description="When benchmark was run")
    notes: str | None = Field(None, description="Additional notes")


class MetricHistoryPoint(BaseModel):
    """Single point in metric history."""

    step: int = Field(..., description="Training step or epoch")
    timestamp: datetime | None = Field(None, description="When this point was recorded")
    value: float = Field(..., description="Metric value")


class MetricHistory(BaseModel):
    """Metric history from an experiment run.

    Tracks the evolution of a metric over training steps or time.
    """

    metric_name: str = Field(..., description="Name of the metric")
    run_id: str = Field(..., description="Experiment run ID")
    history: list[MetricHistoryPoint] = Field(default_factory=list, description="History points")

    def get_last_value(self) -> float | None:
        """Get the most recent metric value."""
        if not self.history:
            return None
        return max(self.history, key=lambda p: p.step).value

    def get_average(self) -> float | None:
        """Get the average metric value across all points."""
        if not self.history:
            return None
        return sum(p.value for p in self.history) / len(self.history)


class BenchmarkData(BaseModel):
    """Model benchmark data for capacity planning.

    Provides performance metrics useful for sizing and deployment decisions.
    """

    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    gpu_type: str = Field(..., description="GPU type (e.g., A100, H100)")
    gpu_count: int = Field(1, description="Number of GPUs used")

    # Latency (ms)
    p50_latency_ms: float = Field(0.0, description="50th percentile latency in ms")
    p95_latency_ms: float = Field(0.0, description="95th percentile latency in ms")
    p99_latency_ms: float = Field(0.0, description="99th percentile latency in ms")

    # Throughput
    tokens_per_second: float = Field(0.0, description="Token throughput")
    requests_per_second: float = Field(0.0, description="Request throughput")

    # Resources
    gpu_memory_gb: float = Field(0.0, description="GPU memory usage in GB")
    gpu_utilization_percent: float = Field(0.0, description="GPU utilization percentage")

    # Test conditions
    input_tokens: int = Field(512, description="Input token count for benchmark")
    output_tokens: int = Field(256, description="Output token count for benchmark")
    batch_size: int = Field(1, description="Batch size used")
    concurrency: int = Field(1, description="Number of concurrent requests")

    # Metadata
    benchmark_date: datetime | None = Field(None, description="When benchmark was run")
    source: str = Field("model_registry", description="Source of benchmark data")
