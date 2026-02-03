"""Benchmark data extraction from Model Registry."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.model_registry.models import (
    BenchmarkData,
    ModelVersion,
    ValidationMetrics,
)

if TYPE_CHECKING:
    from rhoai_mcp.domains.model_registry.client import ModelRegistryClient

# Mapping of canonical property names to possible key variants in custom properties
BENCHMARK_PROPERTY_KEYS: dict[str, list[str]] = {
    # Latency metrics
    "p50_latency_ms": ["p50_latency_ms", "latency_p50", "p50_latency", "p50"],
    "p95_latency_ms": ["p95_latency_ms", "latency_p95", "p95_latency", "p95"],
    "p99_latency_ms": ["p99_latency_ms", "latency_p99", "p99_latency", "p99"],
    "mean_latency_ms": ["mean_latency_ms", "latency_mean", "mean_latency", "avg_latency"],
    # Throughput metrics
    "tokens_per_second": ["tokens_per_second", "tps", "throughput_tps", "token_throughput"],
    "requests_per_second": ["requests_per_second", "rps", "throughput_rps", "qps"],
    # Resource metrics
    "gpu_memory_gb": ["gpu_memory_gb", "memory_gb", "vram_gb", "gpu_mem"],
    "gpu_utilization_percent": [
        "gpu_utilization_percent",
        "gpu_util",
        "gpu_utilization",
        "gpu_pct",
    ],
    "peak_memory_gb": ["peak_memory_gb", "peak_mem", "max_memory_gb"],
    # Test conditions
    "gpu_type": ["gpu_type", "gpu", "accelerator_type", "device_type"],
    "gpu_count": ["gpu_count", "num_gpus", "gpu_num", "accelerator_count"],
    "input_tokens": ["input_tokens", "input_len", "prompt_tokens", "input_length"],
    "output_tokens": ["output_tokens", "output_len", "completion_tokens", "output_length"],
    "batch_size": ["batch_size", "bs", "batch"],
    "concurrency": ["concurrency", "concurrent_requests", "num_concurrent"],
    "tensor_parallel_size": ["tensor_parallel_size", "tp", "tp_size", "tensor_parallel"],
    # Quality metrics
    "accuracy": ["accuracy", "acc", "eval_accuracy"],
    "perplexity": ["perplexity", "ppl", "eval_perplexity"],
    # Metadata
    "benchmark_date": ["benchmark_date", "benchmark_time", "eval_date", "test_date"],
    "notes": ["notes", "description", "benchmark_notes"],
}


def _get_property_value(
    properties: dict[str, Any],
    canonical_key: str,
) -> Any | None:
    """Get a property value by trying multiple key variants."""
    variants = BENCHMARK_PROPERTY_KEYS.get(canonical_key, [canonical_key])
    for key in variants:
        if key in properties:
            return properties[key]
    return None


def _parse_float(value: Any, default: float = 0.0) -> float:
    """Parse a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_int(value: Any, default: int = 0) -> int:
    """Parse a value to int."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _parse_datetime(value: Any) -> datetime | None:
    """Parse a value to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class BenchmarkExtractor:
    """Extracts benchmark data from Model Registry entities.

    This class reads custom properties from model versions and artifacts
    to extract benchmark metrics stored there.
    """

    def __init__(self, client: "ModelRegistryClient") -> None:
        """Initialize extractor.

        Args:
            client: Model Registry client for API access.
        """
        self._client = client

    async def get_benchmark_for_model(
        self,
        model_name: str,
        version_name: str | None = None,
        gpu_type: str | None = None,
    ) -> BenchmarkData | None:
        """Get benchmark data for a model.

        Args:
            model_name: Name of the model.
            version_name: Optional specific version name. If None, uses latest.
            gpu_type: Optional GPU type filter.

        Returns:
            BenchmarkData if found, None otherwise.
        """
        model = await self._client.get_registered_model_by_name(model_name)
        if not model:
            return None

        versions = await self._client.get_model_versions(model.id)
        if not versions:
            return None

        # Find the target version
        target_version: ModelVersion | None = None
        if version_name:
            for v in versions:
                if v.name == version_name:
                    target_version = v
                    break
        else:
            # Use the latest version by creation time
            target_version = max(
                versions,
                key=lambda v: v.create_time or datetime.min,
            )

        if not target_version:
            return None

        # Extract benchmark from version properties
        props = target_version.custom_properties.properties
        extracted_gpu_type = _get_property_value(props, "gpu_type")

        # If GPU type filter specified and doesn't match, return None
        if gpu_type and extracted_gpu_type and extracted_gpu_type != gpu_type:
            return None

        # If no GPU type in properties and filter specified, return None
        if gpu_type and not extracted_gpu_type:
            return None

        return self._extract_benchmark_data(
            model_name=model_name,
            version_name=target_version.name,
            properties=props,
        )

    async def get_all_benchmarks_for_model(self, model_name: str) -> list[BenchmarkData]:
        """Get all benchmark data for all versions of a model.

        Args:
            model_name: Name of the model.

        Returns:
            List of BenchmarkData for all versions with benchmark data.
        """
        model = await self._client.get_registered_model_by_name(model_name)
        if not model:
            return []

        versions = await self._client.get_model_versions(model.id)
        benchmarks: list[BenchmarkData] = []

        for version in versions:
            props = version.custom_properties.properties
            # Only include versions that have some benchmark data
            if self._has_benchmark_data(props):
                benchmark = self._extract_benchmark_data(
                    model_name=model_name,
                    version_name=version.name,
                    properties=props,
                )
                benchmarks.append(benchmark)

        return benchmarks

    async def find_benchmarks_by_gpu(self, gpu_type: str) -> list[BenchmarkData]:
        """Find all benchmarks for a specific GPU type.

        Args:
            gpu_type: GPU type to filter by (e.g., "A100", "H100").

        Returns:
            List of BenchmarkData for models benchmarked on the specified GPU.
        """
        benchmarks: list[BenchmarkData] = []
        models = await self._client.list_registered_models()

        for model in models:
            versions = await self._client.get_model_versions(model.id)
            for version in versions:
                props = version.custom_properties.properties
                version_gpu_type = _get_property_value(props, "gpu_type")

                if version_gpu_type == gpu_type and self._has_benchmark_data(props):
                    benchmark = self._extract_benchmark_data(
                        model_name=model.name,
                        version_name=version.name,
                        properties=props,
                    )
                    benchmarks.append(benchmark)

        return benchmarks

    def extract_validation_metrics(
        self,
        version: ModelVersion,
        model_name: str,
    ) -> ValidationMetrics:
        """Extract validation metrics from a model version.

        Args:
            version: The model version to extract from.
            model_name: Name of the parent model.

        Returns:
            ValidationMetrics populated from version custom properties.
        """
        props = version.custom_properties.properties

        return ValidationMetrics(
            model_name=model_name,
            model_version=version.name,
            run_id=props.get("run_id"),
            # Latency metrics
            p50_latency_ms=self._get_optional_float(props, "p50_latency_ms"),
            p95_latency_ms=self._get_optional_float(props, "p95_latency_ms"),
            p99_latency_ms=self._get_optional_float(props, "p99_latency_ms"),
            mean_latency_ms=self._get_optional_float(props, "mean_latency_ms"),
            # Throughput metrics
            tokens_per_second=self._get_optional_float(props, "tokens_per_second"),
            requests_per_second=self._get_optional_float(props, "requests_per_second"),
            # Resource metrics
            gpu_memory_gb=self._get_optional_float(props, "gpu_memory_gb"),
            gpu_utilization_percent=self._get_optional_float(props, "gpu_utilization_percent"),
            peak_memory_gb=self._get_optional_float(props, "peak_memory_gb"),
            # Test conditions
            gpu_type=_get_property_value(props, "gpu_type"),
            gpu_count=_parse_int(_get_property_value(props, "gpu_count"), 1),
            input_tokens=_parse_int(_get_property_value(props, "input_tokens"), 512),
            output_tokens=_parse_int(_get_property_value(props, "output_tokens"), 256),
            batch_size=_parse_int(_get_property_value(props, "batch_size"), 1),
            concurrency=_parse_int(_get_property_value(props, "concurrency"), 1),
            tensor_parallel_size=_parse_int(_get_property_value(props, "tensor_parallel_size"), 1),
            # Quality metrics
            accuracy=self._get_optional_float(props, "accuracy"),
            perplexity=self._get_optional_float(props, "perplexity"),
            # Metadata
            benchmark_date=_parse_datetime(_get_property_value(props, "benchmark_date")),
            notes=_get_property_value(props, "notes"),
        )

    def _has_benchmark_data(self, properties: dict[str, Any]) -> bool:
        """Check if properties contain any benchmark data."""
        benchmark_keys = [
            "p50_latency_ms",
            "p95_latency_ms",
            "tokens_per_second",
            "requests_per_second",
            "gpu_memory_gb",
            "accuracy",
        ]
        return any(_get_property_value(properties, key) is not None for key in benchmark_keys)

    def _extract_benchmark_data(
        self,
        model_name: str,
        version_name: str,
        properties: dict[str, Any],
    ) -> BenchmarkData:
        """Extract BenchmarkData from properties."""
        gpu_type = _get_property_value(properties, "gpu_type") or "unknown"

        return BenchmarkData(
            model_name=model_name,
            model_version=version_name,
            gpu_type=gpu_type,
            gpu_count=_parse_int(_get_property_value(properties, "gpu_count"), 1),
            # Latency
            p50_latency_ms=_parse_float(_get_property_value(properties, "p50_latency_ms")),
            p95_latency_ms=_parse_float(_get_property_value(properties, "p95_latency_ms")),
            p99_latency_ms=_parse_float(_get_property_value(properties, "p99_latency_ms")),
            # Throughput
            tokens_per_second=_parse_float(_get_property_value(properties, "tokens_per_second")),
            requests_per_second=_parse_float(
                _get_property_value(properties, "requests_per_second")
            ),
            # Resources
            gpu_memory_gb=_parse_float(_get_property_value(properties, "gpu_memory_gb")),
            gpu_utilization_percent=_parse_float(
                _get_property_value(properties, "gpu_utilization_percent")
            ),
            # Test conditions
            input_tokens=_parse_int(_get_property_value(properties, "input_tokens"), 512),
            output_tokens=_parse_int(_get_property_value(properties, "output_tokens"), 256),
            batch_size=_parse_int(_get_property_value(properties, "batch_size"), 1),
            concurrency=_parse_int(_get_property_value(properties, "concurrency"), 1),
            # Metadata
            benchmark_date=_parse_datetime(_get_property_value(properties, "benchmark_date")),
            source="model_registry",
        )

    def _get_optional_float(
        self,
        properties: dict[str, Any],
        key: str,
    ) -> float | None:
        """Get optional float value from properties."""
        value = _get_property_value(properties, key)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
