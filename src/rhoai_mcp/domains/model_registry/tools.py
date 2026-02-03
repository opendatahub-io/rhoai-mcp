"""MCP Tools for Model Registry operations."""

import asyncio
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.model_registry.benchmarks import BenchmarkExtractor
from rhoai_mcp.domains.model_registry.client import ModelRegistryClient
from rhoai_mcp.domains.model_registry.errors import (
    ModelNotFoundError,
    ModelRegistryConnectionError,
    ModelRegistryError,
)
from rhoai_mcp.domains.model_registry.models import BenchmarkData
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    Verbosity,
    paginate,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register Model Registry tools with the MCP server."""

    @mcp.tool()
    def list_registered_models(
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List registered models in the Model Registry with pagination.

        The Model Registry stores metadata about ML models, including versions,
        artifacts, and custom properties. Use this to discover available models
        before deployment.

        Args:
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.

        Returns:
            Paginated list of registered models with metadata.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _list() -> list[dict[str, Any]]:
            async with ModelRegistryClient(server.config) as client:
                models = await client.list_registered_models()
                return [_format_model(m, Verbosity.from_str(verbosity)) for m in models]

        try:
            all_items = asyncio.run(_list())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(all_items, offset, effective_limit)

        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_registered_model(
        model_id: str,
        include_versions: bool = False,
    ) -> dict[str, Any]:
        """Get detailed information about a registered model.

        Args:
            model_id: The model ID in the registry.
            include_versions: If True, also fetch all versions for this model.

        Returns:
            Model details including name, description, owner, and optionally versions.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> dict[str, Any]:
            async with ModelRegistryClient(server.config) as client:
                model = await client.get_registered_model(model_id)
                result = _format_model(model, Verbosity.FULL)

                if include_versions:
                    versions = await client.get_model_versions(model_id)
                    result["versions"] = [_format_version(v, Verbosity.STANDARD) for v in versions]

                return result

        try:
            return asyncio.run(_get())
        except ModelNotFoundError:
            return {"error": f"Model not found: {model_id}"}
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    def list_model_versions(
        model_id: str,
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """List all versions of a registered model with pagination.

        Each version represents a specific iteration of the model with its own
        artifacts and metadata.

        Args:
            model_id: The parent model ID.
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            Paginated list of model versions.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _list() -> list[dict[str, Any]]:
            async with ModelRegistryClient(server.config) as client:
                versions = await client.get_model_versions(model_id)
                return [_format_version(v, Verbosity.from_str(verbosity)) for v in versions]

        try:
            all_items = asyncio.run(_list())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        # Apply config limits
        effective_limit = limit
        if effective_limit is not None:
            effective_limit = min(effective_limit, server.config.max_list_limit)
        elif server.config.default_list_limit is not None:
            effective_limit = server.config.default_list_limit

        # Paginate
        paginated, total = paginate(all_items, offset, effective_limit)

        return PaginatedResponse.build(paginated, total, offset, effective_limit)

    @mcp.tool()
    def get_model_version(version_id: str) -> dict[str, Any]:
        """Get detailed information about a specific model version.

        Args:
            version_id: The version ID.

        Returns:
            Version details including state, author, and custom properties.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> dict[str, Any]:
            async with ModelRegistryClient(server.config) as client:
                version = await client.get_model_version(version_id)
                return _format_version(version, Verbosity.FULL)

        try:
            return asyncio.run(_get())
        except ModelNotFoundError:
            return {"error": f"Version not found: {version_id}"}
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_model_artifacts(
        version_id: str,
        verbosity: str = "standard",
    ) -> dict[str, Any]:
        """Get artifacts (storage URIs) for a model version.

        Artifacts contain the actual model files stored in object storage.
        Use this to find the storage location for model deployment.

        Args:
            version_id: The model version ID.
            verbosity: Response detail level - "minimal", "standard", or "full".

        Returns:
            List of artifacts with storage URIs and format information.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> list[dict[str, Any]]:
            async with ModelRegistryClient(server.config) as client:
                artifacts = await client.get_model_artifacts(version_id)
                return [_format_artifact(a, Verbosity.from_str(verbosity)) for a in artifacts]

        try:
            artifacts = asyncio.run(_get())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        return {
            "version_id": version_id,
            "artifacts": artifacts,
            "count": len(artifacts),
        }

    @mcp.tool()
    def get_model_benchmarks(
        model_name: str,
        version_name: str | None = None,
        gpu_type: str | None = None,
    ) -> dict[str, Any]:
        """Get benchmark data for a model.

        Retrieves performance benchmark metrics stored in model version
        custom properties. Useful for capacity planning and deployment sizing.

        Args:
            model_name: Name of the registered model.
            version_name: Optional specific version name. If not provided,
                returns benchmarks for the latest version.
            gpu_type: Optional GPU type filter (e.g., "A100", "H100").

        Returns:
            Benchmark data including latency, throughput, and resource metrics.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> BenchmarkData | None:
            async with ModelRegistryClient(server.config) as client:
                extractor = BenchmarkExtractor(client)
                return await extractor.get_benchmark_for_model(
                    model_name=model_name,
                    version_name=version_name,
                    gpu_type=gpu_type,
                )

        try:
            benchmark = asyncio.run(_get())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        if benchmark is None:
            return {"error": f"No benchmark data found for model: {model_name}"}

        return _format_benchmark(benchmark)

    @mcp.tool()
    def get_validation_metrics(
        model_name: str,
        version_name: str,
    ) -> dict[str, Any]:
        """Get validation metrics for a specific model version.

        Retrieves detailed validation and benchmark metrics from model
        version custom properties, including latency percentiles,
        throughput, resource usage, and quality metrics.

        Args:
            model_name: Name of the registered model.
            version_name: Name of the model version.

        Returns:
            Validation metrics including latency, throughput, resources,
            test conditions, and quality metrics.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _get() -> dict[str, Any]:
            async with ModelRegistryClient(server.config) as client:
                # Find the model
                model = await client.get_registered_model_by_name(model_name)
                if not model:
                    return {"error": f"Model not found: {model_name}"}

                # Find the version
                versions = await client.get_model_versions(model.id)
                target_version = None
                for v in versions:
                    if v.name == version_name:
                        target_version = v
                        break

                if not target_version:
                    return {"error": f"Version not found: {version_name}"}

                extractor = BenchmarkExtractor(client)
                metrics = extractor.extract_validation_metrics(target_version, model_name)
                return _format_validation_metrics(metrics)

        try:
            return asyncio.run(_get())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    def find_benchmarks_by_gpu(
        gpu_type: str,
    ) -> dict[str, Any]:
        """Find all benchmarks for a specific GPU type.

        Searches across all registered models to find benchmark data
        for models that have been tested on the specified GPU type.
        Useful for comparing model performance on specific hardware.

        Args:
            gpu_type: GPU type to filter by (e.g., "A100", "H100", "L40S").

        Returns:
            List of benchmark data for models tested on the specified GPU.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        async def _find() -> list[dict[str, Any]]:
            async with ModelRegistryClient(server.config) as client:
                extractor = BenchmarkExtractor(client)
                benchmarks = await extractor.find_benchmarks_by_gpu(gpu_type)
                return [_format_benchmark(b) for b in benchmarks]

        try:
            benchmarks = asyncio.run(_find())
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        return {
            "gpu_type": gpu_type,
            "benchmarks": benchmarks,
            "count": len(benchmarks),
        }


def _format_model(model: Any, verbosity: Verbosity) -> dict[str, Any]:
    """Format a registered model for response."""
    if verbosity == Verbosity.MINIMAL:
        return {
            "id": model.id,
            "name": model.name,
            "state": model.state,
        }

    result: dict[str, Any] = {
        "id": model.id,
        "name": model.name,
        "state": model.state,
        "owner": model.owner,
        "description": model.description,
    }

    if verbosity == Verbosity.FULL:
        result["custom_properties"] = model.custom_properties.properties
        if model.create_time:
            result["create_time"] = model.create_time.isoformat()
        if model.update_time:
            result["update_time"] = model.update_time.isoformat()

    return result


def _format_version(version: Any, verbosity: Verbosity) -> dict[str, Any]:
    """Format a model version for response."""
    if verbosity == Verbosity.MINIMAL:
        return {
            "id": version.id,
            "name": version.name,
            "state": version.state,
        }

    result: dict[str, Any] = {
        "id": version.id,
        "name": version.name,
        "registered_model_id": version.registered_model_id,
        "state": version.state,
        "author": version.author,
        "description": version.description,
    }

    if verbosity == Verbosity.FULL:
        result["custom_properties"] = version.custom_properties.properties
        if version.create_time:
            result["create_time"] = version.create_time.isoformat()
        if version.update_time:
            result["update_time"] = version.update_time.isoformat()

    return result


def _format_artifact(artifact: Any, verbosity: Verbosity) -> dict[str, Any]:
    """Format a model artifact for response."""
    if verbosity == Verbosity.MINIMAL:
        return {
            "id": artifact.id,
            "name": artifact.name,
            "uri": artifact.uri,
        }

    result: dict[str, Any] = {
        "id": artifact.id,
        "name": artifact.name,
        "uri": artifact.uri,
        "model_format_name": artifact.model_format_name,
        "model_format_version": artifact.model_format_version,
    }

    if verbosity == Verbosity.FULL:
        result["description"] = artifact.description
        result["storage_key"] = artifact.storage_key
        result["storage_path"] = artifact.storage_path
        result["service_account_name"] = artifact.service_account_name
        result["custom_properties"] = artifact.custom_properties.properties
        if artifact.create_time:
            result["create_time"] = artifact.create_time.isoformat()
        if artifact.update_time:
            result["update_time"] = artifact.update_time.isoformat()

    return result


def _format_benchmark(benchmark: Any) -> dict[str, Any]:
    """Format benchmark data for response."""
    result: dict[str, Any] = {
        "model_name": benchmark.model_name,
        "model_version": benchmark.model_version,
        "gpu_type": benchmark.gpu_type,
        "gpu_count": benchmark.gpu_count,
        # Latency
        "p50_latency_ms": benchmark.p50_latency_ms,
        "p95_latency_ms": benchmark.p95_latency_ms,
        "p99_latency_ms": benchmark.p99_latency_ms,
        # Throughput
        "tokens_per_second": benchmark.tokens_per_second,
        "requests_per_second": benchmark.requests_per_second,
        # Resources
        "gpu_memory_gb": benchmark.gpu_memory_gb,
        "gpu_utilization_percent": benchmark.gpu_utilization_percent,
        # Test conditions
        "input_tokens": benchmark.input_tokens,
        "output_tokens": benchmark.output_tokens,
        "batch_size": benchmark.batch_size,
        "concurrency": benchmark.concurrency,
        # Metadata
        "source": benchmark.source,
    }

    if benchmark.benchmark_date:
        result["benchmark_date"] = benchmark.benchmark_date.isoformat()

    return result


def _format_validation_metrics(metrics: Any) -> dict[str, Any]:
    """Format validation metrics for response."""
    result: dict[str, Any] = {
        "model_name": metrics.model_name,
        "model_version": metrics.model_version,
    }

    # Optional fields - only include if set
    if metrics.run_id:
        result["run_id"] = metrics.run_id

    # Latency metrics
    latency: dict[str, float] = {}
    if metrics.p50_latency_ms is not None:
        latency["p50_ms"] = metrics.p50_latency_ms
    if metrics.p95_latency_ms is not None:
        latency["p95_ms"] = metrics.p95_latency_ms
    if metrics.p99_latency_ms is not None:
        latency["p99_ms"] = metrics.p99_latency_ms
    if metrics.mean_latency_ms is not None:
        latency["mean_ms"] = metrics.mean_latency_ms
    if latency:
        result["latency"] = latency

    # Throughput metrics
    throughput: dict[str, float] = {}
    if metrics.tokens_per_second is not None:
        throughput["tokens_per_second"] = metrics.tokens_per_second
    if metrics.requests_per_second is not None:
        throughput["requests_per_second"] = metrics.requests_per_second
    if throughput:
        result["throughput"] = throughput

    # Resource metrics
    resources: dict[str, float] = {}
    if metrics.gpu_memory_gb is not None:
        resources["gpu_memory_gb"] = metrics.gpu_memory_gb
    if metrics.gpu_utilization_percent is not None:
        resources["gpu_utilization_percent"] = metrics.gpu_utilization_percent
    if metrics.peak_memory_gb is not None:
        resources["peak_memory_gb"] = metrics.peak_memory_gb
    if resources:
        result["resources"] = resources

    # Test conditions
    test_conditions: dict[str, Any] = {}
    if metrics.gpu_type:
        test_conditions["gpu_type"] = metrics.gpu_type
    if metrics.gpu_count != 1:
        test_conditions["gpu_count"] = metrics.gpu_count
    if metrics.input_tokens != 512:
        test_conditions["input_tokens"] = metrics.input_tokens
    if metrics.output_tokens != 256:
        test_conditions["output_tokens"] = metrics.output_tokens
    if metrics.batch_size != 1:
        test_conditions["batch_size"] = metrics.batch_size
    if metrics.concurrency != 1:
        test_conditions["concurrency"] = metrics.concurrency
    if metrics.tensor_parallel_size != 1:
        test_conditions["tensor_parallel_size"] = metrics.tensor_parallel_size
    if test_conditions:
        result["test_conditions"] = test_conditions

    # Quality metrics
    quality: dict[str, float] = {}
    if metrics.accuracy is not None:
        quality["accuracy"] = metrics.accuracy
    if metrics.perplexity is not None:
        quality["perplexity"] = metrics.perplexity
    if quality:
        result["quality"] = quality

    # Metadata
    if metrics.benchmark_date:
        result["benchmark_date"] = metrics.benchmark_date.isoformat()
    if metrics.notes:
        result["notes"] = metrics.notes

    return result
