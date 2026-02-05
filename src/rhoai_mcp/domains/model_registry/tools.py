"""MCP Tools for Model Registry operations."""

import logging
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.domains.model_registry.benchmarks import BenchmarkExtractor
from rhoai_mcp.domains.model_registry.catalog_client import ModelCatalogClient
from rhoai_mcp.domains.model_registry.catalog_models import CatalogModel, CatalogSource
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
from rhoai_mcp.utils.response import (
    PaginatedResponse,
    Verbosity,
    paginate,
)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

# Cache for discovered API type to avoid probing on every call
_cached_api_type: str | None = None
_cached_discovery_url: str | None = None


def _create_cached_catalog_discovery(url: str) -> DiscoveredModelRegistry:
    """Create a cached discovery result for Model Catalog API.

    Used when we already know the API type from probing and need to
    create a discovery result for the ModelCatalogClient.

    Args:
        url: The discovered Model Catalog URL.

    Returns:
        A DiscoveredModelRegistry configured for Model Catalog.
    """
    return DiscoveredModelRegistry(
        url=url,
        namespace="unknown",
        service_name="model-catalog",
        port=443,
        source="cached",
        requires_auth=True,
        api_type="model_catalog",
    )


def _create_cached_registry_discovery(url: str) -> DiscoveredModelRegistry:
    """Create a cached discovery result for standard Model Registry API.

    Used when we already know the API type from probing and need to
    create a discovery result for the ModelRegistryClient.

    Args:
        url: The discovered Model Registry URL.

    Returns:
        A DiscoveredModelRegistry configured for standard Model Registry.
    """
    return DiscoveredModelRegistry(
        url=url,
        namespace="unknown",
        service_name="model-registry",
        port=443,
        source="cached",
        requires_auth=False,  # Port-forward bypasses auth
        api_type="model_registry",
    )


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register Model Registry tools with the MCP server."""

    async def _get_api_type() -> tuple[str, str]:
        """Get the detected API type and URL, with caching.

        Returns:
            Tuple of (api_type, url) where api_type is one of:
            "model_catalog", "model_registry", or "unknown".
        """
        global _cached_api_type, _cached_discovery_url

        # Use cached value if available
        if _cached_api_type is not None and _cached_discovery_url is not None:
            return _cached_api_type, _cached_discovery_url

        # Try discovery first
        from rhoai_mcp.config import ModelRegistryDiscoveryMode

        url = server.config.model_registry_url
        requires_auth = False

        if server.config.model_registry_discovery_mode == ModelRegistryDiscoveryMode.AUTO:
            discovery = ModelRegistryDiscovery(server.k8s)
            # Use async discovery with port-forward for outside-cluster access
            result = await discovery.discover_with_port_forward(fallback_url=url)
            if result:
                url = result.url
                requires_auth = result.requires_auth
                # If discovery already detected the API type, use it
                if result.api_type != "unknown":
                    _cached_api_type = result.api_type
                    _cached_discovery_url = url
                    return result.api_type, url

        # Probe the API type
        api_type = await probe_api_type(url, server.config, requires_auth)
        _cached_api_type = api_type
        _cached_discovery_url = url
        return api_type, url

    async def _require_model_registry(
        hint: str = "",
    ) -> tuple[dict[str, str] | None, DiscoveredModelRegistry | None]:
        """Require standard Model Registry, returning error if Model Catalog.

        This helper centralizes the API type check for tools that only work
        with the standard Kubeflow Model Registry (not Model Catalog).

        Args:
            hint: Additional context for the error message (e.g., "which does
                not support model versions.").

        Returns:
            Tuple of (error, discovery) where exactly one is None:
            - If cluster has Model Catalog: (error_dict, None)
            - If cluster has Model Registry: (None, discovery_result)
        """
        api_type, url = await _get_api_type()
        if api_type == "model_catalog":
            base_msg = (
                "This tool requires a standard Model Registry. "
                "The cluster has a Model Catalog"
            )
            if hint:
                base_msg += f" {hint}"
            else:
                base_msg += "."
            return {"error": base_msg}, None
        return None, _create_cached_registry_discovery(url)

    @mcp.tool()
    async def list_registered_models(
        limit: int | None = None,
        offset: int = 0,
        verbosity: str = "standard",
        source_label: str | None = None,
    ) -> dict[str, Any]:
        """List registered models in the Model Registry with pagination.

        The Model Registry stores metadata about ML models, including versions,
        artifacts, and custom properties. Use this to discover available models
        before deployment.

        This tool automatically detects whether the cluster has a standard
        Kubeflow Model Registry or Red Hat AI Model Catalog and uses the
        appropriate API.

        Args:
            limit: Maximum number of items to return (None for all).
            offset: Starting offset for pagination (default: 0).
            verbosity: Response detail level - "minimal", "standard", or "full".
                Use "minimal" for quick status checks.
            source_label: (Model Catalog only) Filter by source label,
                e.g., "Red Hat AI validated".

        Returns:
            Paginated list of registered models with metadata.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        try:
            # Detect which API is available
            api_type, url = await _get_api_type()

            all_items: list[dict[str, Any]] = []

            if api_type == "model_catalog":
                # Use Model Catalog client
                discovery_result = _create_cached_catalog_discovery(url)
                async with ModelCatalogClient(server.config, discovery_result) as catalog_client:
                    catalog_models = await catalog_client.list_models(source_label=source_label)
                    all_items = [
                        _format_catalog_model(m, Verbosity.from_str(verbosity))
                        for m in catalog_models
                    ]
            else:
                # Use standard Model Registry client
                discovery_result = _create_cached_registry_discovery(url)
                async with ModelRegistryClient(server.config, discovery_result) as registry_client:
                    registry_models = await registry_client.list_registered_models()
                    all_items = [
                        _format_model(m, Verbosity.from_str(verbosity)) for m in registry_models
                    ]

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

        result = PaginatedResponse.build(paginated, total, offset, effective_limit)
        result["api_type"] = api_type
        return result

    @mcp.tool()
    async def get_registered_model(
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

        try:
            error, discovery = await _require_model_registry(
                "which uses different APIs. Use list_registered_models to browse catalog models."
            )
            if error:
                return error
            async with ModelRegistryClient(server.config, discovery) as client:
                model = await client.get_registered_model(model_id)
                result = _format_model(model, Verbosity.FULL)

                if include_versions:
                    versions = await client.get_model_versions(model_id)
                    result["versions"] = [_format_version(v, Verbosity.STANDARD) for v in versions]

                return result
        except ModelNotFoundError:
            return {"error": f"Model not found: {model_id}"}
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    async def list_model_versions(
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

        try:
            error, discovery = await _require_model_registry(
                "which does not support model versions."
            )
            if error:
                return error
            async with ModelRegistryClient(server.config, discovery) as client:
                versions = await client.get_model_versions(model_id)
                all_items = [_format_version(v, Verbosity.from_str(verbosity)) for v in versions]
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
    async def get_model_version(version_id: str) -> dict[str, Any]:
        """Get detailed information about a specific model version.

        Args:
            version_id: The version ID.

        Returns:
            Version details including state, author, and custom properties.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        try:
            error, discovery = await _require_model_registry(
                "which does not support model versions."
            )
            if error:
                return error
            async with ModelRegistryClient(server.config, discovery) as client:
                version = await client.get_model_version(version_id)
                return _format_version(version, Verbosity.FULL)
        except ModelNotFoundError:
            return {"error": f"Version not found: {version_id}"}
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    async def get_model_artifacts(
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

        try:
            error, discovery = await _require_model_registry(
                "- use get_catalog_model_artifacts instead."
            )
            if error:
                return error
            async with ModelRegistryClient(server.config, discovery) as client:
                artifacts = await client.get_model_artifacts(version_id)
                formatted_artifacts = [
                    _format_artifact(a, Verbosity.from_str(verbosity)) for a in artifacts
                ]
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        return {
            "version_id": version_id,
            "artifacts": formatted_artifacts,
            "count": len(formatted_artifacts),
        }

    @mcp.tool()
    async def get_model_benchmarks(
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

        try:
            error, discovery = await _require_model_registry(
                "which does not store benchmark data in the same format."
            )
            if error:
                return error
            async with ModelRegistryClient(server.config, discovery) as client:
                extractor = BenchmarkExtractor(client)
                benchmark = await extractor.get_benchmark_for_model(
                    model_name=model_name,
                    version_name=version_name,
                    gpu_type=gpu_type,
                )
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        if benchmark is None:
            return {"error": f"No benchmark data found for model: {model_name}"}

        return _format_benchmark(benchmark)

    @mcp.tool()
    async def get_validation_metrics(
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

        try:
            error, discovery = await _require_model_registry(
                "which does not store validation metrics in the same format."
            )
            if error:
                return error
            async with ModelRegistryClient(server.config, discovery) as client:
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
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

    @mcp.tool()
    async def find_benchmarks_by_gpu(
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

        try:
            error, discovery = await _require_model_registry(
                "which does not store benchmark data in the same format."
            )
            if error:
                return error
            async with ModelRegistryClient(server.config, discovery) as client:
                extractor = BenchmarkExtractor(client)
                benchmarks = await extractor.find_benchmarks_by_gpu(gpu_type)
                formatted_benchmarks = [_format_benchmark(b) for b in benchmarks]
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        return {
            "gpu_type": gpu_type,
            "benchmarks": formatted_benchmarks,
            "count": len(formatted_benchmarks),
        }

    @mcp.tool()
    async def list_catalog_sources() -> dict[str, Any]:
        """List available sources in the Model Catalog.

        Sources are categories or providers of models in the Red Hat AI
        Model Catalog, such as 'Red Hat AI validated' or 'Community'.

        This tool only works when the cluster has a Model Catalog
        (not a standard Kubeflow Model Registry).

        Returns:
            List of sources with names, labels, and model counts.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        try:
            api_type, url = await _get_api_type()

            if api_type != "model_catalog":
                return {
                    "error": "This tool requires the Red Hat AI Model Catalog. "
                    f"The cluster appears to have a standard Model Registry (api_type={api_type})."
                }

            discovery_result = _create_cached_catalog_discovery(url)
            async with ModelCatalogClient(server.config, discovery_result) as client:
                sources = await client.get_sources()
                formatted_sources = [_format_catalog_source(s) for s in sources]

        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        return {
            "sources": formatted_sources,
            "count": len(formatted_sources),
        }

    @mcp.tool()
    async def get_catalog_model_artifacts(
        model_name: str,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Get artifacts (storage URIs) for a model in the Model Catalog.

        Artifacts contain the storage locations and format details for
        downloading or deploying a model from the catalog.

        This tool only works when the cluster has a Model Catalog
        (not a standard Kubeflow Model Registry).

        Args:
            model_name: Name of the model (e.g., 'RedHatAI/granite-3.1-8b-instruct').
            source_id: Optional source ID. If not provided, will be looked up
                from the model. Use list_catalog_sources to see available sources.

        Returns:
            List of artifacts with storage URIs and format information.
        """
        if not server.config.model_registry_enabled:
            return {"error": "Model Registry is disabled"}

        try:
            api_type, url = await _get_api_type()

            if api_type != "model_catalog":
                return {
                    "error": "This tool requires the Red Hat AI Model Catalog. "
                    f"The cluster appears to have a standard Model Registry (api_type={api_type})."
                }

            discovery_result = _create_cached_catalog_discovery(url)
            async with ModelCatalogClient(server.config, discovery_result) as client:
                # If source_id not provided, look it up from the model
                effective_source_id = source_id
                if not effective_source_id:
                    # Find the model to get its source_id
                    models = await client.list_models()
                    for model in models:
                        if model.name == model_name:
                            effective_source_id = model.source_id
                            break
                    if not effective_source_id:
                        return {"error": f"Model not found: {model_name}"}

                artifacts = await client.get_model_artifacts(effective_source_id, model_name)
                formatted_artifacts = [_format_catalog_artifact(a) for a in artifacts]

        except ModelNotFoundError:
            return {"error": f"Model not found: {model_name}"}
        except ModelRegistryConnectionError as e:
            return {"error": f"Connection failed: {e}"}
        except ModelRegistryError as e:
            return {"error": str(e)}

        return {
            "source_id": effective_source_id,
            "model_name": model_name,
            "artifacts": formatted_artifacts,
            "count": len(formatted_artifacts),
        }


def _format_catalog_model(model: CatalogModel, verbosity: Verbosity) -> dict[str, Any]:
    """Format a catalog model for response."""
    if verbosity == Verbosity.MINIMAL:
        return {
            "name": model.name,
            "provider": model.provider,
            "source_id": model.source_id,
        }

    result: dict[str, Any] = {
        "name": model.name,
        "description": model.description,
        "provider": model.provider,
        "source_id": model.source_id,
        "source_label": model.source_label,
        "task_type": model.task_type,
        "size": model.size,
        "license": model.license,
    }

    if verbosity == Verbosity.FULL:
        result["tags"] = model.tags
        result["long_description"] = model.long_description
        result["readme"] = model.readme
        if model.artifacts:
            result["artifacts"] = [_format_catalog_artifact(a) for a in model.artifacts]

    return result


def _format_catalog_source(source: CatalogSource) -> dict[str, Any]:
    """Format a catalog source for response."""
    return {
        "id": source.id,
        "name": source.name,
        "label": source.label,
        "model_count": source.model_count,
        "description": source.description,
    }


def _format_catalog_artifact(artifact: Any) -> dict[str, Any]:
    """Format a catalog artifact for response."""
    return {
        "uri": artifact.uri,
        "format": artifact.format,
        "size": artifact.size,
        "quantization": artifact.quantization,
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
