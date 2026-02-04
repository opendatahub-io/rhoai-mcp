"""Tests for Model Registry MCP tools."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rhoai_mcp.domains.model_registry.models import (
    CustomProperties,
    ModelArtifact,
    ModelVersion,
    RegisteredModel,
)


class TestListRegisteredModels:
    """Test list_registered_models tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        server.config.default_list_limit = None
        server.config.max_list_limit = 100
        return server

    async def test_list_models_disabled(self, mock_server: MagicMock) -> None:
        """Test listing when registry is disabled."""
        mock_server.config.model_registry_enabled = False

        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        result = await registered_tools["list_registered_models"]()
        assert "error" in result
        assert "disabled" in result["error"]

    async def test_list_models_success(self, mock_server: MagicMock) -> None:
        """Test successful model listing."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        models = [
            RegisteredModel(id="model-1", name="model-one", state="LIVE"),
            RegisteredModel(id="model-2", name="model-two", state="LIVE"),
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_registered_models = AsyncMock(return_value=models)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["list_registered_models"]()

        assert "items" in result
        assert result["total"] == 2
        assert len(result["items"]) == 2

    async def test_list_models_with_pagination(self, mock_server: MagicMock) -> None:
        """Test model listing with pagination."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        models = [
            RegisteredModel(id=f"model-{i}", name=f"model-{i}", state="LIVE")
            for i in range(10)
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_registered_models = AsyncMock(return_value=models)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["list_registered_models"](limit=5, offset=2)

        assert result["total"] == 10
        assert len(result["items"]) == 5
        assert result["offset"] == 2


class TestGetRegisteredModel:
    """Test get_registered_model tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        return server

    async def test_get_model_success(self, mock_server: MagicMock) -> None:
        """Test getting a model successfully."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        model = RegisteredModel(
            id="model-123",
            name="llama-2-7b",
            description="Fine-tuned model",
            owner="data-team",
            state="LIVE",
        )

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model = AsyncMock(return_value=model)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["get_registered_model"]("model-123")

        assert result["id"] == "model-123"
        assert result["name"] == "llama-2-7b"
        assert result["owner"] == "data-team"

    async def test_get_model_with_versions(self, mock_server: MagicMock) -> None:
        """Test getting a model with versions."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        model = RegisteredModel(id="model-123", name="test-model", state="LIVE")
        versions = [
            ModelVersion(id="v1", name="1.0", registered_model_id="model-123"),
            ModelVersion(id="v2", name="2.0", registered_model_id="model-123"),
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model = AsyncMock(return_value=model)
            mock_client.get_model_versions = AsyncMock(return_value=versions)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["get_registered_model"](
                "model-123", include_versions=True
            )

        assert "versions" in result
        assert len(result["versions"]) == 2

    async def test_get_model_not_found(self, mock_server: MagicMock) -> None:
        """Test getting a model that doesn't exist."""
        from rhoai_mcp.domains.model_registry.errors import ModelNotFoundError
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model = AsyncMock(
                side_effect=ModelNotFoundError("Not found")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["get_registered_model"]("nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()


class TestGetModelArtifacts:
    """Test get_model_artifacts tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        return server

    async def test_get_artifacts_success(self, mock_server: MagicMock) -> None:
        """Test getting artifacts successfully."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        artifacts = [
            ModelArtifact(
                id="artifact-1",
                name="weights",
                uri="s3://bucket/model/weights.bin",
                model_format_name="pytorch",
            ),
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_model_artifacts = AsyncMock(return_value=artifacts)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["get_model_artifacts"]("version-123")

        assert result["version_id"] == "version-123"
        assert result["count"] == 1
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["uri"] == "s3://bucket/model/weights.bin"


class TestGetModelBenchmarks:
    """Test get_model_benchmarks tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        return server

    async def test_get_benchmarks_disabled(self, mock_server: MagicMock) -> None:
        """Test getting benchmarks when registry is disabled."""
        mock_server.config.model_registry_enabled = False

        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        result = await registered_tools["get_model_benchmarks"]("llama-2-7b")
        assert "error" in result
        assert "disabled" in result["error"]

    async def test_get_benchmarks_success(self, mock_server: MagicMock) -> None:
        """Test getting benchmarks successfully."""
        from rhoai_mcp.domains.model_registry.models import BenchmarkData
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        benchmark = BenchmarkData(
            model_name="llama-2-7b",
            model_version="v1.0",
            gpu_type="A100",
            p50_latency_ms=45.0,
            tokens_per_second=1500.0,
        )

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class, patch(
            "rhoai_mcp.domains.model_registry.tools.BenchmarkExtractor"
        ) as mock_extractor_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_extractor = AsyncMock()
            mock_extractor.get_benchmark_for_model = AsyncMock(return_value=benchmark)
            mock_extractor_class.return_value = mock_extractor

            result = await registered_tools["get_model_benchmarks"]("llama-2-7b")

        assert result["model_name"] == "llama-2-7b"
        assert result["gpu_type"] == "A100"
        assert result["p50_latency_ms"] == 45.0

    async def test_get_benchmarks_not_found(self, mock_server: MagicMock) -> None:
        """Test getting benchmarks when no data found."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class, patch(
            "rhoai_mcp.domains.model_registry.tools.BenchmarkExtractor"
        ) as mock_extractor_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_extractor = AsyncMock()
            mock_extractor.get_benchmark_for_model = AsyncMock(return_value=None)
            mock_extractor_class.return_value = mock_extractor

            result = await registered_tools["get_model_benchmarks"]("nonexistent")

        assert "error" in result
        assert "No benchmark data found" in result["error"]


class TestGetValidationMetrics:
    """Test get_validation_metrics tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        return server

    async def test_get_validation_metrics_disabled(self, mock_server: MagicMock) -> None:
        """Test getting validation metrics when registry is disabled."""
        mock_server.config.model_registry_enabled = False

        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        result = await registered_tools["get_validation_metrics"]("llama-2-7b", "v1.0")
        assert "error" in result
        assert "disabled" in result["error"]

    async def test_get_validation_metrics_model_not_found(self, mock_server: MagicMock) -> None:
        """Test getting validation metrics when model not found."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model_by_name = AsyncMock(return_value=None)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["get_validation_metrics"]("nonexistent", "v1.0")

        assert "error" in result
        assert "Model not found" in result["error"]

    async def test_get_validation_metrics_version_not_found(self, mock_server: MagicMock) -> None:
        """Test getting validation metrics when version not found."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        model = RegisteredModel(id="model-1", name="llama-2-7b", state="LIVE")
        version = ModelVersion(id="v1", name="v1.0", registered_model_id="model-1")

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model_by_name = AsyncMock(return_value=model)
            mock_client.get_model_versions = AsyncMock(return_value=[version])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await registered_tools["get_validation_metrics"]("llama-2-7b", "v2.0")

        assert "error" in result
        assert "Version not found" in result["error"]

    async def test_get_validation_metrics_success(self, mock_server: MagicMock) -> None:
        """Test getting validation metrics successfully."""
        from rhoai_mcp.domains.model_registry.models import ValidationMetrics
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        model = RegisteredModel(id="model-1", name="llama-2-7b", state="LIVE")
        version = ModelVersion(
            id="v1",
            name="v1.0",
            registered_model_id="model-1",
            custom_properties=CustomProperties(
                properties={"p50_latency_ms": "45.0", "gpu_type": "A100"}
            ),
        )

        metrics = ValidationMetrics(
            model_name="llama-2-7b",
            model_version="v1.0",
            p50_latency_ms=45.0,
            gpu_type="A100",
        )

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class, patch(
            "rhoai_mcp.domains.model_registry.tools.BenchmarkExtractor"
        ) as mock_extractor_class:
            mock_client = AsyncMock()
            mock_client.get_registered_model_by_name = AsyncMock(return_value=model)
            mock_client.get_model_versions = AsyncMock(return_value=[version])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_extractor = MagicMock()
            mock_extractor.extract_validation_metrics.return_value = metrics
            mock_extractor_class.return_value = mock_extractor

            result = await registered_tools["get_validation_metrics"]("llama-2-7b", "v1.0")

        assert result["model_name"] == "llama-2-7b"
        assert result["model_version"] == "v1.0"


class TestFindBenchmarksByGpu:
    """Test find_benchmarks_by_gpu tool."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock server."""
        server = MagicMock()
        server.config.model_registry_enabled = True
        server.config.model_registry_url = "http://registry:8080"
        server.config.model_registry_timeout = 30
        return server

    async def test_find_benchmarks_disabled(self, mock_server: MagicMock) -> None:
        """Test finding benchmarks when registry is disabled."""
        mock_server.config.model_registry_enabled = False

        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        result = await registered_tools["find_benchmarks_by_gpu"]("A100")
        assert "error" in result
        assert "disabled" in result["error"]

    async def test_find_benchmarks_success(self, mock_server: MagicMock) -> None:
        """Test finding benchmarks successfully."""
        from rhoai_mcp.domains.model_registry.models import BenchmarkData
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        benchmarks = [
            BenchmarkData(
                model_name="llama-2-7b",
                model_version="v1.0",
                gpu_type="A100",
                p50_latency_ms=45.0,
            ),
            BenchmarkData(
                model_name="mistral-7b",
                model_version="v2.0",
                gpu_type="A100",
                p50_latency_ms=40.0,
            ),
        ]

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class, patch(
            "rhoai_mcp.domains.model_registry.tools.BenchmarkExtractor"
        ) as mock_extractor_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_extractor = AsyncMock()
            mock_extractor.find_benchmarks_by_gpu = AsyncMock(return_value=benchmarks)
            mock_extractor_class.return_value = mock_extractor

            result = await registered_tools["find_benchmarks_by_gpu"]("A100")

        assert result["gpu_type"] == "A100"
        assert result["count"] == 2
        assert len(result["benchmarks"]) == 2
        assert result["benchmarks"][0]["model_name"] == "llama-2-7b"
        assert result["benchmarks"][1]["model_name"] == "mistral-7b"

    async def test_find_benchmarks_empty(self, mock_server: MagicMock) -> None:
        """Test finding benchmarks with no results."""
        from rhoai_mcp.domains.model_registry.tools import register_tools

        mcp = MagicMock()
        registered_tools: dict[str, Any] = {}

        def capture_tool() -> Any:
            def decorator(func: Any) -> Any:
                registered_tools[func.__name__] = func
                return func

            return decorator

        mcp.tool = capture_tool
        register_tools(mcp, mock_server)

        with patch(
            "rhoai_mcp.domains.model_registry.tools.ModelRegistryClient"
        ) as mock_client_class, patch(
            "rhoai_mcp.domains.model_registry.tools.BenchmarkExtractor"
        ) as mock_extractor_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            mock_extractor = AsyncMock()
            mock_extractor.find_benchmarks_by_gpu = AsyncMock(return_value=[])
            mock_extractor_class.return_value = mock_extractor

            result = await registered_tools["find_benchmarks_by_gpu"]("TPU")

        assert result["gpu_type"] == "TPU"
        assert result["count"] == 0
        assert result["benchmarks"] == []
