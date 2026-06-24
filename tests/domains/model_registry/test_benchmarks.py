"""Tests for BenchmarkExtractor and CatalogBenchmarkExtractor."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from rhoai_mcp.domains.model_registry.benchmarks import (
    BENCHMARK_PROPERTY_KEYS,
    BENCHMARK_SECTION_PATTERNS,
    BenchmarkExtractor,
    CatalogBenchmarkExtractor,
    _get_property_value,
    _parse_datetime,
    _parse_float,
    _parse_int,
)
from rhoai_mcp.domains.model_registry.catalog_models import (
    CatalogBenchmarkContent,
    CatalogModel,
)
from rhoai_mcp.domains.model_registry.models import (
    CustomProperties,
    ModelVersion,
    RegisteredModel,
)


class TestPropertyValueHelpers:
    """Test property value helper functions."""

    def test_get_property_value_canonical_key(self) -> None:
        """Test getting value using canonical key."""
        props = {"p50_latency_ms": 45.0}
        assert _get_property_value(props, "p50_latency_ms") == 45.0

    def test_get_property_value_variant_key(self) -> None:
        """Test getting value using variant key."""
        props = {"latency_p95": 120.0}
        assert _get_property_value(props, "p95_latency_ms") == 120.0

    def test_get_property_value_missing(self) -> None:
        """Test getting missing value returns None."""
        props = {"other_key": 100}
        assert _get_property_value(props, "p50_latency_ms") is None

    def test_parse_float_valid(self) -> None:
        """Test parsing valid float values."""
        assert _parse_float(45.0) == pytest.approx(45.0)
        assert _parse_float("123.45") == pytest.approx(123.45)
        assert _parse_float(100) == pytest.approx(100.0)

    def test_parse_float_invalid(self) -> None:
        """Test parsing invalid float values."""
        assert _parse_float(None) == 0.0
        assert _parse_float("not-a-number") == 0.0
        assert _parse_float(None, 5.0) == 5.0

    def test_parse_int_valid(self) -> None:
        """Test parsing valid int values."""
        assert _parse_int(42) == 42
        assert _parse_int("100") == 100
        assert _parse_int(3.7) == 3

    def test_parse_int_invalid(self) -> None:
        """Test parsing invalid int values."""
        assert _parse_int(None) == 0
        assert _parse_int("not-a-number") == 0
        assert _parse_int(None, 10) == 10

    def test_parse_datetime_valid(self) -> None:
        """Test parsing valid datetime values."""
        result = _parse_datetime("2024-06-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_parse_datetime_already_datetime(self) -> None:
        """Test parsing already-datetime value."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert _parse_datetime(dt) == dt

    def test_parse_datetime_invalid(self) -> None:
        """Test parsing invalid datetime values."""
        assert _parse_datetime(None) is None
        assert _parse_datetime("not-a-date") is None


class TestBenchmarkPropertyKeys:
    """Test benchmark property key mappings."""

    def test_latency_key_variants(self) -> None:
        """Test that latency keys have variants."""
        assert "p50_latency_ms" in BENCHMARK_PROPERTY_KEYS
        assert "latency_p50" in BENCHMARK_PROPERTY_KEYS["p50_latency_ms"]

    def test_throughput_key_variants(self) -> None:
        """Test that throughput keys have variants."""
        assert "tokens_per_second" in BENCHMARK_PROPERTY_KEYS
        assert "tps" in BENCHMARK_PROPERTY_KEYS["tokens_per_second"]

    def test_gpu_type_variants(self) -> None:
        """Test that GPU type has variants."""
        assert "gpu_type" in BENCHMARK_PROPERTY_KEYS
        assert "accelerator_type" in BENCHMARK_PROPERTY_KEYS["gpu_type"]


class TestBenchmarkExtractor:
    """Test BenchmarkExtractor class."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock ModelRegistryClient."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def sample_version_with_benchmarks(self) -> ModelVersion:
        """Create a sample version with benchmark properties."""
        return ModelVersion(
            id="version-1",
            name="v1.0.0",
            registered_model_id="model-1",
            custom_properties=CustomProperties(
                properties={
                    "p50_latency_ms": "45.0",
                    "p95_latency_ms": "120.0",
                    "p99_latency_ms": "250.0",
                    "tokens_per_second": "1500.0",
                    "requests_per_second": "25.0",
                    "gpu_memory_gb": "24.5",
                    "gpu_utilization_percent": "85.0",
                    "gpu_type": "A100",
                    "gpu_count": "2",
                    "input_tokens": "1024",
                    "output_tokens": "512",
                    "batch_size": "4",
                    "concurrency": "8",
                    "benchmark_date": "2024-06-15T10:30:00Z",
                }
            ),
            create_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def sample_version_without_benchmarks(self) -> ModelVersion:
        """Create a sample version without benchmark properties."""
        return ModelVersion(
            id="version-2",
            name="v2.0.0",
            registered_model_id="model-1",
            custom_properties=CustomProperties(properties={"framework": "pytorch"}),
            create_time=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def sample_model(self) -> RegisteredModel:
        """Create a sample registered model."""
        return RegisteredModel(
            id="model-1",
            name="llama-2-7b",
            state="LIVE",
        )

    async def test_get_benchmark_for_model(
        self,
        mock_client: AsyncMock,
        sample_model: RegisteredModel,
        sample_version_with_benchmarks: ModelVersion,
    ) -> None:
        """Test getting benchmark for a model."""
        mock_client.get_registered_model_by_name.return_value = sample_model
        mock_client.get_model_versions.return_value = [sample_version_with_benchmarks]

        extractor = BenchmarkExtractor(mock_client)
        benchmark = await extractor.get_benchmark_for_model("llama-2-7b")

        assert benchmark is not None
        assert benchmark.model_name == "llama-2-7b"
        assert benchmark.model_version == "v1.0.0"
        assert benchmark.gpu_type == "A100"
        assert benchmark.p50_latency_ms == pytest.approx(45.0)
        assert benchmark.tokens_per_second == pytest.approx(1500.0)

    async def test_get_benchmark_model_not_found(
        self,
        mock_client: AsyncMock,
    ) -> None:
        """Test getting benchmark when model not found."""
        mock_client.get_registered_model_by_name.return_value = None

        extractor = BenchmarkExtractor(mock_client)
        benchmark = await extractor.get_benchmark_for_model("nonexistent")

        assert benchmark is None

    async def test_get_benchmark_no_versions(
        self,
        mock_client: AsyncMock,
        sample_model: RegisteredModel,
    ) -> None:
        """Test getting benchmark when model has no versions."""
        mock_client.get_registered_model_by_name.return_value = sample_model
        mock_client.get_model_versions.return_value = []

        extractor = BenchmarkExtractor(mock_client)
        benchmark = await extractor.get_benchmark_for_model("llama-2-7b")

        assert benchmark is None

    async def test_get_benchmark_with_version_filter(
        self,
        mock_client: AsyncMock,
        sample_model: RegisteredModel,
        sample_version_with_benchmarks: ModelVersion,
    ) -> None:
        """Test getting benchmark with specific version."""
        v2 = ModelVersion(
            id="v2",
            name="v2.0.0",
            registered_model_id="model-1",
            custom_properties=CustomProperties(
                properties={"p50_latency_ms": "30.0", "gpu_type": "H100"}
            ),
            create_time=datetime(2024, 7, 1, tzinfo=timezone.utc),
        )
        mock_client.get_registered_model_by_name.return_value = sample_model
        mock_client.get_model_versions.return_value = [sample_version_with_benchmarks, v2]

        extractor = BenchmarkExtractor(mock_client)
        benchmark = await extractor.get_benchmark_for_model("llama-2-7b", version_name="v2.0.0")

        assert benchmark is not None
        assert benchmark.model_version == "v2.0.0"
        assert benchmark.gpu_type == "H100"

    async def test_get_benchmark_with_gpu_filter(
        self,
        mock_client: AsyncMock,
        sample_model: RegisteredModel,
        sample_version_with_benchmarks: ModelVersion,
    ) -> None:
        """Test getting benchmark with GPU type filter."""
        mock_client.get_registered_model_by_name.return_value = sample_model
        mock_client.get_model_versions.return_value = [sample_version_with_benchmarks]

        extractor = BenchmarkExtractor(mock_client)

        # Matching GPU type
        benchmark = await extractor.get_benchmark_for_model("llama-2-7b", gpu_type="A100")
        assert benchmark is not None
        assert benchmark.gpu_type == "A100"

        # Non-matching GPU type
        benchmark = await extractor.get_benchmark_for_model("llama-2-7b", gpu_type="H100")
        assert benchmark is None

    async def test_get_all_benchmarks_for_model(
        self,
        mock_client: AsyncMock,
        sample_model: RegisteredModel,
        sample_version_with_benchmarks: ModelVersion,
        sample_version_without_benchmarks: ModelVersion,
    ) -> None:
        """Test getting all benchmarks for a model."""
        mock_client.get_registered_model_by_name.return_value = sample_model
        mock_client.get_model_versions.return_value = [
            sample_version_with_benchmarks,
            sample_version_without_benchmarks,
        ]

        extractor = BenchmarkExtractor(mock_client)
        benchmarks = await extractor.get_all_benchmarks_for_model("llama-2-7b")

        # Only version with benchmark data should be included
        assert len(benchmarks) == 1
        assert benchmarks[0].model_version == "v1.0.0"

    async def test_get_all_benchmarks_model_not_found(
        self,
        mock_client: AsyncMock,
    ) -> None:
        """Test getting all benchmarks when model not found."""
        mock_client.get_registered_model_by_name.return_value = None

        extractor = BenchmarkExtractor(mock_client)
        benchmarks = await extractor.get_all_benchmarks_for_model("nonexistent")

        assert benchmarks == []

    async def test_find_benchmarks_by_gpu(
        self,
        mock_client: AsyncMock,
    ) -> None:
        """Test finding benchmarks by GPU type."""
        model1 = RegisteredModel(id="m1", name="model-1", state="LIVE")
        model2 = RegisteredModel(id="m2", name="model-2", state="LIVE")

        v1_a100 = ModelVersion(
            id="v1",
            name="v1",
            registered_model_id="m1",
            custom_properties=CustomProperties(
                properties={"gpu_type": "A100", "p50_latency_ms": "50.0"}
            ),
        )
        v2_h100 = ModelVersion(
            id="v2",
            name="v1",
            registered_model_id="m2",
            custom_properties=CustomProperties(
                properties={"gpu_type": "H100", "p50_latency_ms": "30.0"}
            ),
        )

        mock_client.list_registered_models.return_value = [model1, model2]
        mock_client.get_model_versions.side_effect = [[v1_a100], [v2_h100]]

        extractor = BenchmarkExtractor(mock_client)
        benchmarks = await extractor.find_benchmarks_by_gpu("A100")

        assert len(benchmarks) == 1
        assert benchmarks[0].model_name == "model-1"
        assert benchmarks[0].gpu_type == "A100"

    async def test_find_benchmarks_by_gpu_no_matches(
        self,
        mock_client: AsyncMock,
    ) -> None:
        """Test finding benchmarks when no matches for GPU type."""
        model = RegisteredModel(id="m1", name="model-1", state="LIVE")
        version = ModelVersion(
            id="v1",
            name="v1",
            registered_model_id="m1",
            custom_properties=CustomProperties(
                properties={"gpu_type": "A100", "p50_latency_ms": "50.0"}
            ),
        )

        mock_client.list_registered_models.return_value = [model]
        mock_client.get_model_versions.return_value = [version]

        extractor = BenchmarkExtractor(mock_client)
        benchmarks = await extractor.find_benchmarks_by_gpu("TPU")

        assert len(benchmarks) == 0

    def test_extract_validation_metrics(
        self,
        mock_client: AsyncMock,
        sample_version_with_benchmarks: ModelVersion,
    ) -> None:
        """Test extracting validation metrics from version."""
        extractor = BenchmarkExtractor(mock_client)
        metrics = extractor.extract_validation_metrics(
            sample_version_with_benchmarks, "llama-2-7b"
        )

        assert metrics.model_name == "llama-2-7b"
        assert metrics.model_version == "v1.0.0"
        assert metrics.p50_latency_ms == pytest.approx(45.0)
        assert metrics.p95_latency_ms == pytest.approx(120.0)
        assert metrics.tokens_per_second == pytest.approx(1500.0)
        assert metrics.gpu_type == "A100"
        assert metrics.gpu_count == 2
        assert metrics.input_tokens == 1024
        assert metrics.output_tokens == 512
        assert metrics.batch_size == 4
        assert metrics.concurrency == 8
        assert metrics.benchmark_date is not None

    def test_extract_validation_metrics_minimal(
        self,
        mock_client: AsyncMock,
        sample_version_without_benchmarks: ModelVersion,
    ) -> None:
        """Test extracting validation metrics from version with no benchmark data."""
        extractor = BenchmarkExtractor(mock_client)
        metrics = extractor.extract_validation_metrics(
            sample_version_without_benchmarks, "test-model"
        )

        assert metrics.model_name == "test-model"
        assert metrics.model_version == "v2.0.0"
        assert metrics.p50_latency_ms is None
        assert metrics.tokens_per_second is None
        assert metrics.gpu_type is None
        assert metrics.gpu_count == 1  # Default


SAMPLE_README_WITH_BENCHMARKS = """# Granite 3.1 8B Instruct

A fine-tuned large language model for instruction following.

## Model Details

This model is based on the Granite architecture with 8B parameters.

## Evaluation Results

| Benchmark | Score |
|-----------|-------|
| MMLU      | 72.3  |
| HellaSwag | 85.1  |
| ARC-C     | 65.4  |

## Performance

| GPU Type | Throughput (tokens/s) | Latency P50 (ms) |
|----------|----------------------|-------------------|
| A100     | 1500                 | 45                |
| H100     | 2800                 | 25                |

## License

Apache 2.0
"""

SAMPLE_README_NO_BENCHMARKS = """# Simple Model

## Overview

This is a simple model without any benchmark data.

## Usage

Just load and run the model.

## License

MIT
"""


class TestCatalogBenchmarkExtractor:
    """Test CatalogBenchmarkExtractor class."""

    @pytest.fixture
    def extractor(self) -> CatalogBenchmarkExtractor:
        """Create a CatalogBenchmarkExtractor instance."""
        return CatalogBenchmarkExtractor()

    def test_extract_sections_with_benchmark_heading(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test extracting sections with benchmark-related headings."""
        sections = extractor.extract_benchmark_sections(SAMPLE_README_WITH_BENCHMARKS)

        assert len(sections) >= 1
        headings = [s["heading"] for s in sections]
        assert any("Evaluation" in h for h in headings)

    def test_extract_sections_multiple_matches(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test extracting multiple matching sections."""
        sections = extractor.extract_benchmark_sections(SAMPLE_README_WITH_BENCHMARKS)

        headings = [s["heading"] for s in sections]
        # Both "Evaluation Results" and "Performance" should match
        assert any("Evaluation" in h for h in headings)
        assert any("Performance" in h for h in headings)
        assert len(sections) == 2

    def test_extract_sections_no_matches(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test extracting sections with no benchmark headings."""
        sections = extractor.extract_benchmark_sections(SAMPLE_README_NO_BENCHMARKS)

        assert sections == []

    def test_extract_sections_empty_readme(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test extracting sections from empty/None input."""
        assert extractor.extract_benchmark_sections("") == []
        assert extractor.extract_benchmark_sections(None) == []

    def test_extract_sections_case_insensitive(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test that heading matching is case-insensitive."""
        readme = "# Model\n\n## EVALUATION\n\nSome eval data.\n\n## Details\n\nOther stuff.\n"
        sections = extractor.extract_benchmark_sections(readme)

        assert len(sections) == 1
        assert sections[0]["heading"] == "EVALUATION"
        assert "eval data" in sections[0]["content"]

    def test_extract_for_model_with_readme(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test full model extraction with a readme."""
        model = CatalogModel(
            name="granite-3.1-8b-instruct",
            provider="IBM",
            readme=SAMPLE_README_WITH_BENCHMARKS,
        )
        result = extractor.extract_for_model(model)

        assert isinstance(result, CatalogBenchmarkContent)
        assert result.model_name == "granite-3.1-8b-instruct"
        assert result.provider == "IBM"
        assert result.source == "model_catalog"
        assert result.has_benchmark_content is True
        assert len(result.sections) >= 2

    def test_extract_for_model_no_readme(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test extraction when model has no readme."""
        model = CatalogModel(
            name="test-model",
            provider="Test",
            readme=None,
        )
        result = extractor.extract_for_model(model)

        assert result.model_name == "test-model"
        assert result.has_benchmark_content is False
        assert result.sections == []

    def test_readme_mentions_gpu_present(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test GPU mention detection when GPU is present."""
        assert extractor.readme_mentions_gpu(SAMPLE_README_WITH_BENCHMARKS, "A100") is True
        assert extractor.readme_mentions_gpu(SAMPLE_README_WITH_BENCHMARKS, "H100") is True

    def test_readme_mentions_gpu_absent(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test GPU mention detection when GPU is absent."""
        assert extractor.readme_mentions_gpu(SAMPLE_README_WITH_BENCHMARKS, "TPU") is False
        assert extractor.readme_mentions_gpu(SAMPLE_README_WITH_BENCHMARKS, "L40S") is False

    def test_readme_mentions_gpu_case_insensitive(
        self, extractor: CatalogBenchmarkExtractor
    ) -> None:
        """Test GPU mention detection is case-insensitive."""
        readme = "Tested on a100 GPU with great results."
        assert extractor.readme_mentions_gpu(readme, "A100") is True
        assert extractor.readme_mentions_gpu(readme, "a100") is True
