"""Tests for Model Registry models."""

import pytest

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


class TestCustomProperties:
    """Test CustomProperties model."""

    def test_get_property(self) -> None:
        """Test getting a property value."""
        props = CustomProperties(properties={"key": "value"})
        assert props.get("key") == "value"
        assert props.get("missing") is None
        assert props.get("missing", "default") == "default"

    def test_get_float(self) -> None:
        """Test getting a float property."""
        props = CustomProperties(properties={"accuracy": "0.95", "invalid": "not-a-number"})
        assert props.get_float("accuracy") == pytest.approx(0.95)
        assert props.get_float("invalid") == 0.0
        assert props.get_float("missing") == 0.0
        assert props.get_float("missing", 1.0) == 1.0

    def test_get_int(self) -> None:
        """Test getting an int property."""
        props = CustomProperties(properties={"count": "42", "invalid": "not-a-number"})
        assert props.get_int("count") == 42
        assert props.get_int("invalid") == 0
        assert props.get_int("missing") == 0
        assert props.get_int("missing", 10) == 10

    def test_empty_properties(self) -> None:
        """Test empty properties."""
        props = CustomProperties()
        assert props.properties == {}
        assert props.get("anything") is None


class TestModelArtifact:
    """Test ModelArtifact model."""

    def test_artifact_creation(self) -> None:
        """Test creating a model artifact."""
        artifact = ModelArtifact(
            id="artifact-123",
            name="model-weights",
            uri="s3://bucket/path/model.safetensors",
            model_format_name="safetensors",
        )

        assert artifact.id == "artifact-123"
        assert artifact.name == "model-weights"
        assert artifact.uri == "s3://bucket/path/model.safetensors"
        assert artifact.model_format_name == "safetensors"
        assert artifact.description is None
        assert artifact.custom_properties.properties == {}

    def test_artifact_with_all_fields(self) -> None:
        """Test artifact with all optional fields."""
        artifact = ModelArtifact(
            id="artifact-456",
            name="onnx-model",
            uri="s3://bucket/model.onnx",
            description="ONNX exported model",
            model_format_name="onnx",
            model_format_version="1.14",
            storage_key="models-bucket",
            storage_path="exported/model.onnx",
            service_account_name="model-deployer",
            custom_properties=CustomProperties(properties={"quantized": "true"}),
        )

        assert artifact.description == "ONNX exported model"
        assert artifact.storage_key == "models-bucket"
        assert artifact.custom_properties.get("quantized") == "true"


class TestModelVersion:
    """Test ModelVersion model."""

    def test_version_creation(self) -> None:
        """Test creating a model version."""
        version = ModelVersion(
            id="version-123",
            name="v1.0.0",
            registered_model_id="model-456",
        )

        assert version.id == "version-123"
        assert version.name == "v1.0.0"
        assert version.registered_model_id == "model-456"
        assert version.state == "LIVE"
        assert version.artifacts == []

    def test_version_with_artifacts(self) -> None:
        """Test version with artifacts."""
        artifact = ModelArtifact(
            id="artifact-1",
            name="weights",
            uri="s3://bucket/weights.bin",
        )
        version = ModelVersion(
            id="version-1",
            name="v2.0.0",
            registered_model_id="model-1",
            author="ml-team",
            artifacts=[artifact],
        )

        assert version.author == "ml-team"
        assert len(version.artifacts) == 1
        assert version.artifacts[0].name == "weights"


class TestRegisteredModel:
    """Test RegisteredModel model."""

    def test_model_creation(self) -> None:
        """Test creating a registered model."""
        model = RegisteredModel(
            id="model-123",
            name="llama-2-7b",
        )

        assert model.id == "model-123"
        assert model.name == "llama-2-7b"
        assert model.state == "LIVE"
        assert model.versions == []

    def test_model_with_versions(self) -> None:
        """Test model with versions."""
        v1 = ModelVersion(id="v1", name="1.0", registered_model_id="model-1")
        v2 = ModelVersion(id="v2", name="2.0", registered_model_id="model-1")

        model = RegisteredModel(
            id="model-1",
            name="test-model",
            description="A test model",
            owner="data-team",
            versions=[v1, v2],
        )

        assert model.description == "A test model"
        assert model.owner == "data-team"
        assert len(model.versions) == 2

    def test_get_latest_version_empty(self) -> None:
        """Test getting latest version when no versions exist."""
        model = RegisteredModel(id="model-1", name="empty-model")
        assert model.get_latest_version() is None

    def test_get_latest_version(self) -> None:
        """Test getting latest version by creation time."""
        from datetime import datetime, timezone

        v1 = ModelVersion(
            id="v1",
            name="1.0",
            registered_model_id="model-1",
            create_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        v2 = ModelVersion(
            id="v2",
            name="2.0",
            registered_model_id="model-1",
            create_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        v3 = ModelVersion(
            id="v3",
            name="3.0",
            registered_model_id="model-1",
            create_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
        )

        model = RegisteredModel(
            id="model-1",
            name="versioned-model",
            versions=[v1, v3, v2],  # Out of order
        )

        latest = model.get_latest_version()
        assert latest is not None
        assert latest.name == "2.0"


class TestValidationMetrics:
    """Test ValidationMetrics model."""

    def test_validation_metrics_creation(self) -> None:
        """Test creating validation metrics with required fields."""
        metrics = ValidationMetrics(
            model_name="llama-2-7b",
            model_version="v1.0.0",
        )

        assert metrics.model_name == "llama-2-7b"
        assert metrics.model_version == "v1.0.0"
        assert metrics.run_id is None
        assert metrics.gpu_count == 1  # Default
        assert metrics.input_tokens == 512  # Default
        assert metrics.output_tokens == 256  # Default

    def test_validation_metrics_with_latency(self) -> None:
        """Test validation metrics with latency data."""
        metrics = ValidationMetrics(
            model_name="granite-3b",
            model_version="v2.0",
            p50_latency_ms=45.0,
            p95_latency_ms=120.5,
            p99_latency_ms=250.0,
            mean_latency_ms=55.3,
        )

        assert metrics.p50_latency_ms == pytest.approx(45.0)
        assert metrics.p95_latency_ms == pytest.approx(120.5)
        assert metrics.p99_latency_ms == pytest.approx(250.0)
        assert metrics.mean_latency_ms == pytest.approx(55.3)

    def test_validation_metrics_with_throughput(self) -> None:
        """Test validation metrics with throughput data."""
        metrics = ValidationMetrics(
            model_name="mistral-7b",
            model_version="v1",
            tokens_per_second=1500.0,
            requests_per_second=25.5,
        )

        assert metrics.tokens_per_second == pytest.approx(1500.0)
        assert metrics.requests_per_second == pytest.approx(25.5)

    def test_validation_metrics_with_resources(self) -> None:
        """Test validation metrics with resource usage data."""
        metrics = ValidationMetrics(
            model_name="codellama-13b",
            model_version="v1.2",
            gpu_memory_gb=24.5,
            gpu_utilization_percent=85.0,
            peak_memory_gb=28.0,
            gpu_type="A100",
            gpu_count=2,
        )

        assert metrics.gpu_memory_gb == pytest.approx(24.5)
        assert metrics.gpu_utilization_percent == pytest.approx(85.0)
        assert metrics.peak_memory_gb == pytest.approx(28.0)
        assert metrics.gpu_type == "A100"
        assert metrics.gpu_count == 2

    def test_validation_metrics_full(self) -> None:
        """Test validation metrics with all fields."""
        from datetime import datetime, timezone

        metrics = ValidationMetrics(
            model_name="falcon-40b",
            model_version="v3.0",
            run_id="run-12345",
            p50_latency_ms=100.0,
            p95_latency_ms=200.0,
            p99_latency_ms=350.0,
            mean_latency_ms=120.0,
            tokens_per_second=800.0,
            requests_per_second=15.0,
            gpu_memory_gb=78.5,
            gpu_utilization_percent=92.0,
            peak_memory_gb=80.0,
            gpu_type="H100",
            gpu_count=4,
            input_tokens=1024,
            output_tokens=512,
            batch_size=8,
            concurrency=4,
            tensor_parallel_size=4,
            accuracy=0.92,
            perplexity=5.4,
            benchmark_date=datetime(2024, 6, 15, tzinfo=timezone.utc),
            notes="Production benchmark",
        )

        assert metrics.run_id == "run-12345"
        assert metrics.batch_size == 8
        assert metrics.concurrency == 4
        assert metrics.tensor_parallel_size == 4
        assert metrics.accuracy == pytest.approx(0.92)
        assert metrics.perplexity == pytest.approx(5.4)
        assert metrics.notes == "Production benchmark"


class TestMetricHistoryPoint:
    """Test MetricHistoryPoint model."""

    def test_metric_point_creation(self) -> None:
        """Test creating a metric history point."""
        point = MetricHistoryPoint(step=100, value=0.85)

        assert point.step == 100
        assert point.value == pytest.approx(0.85)
        assert point.timestamp is None

    def test_metric_point_with_timestamp(self) -> None:
        """Test metric point with timestamp."""
        from datetime import datetime, timezone

        ts = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        point = MetricHistoryPoint(step=200, value=0.92, timestamp=ts)

        assert point.step == 200
        assert point.timestamp == ts


class TestMetricHistory:
    """Test MetricHistory model."""

    def test_metric_history_creation(self) -> None:
        """Test creating a metric history."""
        history = MetricHistory(
            metric_name="accuracy",
            run_id="run-123",
        )

        assert history.metric_name == "accuracy"
        assert history.run_id == "run-123"
        assert history.history == []

    def test_get_last_value_empty(self) -> None:
        """Test getting last value from empty history."""
        history = MetricHistory(metric_name="loss", run_id="run-1")
        assert history.get_last_value() is None

    def test_get_last_value(self) -> None:
        """Test getting last value from history."""
        points = [
            MetricHistoryPoint(step=10, value=0.6),
            MetricHistoryPoint(step=50, value=0.85),
            MetricHistoryPoint(step=30, value=0.75),
        ]
        history = MetricHistory(metric_name="accuracy", run_id="run-1", history=points)

        assert history.get_last_value() == pytest.approx(0.85)  # step 50 is highest

    def test_get_average_empty(self) -> None:
        """Test getting average from empty history."""
        history = MetricHistory(metric_name="loss", run_id="run-1")
        assert history.get_average() is None

    def test_get_average(self) -> None:
        """Test getting average from history."""
        points = [
            MetricHistoryPoint(step=10, value=0.6),
            MetricHistoryPoint(step=20, value=0.8),
            MetricHistoryPoint(step=30, value=1.0),
        ]
        history = MetricHistory(metric_name="score", run_id="run-1", history=points)

        assert history.get_average() == pytest.approx(0.8)  # (0.6 + 0.8 + 1.0) / 3


class TestBenchmarkData:
    """Test BenchmarkData model."""

    def test_benchmark_creation(self) -> None:
        """Test creating benchmark data with required fields."""
        benchmark = BenchmarkData(
            model_name="llama-2-7b",
            model_version="v1.0",
            gpu_type="A100",
        )

        assert benchmark.model_name == "llama-2-7b"
        assert benchmark.model_version == "v1.0"
        assert benchmark.gpu_type == "A100"
        assert benchmark.gpu_count == 1  # Default
        assert benchmark.source == "model_registry"  # Default

    def test_benchmark_with_latency(self) -> None:
        """Test benchmark with latency data."""
        benchmark = BenchmarkData(
            model_name="mistral-7b",
            model_version="v2",
            gpu_type="H100",
            p50_latency_ms=25.0,
            p95_latency_ms=50.0,
            p99_latency_ms=75.0,
        )

        assert benchmark.p50_latency_ms == pytest.approx(25.0)
        assert benchmark.p95_latency_ms == pytest.approx(50.0)
        assert benchmark.p99_latency_ms == pytest.approx(75.0)

    def test_benchmark_with_throughput(self) -> None:
        """Test benchmark with throughput data."""
        benchmark = BenchmarkData(
            model_name="granite-3b",
            model_version="v1",
            gpu_type="L40S",
            tokens_per_second=2500.0,
            requests_per_second=50.0,
        )

        assert benchmark.tokens_per_second == pytest.approx(2500.0)
        assert benchmark.requests_per_second == pytest.approx(50.0)

    def test_benchmark_full(self) -> None:
        """Test benchmark with all fields."""
        from datetime import datetime, timezone

        benchmark = BenchmarkData(
            model_name="falcon-40b",
            model_version="v3",
            gpu_type="A100",
            gpu_count=4,
            p50_latency_ms=80.0,
            p95_latency_ms=150.0,
            p99_latency_ms=220.0,
            tokens_per_second=1200.0,
            requests_per_second=20.0,
            gpu_memory_gb=76.0,
            gpu_utilization_percent=88.0,
            input_tokens=1024,
            output_tokens=512,
            batch_size=4,
            concurrency=8,
            benchmark_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
            source="benchmark_suite",
        )

        assert benchmark.gpu_count == 4
        assert benchmark.input_tokens == 1024
        assert benchmark.output_tokens == 512
        assert benchmark.batch_size == 4
        assert benchmark.concurrency == 8
        assert benchmark.source == "benchmark_suite"
