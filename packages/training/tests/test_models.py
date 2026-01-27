"""Tests for training models."""

import json
from datetime import datetime, timezone

import pytest

from rhoai_mcp_training.models import (
    ClusterResources,
    GPUInfo,
    NodeResources,
    PeftMethod,
    TrainJob,
    TrainJobStatus,
    TrainingProgress,
    TrainingState,
)


class TestTrainingState:
    """Test TrainingState enum."""

    def test_training_states(self) -> None:
        """Test all training states exist."""
        assert TrainingState.INITIALIZING == "Initializing"
        assert TrainingState.TRAINING == "Training"
        assert TrainingState.COMPLETED == "Completed"
        assert TrainingState.FAILED == "Failed"
        assert TrainingState.SUSPENDED == "Suspended"


class TestPeftMethod:
    """Test PeftMethod enum."""

    def test_peft_methods(self) -> None:
        """Test all PEFT methods exist."""
        assert PeftMethod.FULL == "full"
        assert PeftMethod.LORA == "lora"
        assert PeftMethod.QLORA == "qlora"
        assert PeftMethod.DORA == "dora"


class TestTrainingProgress:
    """Test TrainingProgress model."""

    def test_from_annotation_full(self) -> None:
        """Test parsing full progress annotation."""
        annotation = json.dumps({
            "trainingState": "Training",
            "currentEpoch": 3,
            "totalEpochs": 10,
            "currentStep": 1500,
            "totalSteps": 5000,
            "loss": 2.34,
            "learningRate": 0.0001,
            "throughput": 450.5,
            "gradientNorm": 2.1,
            "estimatedTimeRemaining": 3600,
        })

        progress = TrainingProgress.from_annotation(annotation)

        assert progress.state == TrainingState.TRAINING
        assert progress.current_epoch == 3
        assert progress.total_epochs == 10
        assert progress.current_step == 1500
        assert progress.total_steps == 5000
        assert progress.loss == pytest.approx(2.34)
        assert progress.learning_rate == pytest.approx(0.0001)
        assert progress.throughput == pytest.approx(450.5)
        assert progress.gradient_norm == pytest.approx(2.1)
        assert progress.eta_seconds == 3600

    def test_from_annotation_minimal(self) -> None:
        """Test parsing minimal progress annotation."""
        annotation = json.dumps({
            "trainingState": "Initializing",
        })

        progress = TrainingProgress.from_annotation(annotation)

        assert progress.state == TrainingState.INITIALIZING
        assert progress.current_epoch == 0
        assert progress.total_epochs == 0

    def test_from_annotation_empty(self) -> None:
        """Test parsing empty annotation returns unknown state."""
        progress = TrainingProgress.from_annotation("")
        assert progress.state == TrainingState.INITIALIZING
        assert progress.current_step == 0

    def test_from_annotation_invalid_json(self) -> None:
        """Test parsing invalid JSON returns default progress."""
        progress = TrainingProgress.from_annotation("not-json")
        assert progress.state == TrainingState.INITIALIZING

    def test_progress_percent(self) -> None:
        """Test progress percentage calculation."""
        progress = TrainingProgress(
            state=TrainingState.TRAINING,
            current_step=500,
            total_steps=1000,
        )
        assert progress.progress_percent == pytest.approx(50.0)

    def test_progress_percent_zero_total(self) -> None:
        """Test progress percentage with zero total steps."""
        progress = TrainingProgress(state=TrainingState.TRAINING, total_steps=0)
        assert progress.progress_percent == 0.0

    def test_progress_bar(self) -> None:
        """Test progress bar rendering."""
        progress = TrainingProgress(
            state=TrainingState.TRAINING,
            current_step=500,
            total_steps=1000,
        )
        bar = progress.progress_bar(width=20)
        assert len(bar) == 20
        assert bar.count("=") == 10
        assert bar.count("-") == 10


class TestTrainJobStatus:
    """Test TrainJobStatus enum."""

    def test_job_statuses(self) -> None:
        """Test all job statuses exist."""
        assert TrainJobStatus.CREATED == "Created"
        assert TrainJobStatus.RUNNING == "Running"
        assert TrainJobStatus.COMPLETED == "Completed"
        assert TrainJobStatus.FAILED == "Failed"
        assert TrainJobStatus.SUSPENDED == "Suspended"


class TestTrainJob:
    """Test TrainJob model."""

    def test_from_resource_minimal(self) -> None:
        """Test creating TrainJob from minimal resource."""
        resource = _make_resource(
            name="test-job",
            namespace="default",
        )

        job = TrainJob.from_resource(resource)

        assert job.name == "test-job"
        assert job.namespace == "default"
        assert job.status == TrainJobStatus.CREATED

    def test_from_resource_with_spec(self) -> None:
        """Test creating TrainJob from resource with spec."""
        resource = _make_resource(
            name="fine-tune-llama",
            namespace="training",
            spec={
                "modelConfig": {
                    "name": "meta-llama/Llama-2-7b-hf",
                },
                "datasetConfig": {
                    "name": "tatsu-lab/alpaca",
                },
                "trainer": {
                    "numNodes": 2,
                },
            },
            annotations={
                "trainer.opendatahub.io/trainerStatus": json.dumps({
                    "trainingState": "Training",
                    "currentEpoch": 5,
                    "totalEpochs": 10,
                    "currentStep": 2500,
                    "totalSteps": 5000,
                    "loss": 1.5,
                }),
            },
        )

        job = TrainJob.from_resource(resource)

        assert job.name == "fine-tune-llama"
        assert job.namespace == "training"
        assert job.model_id == "meta-llama/Llama-2-7b-hf"
        assert job.dataset_id == "tatsu-lab/alpaca"
        assert job.num_nodes == 2
        assert job.progress is not None
        assert job.progress.state == TrainingState.TRAINING
        assert job.progress.current_epoch == 5

    def test_from_resource_with_conditions(self) -> None:
        """Test creating TrainJob with status conditions."""
        resource = _make_resource(
            name="test-job",
            namespace="default",
            status={
                "conditions": [
                    {
                        "type": "Created",
                        "status": "True",
                    },
                    {
                        "type": "Running",
                        "status": "True",
                    },
                ],
            },
        )

        job = TrainJob.from_resource(resource)
        assert job.status == TrainJobStatus.RUNNING


class TestNodeResources:
    """Test NodeResources model."""

    def test_node_resources(self) -> None:
        """Test NodeResources creation."""
        node = NodeResources(
            name="worker-1",
            cpu_total=32,
            cpu_allocatable=30,
            memory_total_gb=128.0,
            memory_allocatable_gb=120.0,
            gpu_count=4,
            gpu_type="nvidia.com/gpu",
        )

        assert node.name == "worker-1"
        assert node.cpu_total == 32
        assert node.gpu_count == 4


class TestGPUInfo:
    """Test GPUInfo model."""

    def test_gpu_info(self) -> None:
        """Test GPUInfo creation."""
        gpu = GPUInfo(
            type="nvidia.com/gpu",
            total=8,
            available=4,
            nodes_with_gpu=2,
        )

        assert gpu.type == "nvidia.com/gpu"
        assert gpu.total == 8
        assert gpu.available == 4


class TestClusterResources:
    """Test ClusterResources model."""

    def test_cluster_resources(self) -> None:
        """Test ClusterResources creation."""
        resources = ClusterResources(
            cpu_total=128,
            cpu_allocatable=120,
            memory_total_gb=512.0,
            memory_allocatable_gb=480.0,
            gpu_info=GPUInfo(
                type="nvidia.com/gpu",
                total=16,
                available=8,
                nodes_with_gpu=4,
            ),
            node_count=4,
        )

        assert resources.cpu_total == 128
        assert resources.gpu_info is not None
        assert resources.gpu_info.total == 16

    def test_has_gpus(self) -> None:
        """Test has_gpus property."""
        with_gpus = ClusterResources(
            cpu_total=32,
            cpu_allocatable=30,
            memory_total_gb=64.0,
            memory_allocatable_gb=60.0,
            gpu_info=GPUInfo(type="nvidia.com/gpu", total=4, available=2),
            node_count=1,
        )
        assert with_gpus.has_gpus is True

        without_gpus = ClusterResources(
            cpu_total=32,
            cpu_allocatable=30,
            memory_total_gb=64.0,
            memory_allocatable_gb=60.0,
            gpu_info=None,
            node_count=1,
        )
        assert without_gpus.has_gpus is False


def _make_resource(
    name: str,
    namespace: str = "default",
    spec: dict | None = None,
    status: dict | None = None,
    annotations: dict | None = None,
) -> object:
    """Create a mock Kubernetes resource for testing."""

    class MockMetadata:
        def __init__(self) -> None:
            self.name = name
            self.namespace = namespace
            self.uid = f"{name}-uid"
            self.creation_timestamp = datetime.now(timezone.utc)
            self.labels = {}
            self.annotations = annotations or {}

    class MockResource:
        def __init__(self) -> None:
            self.metadata = MockMetadata()
            self.spec = spec or {}
            self.status = status or {}

        def __getattr__(self, name: str) -> object:
            if name == "spec":
                return self.spec
            if name == "status":
                return self.status
            raise AttributeError(name)

    return MockResource()
