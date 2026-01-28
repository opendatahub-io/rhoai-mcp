"""Tests for TrainingClient."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rhoai_mcp_core.domains.training.client import TrainingClient
from rhoai_mcp_core.domains.training.crds import TrainingCRDs
from rhoai_mcp_core.domains.training.models import (
    TrainingState,
)


class TestTrainingClient:
    """Test TrainingClient operations."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        return MagicMock()

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> TrainingClient:
        """Create a TrainingClient with mocked K8sClient."""
        return TrainingClient(mock_k8s)

    def test_list_training_jobs_empty(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test listing jobs when none exist."""
        mock_k8s.list_resources.return_value = []

        jobs = client.list_training_jobs("default")

        assert jobs == []
        mock_k8s.list_resources.assert_called_once_with(TrainingCRDs.TRAIN_JOB, namespace="default")

    def test_list_training_jobs_with_results(
        self, client: TrainingClient, mock_k8s: MagicMock
    ) -> None:
        """Test listing jobs returns parsed TrainJob models."""
        mock_k8s.list_resources.return_value = [
            _make_mock_resource("job-1", "default"),
            _make_mock_resource("job-2", "default"),
        ]

        jobs = client.list_training_jobs("default")

        assert len(jobs) == 2
        assert jobs[0].name == "job-1"
        assert jobs[1].name == "job-2"

    def test_get_training_job(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test getting a specific training job."""
        mock_k8s.get.return_value = _make_mock_resource(
            "my-job",
            "training",
            spec={
                "modelConfig": {"name": "meta-llama/Llama-2-7b-hf"},
            },
        )

        job = client.get_training_job("training", "my-job")

        assert job.name == "my-job"
        assert job.namespace == "training"
        assert job.model_id == "meta-llama/Llama-2-7b-hf"
        mock_k8s.get.assert_called_once_with(TrainingCRDs.TRAIN_JOB, "my-job", namespace="training")

    def test_get_training_job_with_progress(
        self, client: TrainingClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting a job with training progress annotation."""
        mock_k8s.get.return_value = _make_mock_resource(
            "my-job",
            "training",
            annotations={
                "trainer.opendatahub.io/trainerStatus": json.dumps(
                    {
                        "trainingState": "Training",
                        "currentEpoch": 5,
                        "totalEpochs": 10,
                        "currentStep": 2500,
                        "totalSteps": 5000,
                        "loss": 1.5,
                    }
                ),
            },
        )

        job = client.get_training_job("training", "my-job")

        assert job.progress is not None
        assert job.progress.state == TrainingState.TRAINING
        assert job.progress.current_epoch == 5
        assert job.progress.loss == pytest.approx(1.5)

    def test_create_training_job(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test creating a training job."""
        mock_k8s.create.return_value = _make_mock_resource("new-job", "training")

        job = client.create_training_job(
            namespace="training",
            name="new-job",
            model_id="meta-llama/Llama-2-7b-hf",
            dataset_id="tatsu-lab/alpaca",
            runtime_ref="transformers-runtime",
        )

        assert job.name == "new-job"
        mock_k8s.create.assert_called_once()
        call_args = mock_k8s.create.call_args
        assert call_args[0][0] == TrainingCRDs.TRAIN_JOB
        body = call_args[1]["body"]
        assert body["metadata"]["name"] == "new-job"
        assert body["spec"]["modelConfig"]["name"] == "meta-llama/Llama-2-7b-hf"

    def test_delete_training_job(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test deleting a training job."""
        client.delete_training_job("training", "my-job")

        mock_k8s.delete.assert_called_once_with(
            TrainingCRDs.TRAIN_JOB, "my-job", namespace="training"
        )

    def test_suspend_training_job(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test suspending a training job."""
        mock_k8s.patch.return_value = _make_mock_resource("my-job", "training")

        client.suspend_training_job("training", "my-job")

        mock_k8s.patch.assert_called_once()
        call_args = mock_k8s.patch.call_args
        assert call_args[0][0] == TrainingCRDs.TRAIN_JOB
        assert call_args[0][1] == "my-job"
        body = call_args[1]["body"]
        assert body["spec"]["suspend"] is True

    def test_resume_training_job(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test resuming a training job."""
        mock_k8s.patch.return_value = _make_mock_resource("my-job", "training")

        client.resume_training_job("training", "my-job")

        mock_k8s.patch.assert_called_once()
        call_args = mock_k8s.patch.call_args
        body = call_args[1]["body"]
        assert body["spec"]["suspend"] is False

    def test_list_cluster_training_runtimes(
        self, client: TrainingClient, mock_k8s: MagicMock
    ) -> None:
        """Test listing cluster training runtimes."""
        mock_k8s.list_resources.return_value = [
            _make_mock_resource("transformers-runtime", None),
            _make_mock_resource("pytorch-runtime", None),
        ]

        runtimes = client.list_cluster_training_runtimes()

        assert len(runtimes) == 2
        mock_k8s.list_resources.assert_called_once_with(TrainingCRDs.CLUSTER_TRAINING_RUNTIME)

    def test_get_cluster_training_runtime(
        self, client: TrainingClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting a specific cluster training runtime."""
        mock_k8s.get.return_value = _make_mock_resource("transformers-runtime", None)

        runtime = client.get_cluster_training_runtime("transformers-runtime")

        assert runtime.name == "transformers-runtime"
        mock_k8s.get.assert_called_once_with(
            TrainingCRDs.CLUSTER_TRAINING_RUNTIME, "transformers-runtime"
        )


class TestTrainingClientPodOperations:
    """Test TrainingClient pod operations."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        mock = MagicMock()
        mock.core_v1 = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> TrainingClient:
        """Create a TrainingClient with mocked K8sClient."""
        return TrainingClient(mock_k8s)

    def test_get_training_logs(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test getting training logs."""
        mock_k8s.core_v1.read_namespaced_pod_log.return_value = "Training started..."

        # Mock listing pods for the job
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-job-trainer-0"
        mock_pod.status.phase = "Running"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        logs = client.get_training_logs("training", "my-job", tail_lines=100)

        assert logs == "Training started..."

    def test_get_job_events(self, client: TrainingClient, mock_k8s: MagicMock) -> None:
        """Test getting job events."""
        mock_event = MagicMock()
        mock_event.type = "Normal"
        mock_event.reason = "Created"
        mock_event.message = "Job created"
        mock_event.last_timestamp = datetime.now(timezone.utc)
        mock_event.involved_object.name = "my-job"
        mock_event.involved_object.kind = "TrainJob"

        mock_result = MagicMock()
        mock_result.items = [mock_event]
        mock_k8s.core_v1.list_namespaced_event.return_value = mock_result

        events = client.get_job_events("training", "my-job")

        assert len(events) == 1
        assert events[0]["type"] == "Normal"
        assert events[0]["reason"] == "Created"


def _make_mock_resource(
    name: str,
    namespace: str | None = "default",
    spec: dict | None = None,
    status: dict | None = None,
    annotations: dict | None = None,
    labels: dict | None = None,
) -> MagicMock:
    """Create a mock Kubernetes resource."""
    mock = MagicMock()
    mock.metadata.name = name
    mock.metadata.namespace = namespace
    mock.metadata.uid = f"{name}-uid"
    mock.metadata.creation_timestamp = datetime.now(timezone.utc)
    mock.metadata.labels = labels or {}
    mock.metadata.annotations = annotations or {}
    mock.spec = spec or {}
    mock.status = status or {}
    return mock
