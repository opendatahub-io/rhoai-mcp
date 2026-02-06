"""Tests for InferenceClient pod diagnostic methods."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.inference.client import InferenceClient


class TestInferenceClientDiagnostics:
    """Test InferenceClient pod/events/logs operations."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        mock = MagicMock()
        mock.core_v1 = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> InferenceClient:
        """Create an InferenceClient with mocked K8sClient."""
        return InferenceClient(mock_k8s)

    def test_get_inference_service_logs(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs from an InferenceService pod."""
        mock_k8s.core_v1.read_namespaced_pod_log.return_value = "Model loaded successfully"

        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-model-predictor-00001-abc"
        mock_pod.status.phase = "Running"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        logs = client.get_inference_service_logs("test-ns", "my-model", tail_lines=50)

        assert logs == "Model loaded successfully"
        mock_k8s.core_v1.list_namespaced_pod.assert_called_once_with(
            namespace="test-ns",
            label_selector="serving.kserve.io/inferenceservice=my-model",
        )

    def test_get_inference_service_logs_with_container(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs with a specific container name."""
        mock_k8s.core_v1.read_namespaced_pod_log.return_value = "container logs"

        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-model-pod-0"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        logs = client.get_inference_service_logs(
            "test-ns", "my-model", container="kserve-container"
        )

        assert logs == "container logs"
        mock_k8s.core_v1.read_namespaced_pod_log.assert_called_once_with(
            name="my-model-pod-0",
            namespace="test-ns",
            tail_lines=100,
            previous=False,
            container="kserve-container",
        )

    def test_get_inference_service_logs_no_pods(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs when no pods exist."""
        mock_result = MagicMock()
        mock_result.items = []
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        logs = client.get_inference_service_logs("test-ns", "my-model")

        assert "No pods found" in logs

    def test_get_inference_service_logs_error(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs when read fails."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-model-pod-0"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result
        mock_k8s.core_v1.read_namespaced_pod_log.side_effect = Exception("forbidden")

        logs = client.get_inference_service_logs("test-ns", "my-model")

        assert "Error getting logs" in logs

    def test_get_inference_service_events(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting events for an InferenceService and its pods."""
        # InferenceService-level event
        isvc_event = MagicMock()
        isvc_event.type = "Normal"
        isvc_event.reason = "Created"
        isvc_event.message = "InferenceService created"
        isvc_event.last_timestamp = datetime.now(timezone.utc)
        isvc_event.count = 1

        isvc_events_result = MagicMock()
        isvc_events_result.items = [isvc_event]

        # Pod-level event
        pod_event = MagicMock()
        pod_event.type = "Warning"
        pod_event.reason = "FailedScheduling"
        pod_event.message = "Insufficient nvidia.com/gpu"
        pod_event.last_timestamp = datetime.now(timezone.utc)
        pod_event.count = 3

        pod_events_result = MagicMock()
        pod_events_result.items = [pod_event]

        # Pod
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-model-predictor-abc"
        pods_result = MagicMock()
        pods_result.items = [mock_pod]

        mock_k8s.core_v1.list_namespaced_event.side_effect = [
            isvc_events_result,
            pod_events_result,
        ]
        mock_k8s.core_v1.list_namespaced_pod.return_value = pods_result

        events = client.get_inference_service_events("test-ns", "my-model")

        assert len(events) == 2
        assert events[0]["type"] == "Normal"
        assert events[0]["reason"] == "Created"
        assert events[1]["type"] == "Warning"
        assert events[1]["reason"] == "FailedScheduling"

    def test_get_inference_service_events_no_pods(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting events when there are no pods."""
        isvc_event = MagicMock()
        isvc_event.type = "Normal"
        isvc_event.reason = "Created"
        isvc_event.message = "Created"
        isvc_event.last_timestamp = None
        isvc_event.count = 1

        isvc_events_result = MagicMock()
        isvc_events_result.items = [isvc_event]

        pods_result = MagicMock()
        pods_result.items = []

        mock_k8s.core_v1.list_namespaced_event.return_value = isvc_events_result
        mock_k8s.core_v1.list_namespaced_pod.return_value = pods_result

        events = client.get_inference_service_events("test-ns", "my-model")

        assert len(events) == 1
        assert events[0]["timestamp"] is None

    def test_get_inference_service_pods(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test listing pods for an InferenceService."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-model-predictor-00001-abc"
        mock_pod.status.phase = "Running"
        mock_pod.spec.node_name = "worker-0"
        ready_condition = MagicMock()
        ready_condition.type = "Ready"
        ready_condition.status = "True"
        mock_pod.status.conditions = [ready_condition]

        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        pods = client.get_inference_service_pods("test-ns", "my-model")

        assert len(pods) == 1
        assert pods[0]["name"] == "my-model-predictor-00001-abc"
        assert pods[0]["phase"] == "Running"
        assert pods[0]["node"] == "worker-0"
        assert pods[0]["ready"] is True

    def test_get_inference_service_pods_not_ready(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test listing pods where pod is not ready."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-model-pod-0"
        mock_pod.status.phase = "Pending"
        mock_pod.spec.node_name = None
        mock_pod.status.conditions = None

        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        pods = client.get_inference_service_pods("test-ns", "my-model")

        assert len(pods) == 1
        assert pods[0]["ready"] is False

    def test_is_pod_ready_true(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test _is_pod_ready returns True for ready pods."""
        mock_pod = MagicMock()
        ready_condition = MagicMock()
        ready_condition.type = "Ready"
        ready_condition.status = "True"
        mock_pod.status.conditions = [ready_condition]

        assert client._is_pod_ready(mock_pod) is True

    def test_is_pod_ready_false(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test _is_pod_ready returns False for non-ready pods."""
        mock_pod = MagicMock()
        ready_condition = MagicMock()
        ready_condition.type = "Ready"
        ready_condition.status = "False"
        mock_pod.status.conditions = [ready_condition]

        assert client._is_pod_ready(mock_pod) is False

    def test_is_pod_ready_no_conditions(
        self, client: InferenceClient, mock_k8s: MagicMock
    ) -> None:
        """Test _is_pod_ready returns False when no conditions."""
        mock_pod = MagicMock()
        mock_pod.status.conditions = None

        assert client._is_pod_ready(mock_pod) is False
