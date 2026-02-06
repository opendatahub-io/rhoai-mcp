"""Tests for NotebookClient pod diagnostic methods."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.domains.notebooks.client import NotebookClient


class TestNotebookClientDiagnostics:
    """Test NotebookClient pod/events/logs operations."""

    @pytest.fixture
    def mock_k8s(self) -> MagicMock:
        """Create a mock K8sClient."""
        mock = MagicMock()
        mock.core_v1 = MagicMock()
        return mock

    @pytest.fixture
    def client(self, mock_k8s: MagicMock) -> NotebookClient:
        """Create a NotebookClient with mocked K8sClient."""
        return NotebookClient(mock_k8s)

    def test_get_workbench_logs(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs from a workbench pod."""
        mock_k8s.core_v1.read_namespaced_pod_log.return_value = "Jupyter started"

        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-wb-0"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        logs = client.get_workbench_logs("test-ns", "my-wb", tail_lines=50)

        assert logs == "Jupyter started"
        mock_k8s.core_v1.list_namespaced_pod.assert_called_once_with(
            namespace="test-ns",
            label_selector="notebook-name=my-wb",
        )

    def test_get_workbench_logs_with_container(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs with a specific container name."""
        mock_k8s.core_v1.read_namespaced_pod_log.return_value = "oauth logs"

        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-wb-0"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        logs = client.get_workbench_logs(
            "test-ns", "my-wb", container="oauth-proxy"
        )

        assert logs == "oauth logs"
        mock_k8s.core_v1.read_namespaced_pod_log.assert_called_once_with(
            name="my-wb-0",
            namespace="test-ns",
            tail_lines=100,
            previous=False,
            container="oauth-proxy",
        )

    def test_get_workbench_logs_no_pods(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs when no pods exist."""
        mock_result = MagicMock()
        mock_result.items = []
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        logs = client.get_workbench_logs("test-ns", "my-wb")

        assert "No pods found" in logs

    def test_get_workbench_logs_error(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting logs when read fails."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-wb-0"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result
        mock_k8s.core_v1.read_namespaced_pod_log.side_effect = Exception("forbidden")

        logs = client.get_workbench_logs("test-ns", "my-wb")

        assert "Error getting logs" in logs

    def test_get_workbench_events(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting events for a workbench and its pods."""
        # Notebook-level event
        nb_event = MagicMock()
        nb_event.type = "Normal"
        nb_event.reason = "Created"
        nb_event.message = "Notebook created"
        nb_event.last_timestamp = datetime.now(timezone.utc)
        nb_event.count = 1

        nb_events_result = MagicMock()
        nb_events_result.items = [nb_event]

        # Pod-level event
        pod_event = MagicMock()
        pod_event.type = "Warning"
        pod_event.reason = "FailedMount"
        pod_event.message = "Unable to attach or mount volumes"
        pod_event.last_timestamp = datetime.now(timezone.utc)
        pod_event.count = 2

        pod_events_result = MagicMock()
        pod_events_result.items = [pod_event]

        # Pod
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-wb-0"
        pods_result = MagicMock()
        pods_result.items = [mock_pod]

        mock_k8s.core_v1.list_namespaced_event.side_effect = [
            nb_events_result,
            pod_events_result,
        ]
        mock_k8s.core_v1.list_namespaced_pod.return_value = pods_result

        events = client.get_workbench_events("test-ns", "my-wb")

        assert len(events) == 2
        assert events[0]["type"] == "Normal"
        assert events[0]["reason"] == "Created"
        assert events[1]["type"] == "Warning"
        assert events[1]["reason"] == "FailedMount"

    def test_get_workbench_events_no_pods(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test getting events when there are no pods."""
        nb_event = MagicMock()
        nb_event.type = "Normal"
        nb_event.reason = "Created"
        nb_event.message = "Created"
        nb_event.last_timestamp = None
        nb_event.count = 1

        nb_events_result = MagicMock()
        nb_events_result.items = [nb_event]

        pods_result = MagicMock()
        pods_result.items = []

        mock_k8s.core_v1.list_namespaced_event.return_value = nb_events_result
        mock_k8s.core_v1.list_namespaced_pod.return_value = pods_result

        events = client.get_workbench_events("test-ns", "my-wb")

        assert len(events) == 1
        assert events[0]["timestamp"] is None

    def test_get_workbench_pods(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test listing pods for a workbench."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-wb-0"
        mock_pod.status.phase = "Running"
        mock_pod.spec.node_name = "worker-0"
        ready_condition = MagicMock()
        ready_condition.type = "Ready"
        ready_condition.status = "True"
        mock_pod.status.conditions = [ready_condition]

        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        pods = client.get_workbench_pods("test-ns", "my-wb")

        assert len(pods) == 1
        assert pods[0]["name"] == "my-wb-0"
        assert pods[0]["phase"] == "Running"
        assert pods[0]["node"] == "worker-0"
        assert pods[0]["ready"] is True

    def test_get_workbench_pods_not_ready(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test listing pods where pod is not ready."""
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-wb-0"
        mock_pod.status.phase = "Pending"
        mock_pod.spec.node_name = None
        mock_pod.status.conditions = None

        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_k8s.core_v1.list_namespaced_pod.return_value = mock_result

        pods = client.get_workbench_pods("test-ns", "my-wb")

        assert len(pods) == 1
        assert pods[0]["ready"] is False

    def test_is_pod_ready_true(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test _is_pod_ready returns True for ready pods."""
        mock_pod = MagicMock()
        ready_condition = MagicMock()
        ready_condition.type = "Ready"
        ready_condition.status = "True"
        mock_pod.status.conditions = [ready_condition]

        assert client._is_pod_ready(mock_pod) is True

    def test_is_pod_ready_false(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test _is_pod_ready returns False for non-ready pods."""
        mock_pod = MagicMock()
        ready_condition = MagicMock()
        ready_condition.type = "Ready"
        ready_condition.status = "False"
        mock_pod.status.conditions = [ready_condition]

        assert client._is_pod_ready(mock_pod) is False

    def test_is_pod_ready_no_conditions(
        self, client: NotebookClient, mock_k8s: MagicMock
    ) -> None:
        """Test _is_pod_ready returns False when no conditions."""
        mock_pod = MagicMock()
        mock_pod.status.conditions = None

        assert client._is_pod_ready(mock_pod) is False
