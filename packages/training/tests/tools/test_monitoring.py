"""Tests for training monitoring tools."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from rhoai_mcp_training.tools.monitoring import register_tools


class TestMonitoringTools:
    """Test monitoring tools registration and execution."""

    @pytest.fixture
    def mock_mcp(self) -> MagicMock:
        """Create a mock FastMCP server."""
        mock = MagicMock()
        mock.tool = MagicMock(return_value=lambda f: f)
        return mock

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock RHOAIServer."""
        server = MagicMock()
        server.k8s = MagicMock()
        server.k8s.core_v1 = MagicMock()
        server.config.is_operation_allowed.return_value = (True, None)
        return server

    def test_register_tools_registers_all_monitoring_tools(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test that all monitoring tools are registered."""
        register_tools(mock_mcp, mock_server)

        # Check that tool decorator was called for each tool
        assert mock_mcp.tool.call_count >= 4  # At least 4 monitoring tools

    def test_get_training_progress(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test getting training progress."""
        mock_server.k8s.get.return_value = _make_mock_resource(
            "my-job",
            "training",
            annotations={
                "trainer.opendatahub.io/trainerStatus": json.dumps({
                    "trainingState": "Training",
                    "currentEpoch": 5,
                    "totalEpochs": 10,
                    "currentStep": 2500,
                    "totalSteps": 5000,
                    "loss": 1.5,
                    "learningRate": 0.0001,
                    "throughput": 100.5,
                }),
            },
        )

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f
            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["get_training_progress"](namespace="training", name="my-job")

        assert result["state"] == "Training"
        assert result["current_epoch"] == 5
        assert result["total_epochs"] == 10
        assert result["loss"] == pytest.approx(1.5)
        assert "progress_bar" in result

    def test_get_training_logs(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test getting training logs."""
        # Mock pod listing
        mock_pod = MagicMock()
        mock_pod.metadata.name = "my-job-trainer-0"
        mock_pod.status.phase = "Running"
        mock_result = MagicMock()
        mock_result.items = [mock_pod]
        mock_server.k8s.core_v1.list_namespaced_pod.return_value = mock_result

        # Mock log reading
        mock_server.k8s.core_v1.read_namespaced_pod_log.return_value = (
            "Epoch 1/10\nLoss: 2.5\nEpoch 2/10\nLoss: 2.0"
        )

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f
            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["get_training_logs"](namespace="training", name="my-job")

        assert "logs" in result
        assert "Epoch 1/10" in result["logs"]

    def test_get_job_events(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test getting job events."""
        mock_event1 = MagicMock()
        mock_event1.type = "Normal"
        mock_event1.reason = "Created"
        mock_event1.message = "Job created"
        mock_event1.last_timestamp = datetime.now(timezone.utc)

        mock_event2 = MagicMock()
        mock_event2.type = "Warning"
        mock_event2.reason = "PodScheduled"
        mock_event2.message = "Pod scheduled on node"
        mock_event2.last_timestamp = datetime.now(timezone.utc)

        mock_result = MagicMock()
        mock_result.items = [mock_event1, mock_event2]
        mock_server.k8s.core_v1.list_namespaced_event.return_value = mock_result

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f
            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["get_job_events"](namespace="training", name="my-job")

        assert len(result["events"]) == 2
        assert result["has_warnings"] is True

    def test_manage_checkpoints(
        self, mock_mcp: MagicMock, mock_server: MagicMock
    ) -> None:
        """Test managing checkpoints."""
        mock_server.k8s.get.return_value = _make_mock_resource(
            "my-job",
            "training",
            annotations={
                "trainer.opendatahub.io/checkpoint": json.dumps({
                    "latest": "/workspace/checkpoints/checkpoint-2500",
                    "checkpoints": [
                        {"step": 1000, "path": "/workspace/checkpoints/checkpoint-1000"},
                        {"step": 2500, "path": "/workspace/checkpoints/checkpoint-2500"},
                    ],
                }),
            },
        )

        tools = {}

        def capture_tool():
            def decorator(f):
                tools[f.__name__] = f
                return f
            return decorator

        mock_mcp.tool = capture_tool
        register_tools(mock_mcp, mock_server)

        result = tools["manage_checkpoints"](namespace="training", job_name="my-job")

        assert result["job_name"] == "my-job"
        assert "latest" in result
        assert len(result["checkpoints"]) == 2


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
