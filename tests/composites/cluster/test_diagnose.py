"""Tests for diagnose_resource diagnostic functions."""

from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.composites.cluster.tools import _diagnose_model, _diagnose_workbench


class TestDiagnoseModel:
    """Test _diagnose_model function."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock RHOAIServer."""
        server = MagicMock()
        server.k8s = MagicMock()
        server.k8s.core_v1 = MagicMock()
        return server

    def test_diagnose_model_ready(self, mock_server: MagicMock) -> None:
        """Test diagnosing a ready model returns events and logs."""
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "my-model"
        mock_isvc.status.value = "Ready"
        mock_isvc.runtime = "vllm-runtime"
        mock_isvc.url = "https://my-model.example.com"
        mock_isvc.storage_uri = "s3://bucket/model"

        with patch(
            "rhoai_mcp.domains.inference.client.InferenceClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_inference_service.return_value = mock_isvc
            mock_client.get_inference_service_events.return_value = [
                {"type": "Normal", "reason": "Created", "message": "Created"}
            ]
            mock_client.get_inference_service_logs.return_value = "Model loaded"

            result = _diagnose_model(mock_server, "my-model", "test-ns")

        assert result["resource"]["name"] == "my-model"
        assert result["resource"]["status"] == "Ready"
        assert result["status_summary"] == "Model is Ready"
        assert len(result["events"]) == 1
        assert result["logs"] == "Model loaded"
        assert "ServingRuntime: vllm-runtime" in result["related_resources"]
        assert "StorageURI: s3://bucket/model" in result["related_resources"]
        assert len(result["issues_detected"]) == 0

    def test_diagnose_model_not_ready(self, mock_server: MagicMock) -> None:
        """Test diagnosing a model that is not ready."""
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "my-model"
        mock_isvc.status.value = "Failed"
        mock_isvc.runtime = "vllm-runtime"
        mock_isvc.url = None
        mock_isvc.storage_uri = None

        with patch(
            "rhoai_mcp.domains.inference.client.InferenceClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_inference_service.return_value = mock_isvc
            mock_client.get_inference_service_events.return_value = []
            mock_client.get_inference_service_logs.return_value = ""

            result = _diagnose_model(mock_server, "my-model", "test-ns")

        assert "Model not ready: Failed" in result["issues_detected"]

    def test_diagnose_model_image_pull_failure(self, mock_server: MagicMock) -> None:
        """Test diagnosing a model with ImagePullBackOff."""
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "my-model"
        mock_isvc.status.value = "Failed"
        mock_isvc.runtime = "vllm-runtime"
        mock_isvc.url = None
        mock_isvc.storage_uri = None

        with patch(
            "rhoai_mcp.domains.inference.client.InferenceClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_inference_service.return_value = mock_isvc
            mock_client.get_inference_service_events.return_value = [
                {
                    "type": "Warning",
                    "reason": "Failed",
                    "message": "ImagePullBackOff: unable to pull image",
                }
            ]
            mock_client.get_inference_service_logs.return_value = ""

            result = _diagnose_model(mock_server, "my-model", "test-ns")

        assert "Image pull failure" in result["issues_detected"]
        assert any("image name" in fix for fix in result["suggested_fixes"])

    def test_diagnose_model_failed_scheduling(self, mock_server: MagicMock) -> None:
        """Test diagnosing a model with FailedScheduling."""
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "my-model"
        mock_isvc.status.value = "Pending"
        mock_isvc.runtime = "vllm-runtime"
        mock_isvc.url = None
        mock_isvc.storage_uri = None

        with patch(
            "rhoai_mcp.domains.inference.client.InferenceClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_inference_service.return_value = mock_isvc
            mock_client.get_inference_service_events.return_value = [
                {
                    "type": "Warning",
                    "reason": "FailedScheduling",
                    "message": "Insufficient nvidia.com/gpu",
                }
            ]
            mock_client.get_inference_service_logs.return_value = ""

            result = _diagnose_model(mock_server, "my-model", "test-ns")

        assert "Pod scheduling failed" in result["issues_detected"]
        assert any("GPU" in fix for fix in result["suggested_fixes"])

    def test_diagnose_model_crash_loop(self, mock_server: MagicMock) -> None:
        """Test diagnosing a model with CrashLoopBackOff."""
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "my-model"
        mock_isvc.status.value = "Failed"
        mock_isvc.runtime = "vllm-runtime"
        mock_isvc.url = None
        mock_isvc.storage_uri = None

        with patch(
            "rhoai_mcp.domains.inference.client.InferenceClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_inference_service.return_value = mock_isvc
            mock_client.get_inference_service_events.return_value = [
                {
                    "type": "Warning",
                    "reason": "BackOff",
                    "message": "Back-off restarting failed container (CrashLoopBackOff)",
                }
            ]
            mock_client.get_inference_service_logs.return_value = ""

            result = _diagnose_model(mock_server, "my-model", "test-ns")

        assert "Container crash loop" in result["issues_detected"]

    def test_diagnose_model_oom_killed(self, mock_server: MagicMock) -> None:
        """Test diagnosing a model with OOMKilled."""
        mock_isvc = MagicMock()
        mock_isvc.metadata.name = "my-model"
        mock_isvc.status.value = "Failed"
        mock_isvc.runtime = "vllm-runtime"
        mock_isvc.url = None
        mock_isvc.storage_uri = None

        with patch(
            "rhoai_mcp.domains.inference.client.InferenceClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_inference_service.return_value = mock_isvc
            mock_client.get_inference_service_events.return_value = [
                {
                    "type": "Warning",
                    "reason": "OOMKilled",
                    "message": "Container killed due to OOM",
                }
            ]
            mock_client.get_inference_service_logs.return_value = ""

            result = _diagnose_model(mock_server, "my-model", "test-ns")

        assert "Out of memory" in result["issues_detected"]
        assert any("memory" in fix for fix in result["suggested_fixes"])

    def test_diagnose_model_exception(self, mock_server: MagicMock) -> None:
        """Test diagnosing a model when the client throws."""
        with patch(
            "rhoai_mcp.domains.inference.client.InferenceClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_inference_service.side_effect = Exception("not found")

            result = _diagnose_model(mock_server, "my-model", "test-ns")

        assert any("Failed to get model" in i for i in result["issues_detected"])


class TestDiagnoseWorkbench:
    """Test _diagnose_workbench function."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Create a mock RHOAIServer."""
        server = MagicMock()
        server.k8s = MagicMock()
        server.k8s.core_v1 = MagicMock()
        return server

    def test_diagnose_workbench_running(self, mock_server: MagicMock) -> None:
        """Test diagnosing a running workbench returns events and logs."""
        mock_wb = MagicMock()
        mock_wb.metadata.name = "my-wb"
        mock_wb.status.value = "Running"
        mock_wb.image = "jupyter:latest"
        mock_wb.url = "https://my-wb.example.com"
        mock_wb.volumes = ["my-wb-pvc", "shared-data"]

        with patch(
            "rhoai_mcp.domains.notebooks.client.NotebookClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_workbench.return_value = mock_wb
            mock_client.get_workbench_events.return_value = [
                {"type": "Normal", "reason": "Started", "message": "Started"}
            ]
            mock_client.get_workbench_logs.return_value = "Jupyter running"

            result = _diagnose_workbench(mock_server, "my-wb", "test-ns")

        assert result["resource"]["name"] == "my-wb"
        assert result["resource"]["status"] == "Running"
        assert len(result["events"]) == 1
        assert result["logs"] == "Jupyter running"
        assert "PVC: my-wb-pvc" in result["related_resources"]
        assert "PVC: shared-data" in result["related_resources"]
        assert len(result["issues_detected"]) == 0

    def test_diagnose_workbench_stopped(self, mock_server: MagicMock) -> None:
        """Test diagnosing a stopped workbench skips logs."""
        mock_wb = MagicMock()
        mock_wb.metadata.name = "my-wb"
        mock_wb.status.value = "Stopped"
        mock_wb.image = "jupyter:latest"
        mock_wb.url = "https://my-wb.example.com"
        mock_wb.volumes = []

        with patch(
            "rhoai_mcp.domains.notebooks.client.NotebookClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_workbench.return_value = mock_wb
            mock_client.get_workbench_events.return_value = []

            result = _diagnose_workbench(mock_server, "my-wb", "test-ns")

        assert "Workbench is stopped" in result["issues_detected"]
        assert result["logs"] is None
        # Should not have called get_workbench_logs
        mock_client.get_workbench_logs.assert_not_called()
        # With no volumes, falls back to default PVC name
        assert "PVC: my-wb-pvc" in result["related_resources"]

    def test_diagnose_workbench_error_with_events(
        self, mock_server: MagicMock
    ) -> None:
        """Test diagnosing a workbench in error state with failure events."""
        mock_wb = MagicMock()
        mock_wb.metadata.name = "my-wb"
        mock_wb.status.value = "Error"
        mock_wb.image = "jupyter:latest"
        mock_wb.url = None
        mock_wb.volumes = ["my-wb-pvc"]

        with patch(
            "rhoai_mcp.domains.notebooks.client.NotebookClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_workbench.return_value = mock_wb
            mock_client.get_workbench_events.return_value = [
                {
                    "type": "Warning",
                    "reason": "ErrImagePull",
                    "message": "rpc error: image not found",
                }
            ]
            mock_client.get_workbench_logs.return_value = ""

            result = _diagnose_workbench(mock_server, "my-wb", "test-ns")

        assert "Image pull failure" in result["issues_detected"]
        assert any("image name" in fix for fix in result["suggested_fixes"])

    def test_diagnose_workbench_error_no_specific_events(
        self, mock_server: MagicMock
    ) -> None:
        """Test diagnosing an error workbench with no specific failure pattern."""
        mock_wb = MagicMock()
        mock_wb.metadata.name = "my-wb"
        mock_wb.status.value = "Error"
        mock_wb.image = "jupyter:latest"
        mock_wb.url = None
        mock_wb.volumes = []

        with patch(
            "rhoai_mcp.domains.notebooks.client.NotebookClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_workbench.return_value = mock_wb
            mock_client.get_workbench_events.return_value = []
            mock_client.get_workbench_logs.return_value = ""

            result = _diagnose_workbench(mock_server, "my-wb", "test-ns")

        assert "Workbench is in error state" in result["issues_detected"]
        assert any("events and logs" in fix for fix in result["suggested_fixes"])

    def test_diagnose_workbench_failed_scheduling(
        self, mock_server: MagicMock
    ) -> None:
        """Test diagnosing a workbench with FailedScheduling."""
        mock_wb = MagicMock()
        mock_wb.metadata.name = "my-wb"
        mock_wb.status.value = "Starting"
        mock_wb.image = "jupyter:latest"
        mock_wb.url = None
        mock_wb.volumes = ["my-wb-pvc"]

        with patch(
            "rhoai_mcp.domains.notebooks.client.NotebookClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_workbench.return_value = mock_wb
            mock_client.get_workbench_events.return_value = [
                {
                    "type": "Warning",
                    "reason": "FailedScheduling",
                    "message": "0/6 nodes are available: Insufficient cpu",
                }
            ]
            mock_client.get_workbench_logs.return_value = ""

            result = _diagnose_workbench(mock_server, "my-wb", "test-ns")

        assert "Pod scheduling failed" in result["issues_detected"]

    def test_diagnose_workbench_exception(self, mock_server: MagicMock) -> None:
        """Test diagnosing a workbench when the client throws."""
        with patch(
            "rhoai_mcp.domains.notebooks.client.NotebookClient"
        ) as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.get_workbench.side_effect = Exception("not found")

            result = _diagnose_workbench(mock_server, "my-wb", "test-ns")

        assert any("Failed to get workbench" in i for i in result["issues_detected"])
