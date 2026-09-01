"""Tests for deployment execution module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rhoai_mcp.composites.planner.models import (
    DeploymentPlan,
    DeploymentPlanIssue,
    DeploymentPlanStep,
    ResolvedDeployParams,
)


def _make_params(**overrides: object) -> ResolvedDeployParams:
    defaults = {
        "name": "llama-3-1-8b",
        "namespace": "production",
        "runtime": "vllm-cuda-runtime",
        "model_format": "pytorch",
        "storage_uri": "oci://meta-llama/Llama-3.1-8B",
        "min_replicas": 1,
        "max_replicas": 2,
        "cpu_request": "24",
        "cpu_limit": "48",
        "memory_request": "128Gi",
        "memory_limit": "256Gi",
        "gpu_count": 1,
    }
    defaults.update(overrides)
    return ResolvedDeployParams(**defaults)


def _make_plan(ready: bool = True, **overrides: object) -> DeploymentPlan:
    defaults: dict = {
        "ready": ready,
        "recommendation_summary": {
            "model_id": "meta-llama/Llama-3.1-8B",
            "gpu_type": "A100-80",
            "gpu_count": 1,
            "replicas": 1,
            "predicted_cost_month": 1200,
            "meets_slo": True,
        },
        "resolved_params": _make_params(),
        "steps": [
            DeploymentPlanStep(action="runtime_ready", description="Runtime ready"),
            DeploymentPlanStep(action="deploy_model", description="Deploy model"),
            DeploymentPlanStep(action="wait_ready", description="Wait"),
            DeploymentPlanStep(action="test_endpoint", description="Test"),
        ],
        "issues": [],
        "warnings": [],
    }
    defaults.update(overrides)
    return DeploymentPlan(**defaults)


def _make_server(create_allowed: bool = True) -> MagicMock:
    server = MagicMock()
    server.config.is_operation_allowed.return_value = create_allowed
    server.k8s = MagicMock()
    return server


class TestExecuteDeployment:
    """Tests for execute_deployment."""

    async def test_plan_not_ready_returns_failure(self) -> None:
        from rhoai_mcp.composites.planner.execution import execute_deployment

        plan = _make_plan(
            ready=False,
            issues=[
                DeploymentPlanIssue(
                    category="storage",
                    message="Missing storage URI",
                    blocking=True,
                )
            ],
        )

        result = await execute_deployment(_make_server(), plan)

        assert result.success is False
        assert "blocking" in result.message.lower()

    async def test_create_not_allowed_returns_failure(self) -> None:
        from rhoai_mcp.composites.planner.execution import execute_deployment

        result = await execute_deployment(
            _make_server(create_allowed=False), _make_plan()
        )

        assert result.success is False
        assert "not allowed" in result.message.lower() or "read-only" in result.message.lower()

    @patch("rhoai_mcp.composites.planner.execution._get_endpoint_info")
    @patch("rhoai_mcp.composites.planner.execution._wait_for_ready")
    @patch("rhoai_mcp.domains.inference.models.InferenceServiceCreate")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_happy_path(
        self,
        _mock_client_cls: MagicMock,
        _mock_isvc_create: MagicMock,
        mock_wait: MagicMock,
        mock_endpoint: MagicMock,
    ) -> None:
        from rhoai_mcp.composites.planner.execution import execute_deployment

        mock_wait.return_value = (True, "Ready")
        mock_endpoint.return_value = {
            "url": "https://llama-3-1-8b.example.com",
            "status": "Ready",
        }

        result = await execute_deployment(_make_server(), _make_plan())

        assert result.success is True
        assert result.deployment_name == "llama-3-1-8b"
        assert result.namespace == "production"
        assert result.status == "Ready"
        assert result.endpoint_url == "https://llama-3-1-8b.example.com"
        assert result.validation is not None
        assert result.validation.reachable is True
        assert result.slo_comparison is not None

    @patch("rhoai_mcp.domains.inference.models.InferenceServiceCreate")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_deploy_model_exception(
        self,
        mock_client_cls: MagicMock,
        _mock_isvc_create: MagicMock,
    ) -> None:
        from rhoai_mcp.composites.planner.execution import execute_deployment

        mock_client_cls.return_value.deploy_model.side_effect = RuntimeError("K8s error")

        result = await execute_deployment(_make_server(), _make_plan())

        assert result.success is False
        assert "Failed to create InferenceService" in result.message

    @patch("rhoai_mcp.composites.planner.execution._get_endpoint_info")
    @patch("rhoai_mcp.composites.planner.execution._wait_for_ready")
    @patch("rhoai_mcp.domains.inference.models.InferenceServiceCreate")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_not_ready_after_timeout(
        self,
        _mock_client_cls: MagicMock,
        _mock_isvc_create: MagicMock,
        mock_wait: MagicMock,
        _mock_endpoint: MagicMock,
    ) -> None:
        from rhoai_mcp.composites.planner.execution import execute_deployment

        mock_wait.return_value = (False, "Timeout after 600s")

        result = await execute_deployment(_make_server(), _make_plan())

        assert result.success is False
        assert "not Ready" in result.message
        assert result.status == "Pending"

    @patch("rhoai_mcp.composites.planner.execution._get_endpoint_info")
    @patch("rhoai_mcp.composites.planner.execution._wait_for_ready")
    @patch("rhoai_mcp.composites.planner.execution._ensure_serving_runtime")
    @patch("rhoai_mcp.domains.inference.models.InferenceServiceCreate")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_runtime_creation_step(
        self,
        _mock_client_cls: MagicMock,
        _mock_isvc_create: MagicMock,
        mock_ensure_rt: MagicMock,
        mock_wait: MagicMock,
        mock_endpoint: MagicMock,
    ) -> None:
        from rhoai_mcp.composites.planner.execution import execute_deployment

        mock_ensure_rt.return_value = (True, "Created runtime")
        mock_wait.return_value = (True, "Ready")
        mock_endpoint.return_value = {
            "url": "https://model.example.com",
            "status": "Ready",
        }

        plan = _make_plan(
            steps=[
                DeploymentPlanStep(
                    action="create_runtime", description="Create runtime"
                ),
                DeploymentPlanStep(action="deploy_model", description="Deploy"),
                DeploymentPlanStep(action="wait_ready", description="Wait"),
                DeploymentPlanStep(action="test_endpoint", description="Test"),
            ]
        )

        result = await execute_deployment(_make_server(), plan)

        assert result.success is True
        mock_ensure_rt.assert_called_once()

    @patch("rhoai_mcp.composites.planner.execution._ensure_serving_runtime")
    @patch("rhoai_mcp.domains.inference.models.InferenceServiceCreate")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_runtime_creation_failure(
        self,
        _mock_client_cls: MagicMock,
        _mock_isvc_create: MagicMock,
        mock_ensure_rt: MagicMock,
    ) -> None:
        from rhoai_mcp.composites.planner.execution import execute_deployment

        mock_ensure_rt.return_value = (False, "No template found")

        plan = _make_plan(
            steps=[
                DeploymentPlanStep(
                    action="create_runtime", description="Create runtime"
                ),
                DeploymentPlanStep(action="deploy_model", description="Deploy"),
            ]
        )

        result = await execute_deployment(_make_server(), plan)

        assert result.success is False
        assert "Failed to create serving runtime" in result.message


class TestWaitForReady:
    """Tests for _wait_for_ready."""

    async def test_ready_immediately(self) -> None:
        from rhoai_mcp.composites.planner.execution import _wait_for_ready

        client = MagicMock()
        isvc = MagicMock()
        isvc.status.value = "Ready"
        client.get_inference_service.return_value = isvc

        ready, msg = await _wait_for_ready(client, "model", "ns")

        assert ready is True
        assert msg == "Ready"

    async def test_failed_status(self) -> None:
        from rhoai_mcp.composites.planner.execution import _wait_for_ready

        client = MagicMock()
        isvc = MagicMock()
        isvc.status.value = "Failed"
        condition = MagicMock()
        condition.reason = "ImagePullBackOff"
        isvc.conditions = [condition]
        client.get_inference_service.return_value = isvc

        ready, msg = await _wait_for_ready(client, "model", "ns")

        assert ready is False
        assert "ImagePullBackOff" in msg

    @patch("rhoai_mcp.composites.planner.execution.MAX_READINESS_WAIT_SECONDS", 0.01)
    @patch("rhoai_mcp.composites.planner.execution.READINESS_POLL_INTERVAL", 0.005)
    async def test_timeout(self) -> None:
        from rhoai_mcp.composites.planner.execution import _wait_for_ready

        client = MagicMock()
        isvc = MagicMock()
        isvc.status.value = "Pending"
        client.get_inference_service.return_value = isvc

        ready, msg = await _wait_for_ready(client, "model", "ns")

        assert ready is False
        assert "Timeout" in msg


class TestGetEndpointInfo:
    """Tests for _get_endpoint_info."""

    def test_success(self) -> None:
        from rhoai_mcp.composites.planner.execution import _get_endpoint_info

        client = MagicMock()
        client.get_model_endpoint.return_value = {
            "url": "https://model.example.com",
            "status": "Ready",
        }

        result = _get_endpoint_info(client, "model", "ns")
        assert result["url"] == "https://model.example.com"

    def test_exception_returns_fallback(self) -> None:
        from rhoai_mcp.composites.planner.execution import _get_endpoint_info

        client = MagicMock()
        client.get_model_endpoint.side_effect = RuntimeError("fail")

        result = _get_endpoint_info(client, "model", "ns")
        assert result["status"] == "Unknown"
        assert result["url"] is None


class TestEnsureServingRuntime:
    """Tests for _ensure_serving_runtime."""

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_runtime_already_exists(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.execution import _ensure_serving_runtime

        mock_cls.return_value.list_serving_runtimes.return_value = [
            {"name": "vllm-cuda-runtime"},
        ]

        ok, msg = await _ensure_serving_runtime(MagicMock(), "ns", "vllm-cuda-runtime")

        assert ok is True
        assert "already exists" in msg

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_created_from_template(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.execution import _ensure_serving_runtime

        mock_cls.return_value.list_serving_runtimes.return_value = []
        mock_cls.return_value.list_serving_runtime_templates.return_value = [
            {"name": "vllm-template", "creates_runtime": "vllm-cuda-runtime"},
        ]

        ok, msg = await _ensure_serving_runtime(MagicMock(), "ns", "vllm-cuda-runtime")

        assert ok is True
        assert "Created" in msg
        mock_cls.return_value.instantiate_serving_runtime_template.assert_called_once_with(
            template_name="vllm-template",
            target_namespace="ns",
        )

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_no_template_found(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.execution import _ensure_serving_runtime

        mock_cls.return_value.list_serving_runtimes.return_value = []
        mock_cls.return_value.list_serving_runtime_templates.return_value = [
            {"name": "other-template", "creates_runtime": "other-runtime"},
        ]

        ok, msg = await _ensure_serving_runtime(MagicMock(), "ns", "vllm-cuda-runtime")

        assert ok is False
        assert "No template found" in msg

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_exception_returns_failure(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.execution import _ensure_serving_runtime

        mock_cls.side_effect = RuntimeError("K8s connection failed")

        ok, msg = await _ensure_serving_runtime(MagicMock(), "ns", "vllm-cuda-runtime")

        assert ok is False
        assert "K8s connection failed" in msg
