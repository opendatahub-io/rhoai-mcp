"""Tests for deployment planning module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rhoai_mcp.composites.planner.deployment import (
    GPU_RESOURCE_PROFILES,
    _build_execution_steps,
    generate_deployment_name,
    plan_deployment,
)


class TestGenerateDeploymentName:
    """Tests for generate_deployment_name."""

    def test_simple_model_id(self) -> None:
        name = generate_deployment_name("meta-llama/Llama-3.1-70B-Instruct")
        assert name == "llama-3-1-70b-instruct"

    def test_no_org_prefix(self) -> None:
        name = generate_deployment_name("gpt2")
        assert name == "gpt2"

    def test_dots_replaced(self) -> None:
        name = generate_deployment_name("org/model.v1.2")
        assert name == "model-v1-2"

    def test_consecutive_dashes_collapsed(self) -> None:
        name = generate_deployment_name("org/model--name")
        assert name == "model-name"

    def test_leading_trailing_dashes_stripped(self) -> None:
        name = generate_deployment_name("org/-model-")
        assert name == "model"

    def test_long_name_truncated(self) -> None:
        long_id = "org/" + "a" * 60
        name = generate_deployment_name(long_id)
        assert len(name) <= 50

    def test_uppercase_lowered(self) -> None:
        name = generate_deployment_name("Org/MyModel")
        assert name == "mymodel"


class TestBuildExecutionSteps:
    """Tests for _build_execution_steps."""

    def test_with_runtime_creation(self) -> None:
        steps = _build_execution_steps("my-model", "ns", "vllm-cuda-runtime", True)
        actions = [s.action for s in steps]
        assert actions == ["create_runtime", "deploy_model", "wait_ready", "test_endpoint"]

    def test_without_runtime_creation(self) -> None:
        steps = _build_execution_steps("my-model", "ns", "vllm-cuda-runtime", False)
        actions = [s.action for s in steps]
        assert actions == ["runtime_ready", "deploy_model", "wait_ready", "test_endpoint"]

    def test_step_descriptions_reference_params(self) -> None:
        steps = _build_execution_steps("llama-70b", "prod", "my-rt", True)
        assert "my-rt" in steps[0].description
        assert "llama-70b" in steps[1].description
        assert "prod" in steps[1].description


class TestPlanDeployment:
    """Tests for plan_deployment."""

    def _make_server(self) -> MagicMock:
        server = MagicMock()
        server.k8s = MagicMock()
        return server

    def _make_recommendation(self, **overrides: object) -> dict:
        rec: dict = {
            "model_id": "meta-llama/Llama-3.1-8B",
            "model_name": "Llama 3.1 8B",
            "gpu_config": {
                "gpu_type": "A100-80",
                "gpu_count": 1,
                "tensor_parallel": 1,
                "replicas": 1,
            },
            "cost_per_month_usd": 1200,
            "meets_slo": True,
        }
        rec.update(overrides)
        return rec

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_happy_path(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        mock_runtime.return_value = ("vllm-runtime", [])
        mock_storage.return_value = ("oci://meta-llama/Llama-3.1-8B", [])
        mock_preflight.return_value = ([], [])

        plan = await plan_deployment(
            self._make_server(),
            self._make_recommendation(),
            namespace="production",
        )

        assert plan.ready is True
        assert plan.resolved_params.name == "llama-3-1-8b"
        assert plan.resolved_params.namespace == "production"
        assert plan.resolved_params.runtime == "vllm-runtime"
        assert plan.resolved_params.storage_uri == "oci://meta-llama/Llama-3.1-8B"
        assert plan.resolved_params.gpu_count == 1
        assert plan.recommendation_summary["model_id"] == "meta-llama/Llama-3.1-8B"

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_with_name_override(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        mock_runtime.return_value = ("vllm-runtime", [])
        mock_storage.return_value = ("s3://bucket/model", [])
        mock_preflight.return_value = ([], [])

        plan = await plan_deployment(
            self._make_server(),
            self._make_recommendation(),
            namespace="prod",
            name_override="my-custom-name",
        )

        assert plan.resolved_params.name == "my-custom-name"

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_with_storage_uri_override(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        mock_runtime.return_value = ("vllm-runtime", [])
        mock_storage.return_value = ("s3://my-bucket/model", [])
        mock_preflight.return_value = ([], [])

        server = self._make_server()
        plan = await plan_deployment(
            server,
            self._make_recommendation(),
            namespace="prod",
            storage_uri_override="s3://my-bucket/model",
        )

        assert plan.resolved_params.storage_uri == "s3://my-bucket/model"

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_with_runtime_override(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        mock_runtime.return_value = ("custom-runtime", [])
        mock_storage.return_value = ("oci://model", [])
        mock_preflight.return_value = ([], [])

        plan = await plan_deployment(
            self._make_server(),
            self._make_recommendation(),
            namespace="prod",
            runtime_override="custom-runtime",
        )

        assert plan.resolved_params.runtime == "custom-runtime"

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_blocking_issues_make_plan_not_ready(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        from rhoai_mcp.composites.planner.models import DeploymentPlanIssue

        mock_runtime.return_value = ("vllm-runtime", [])
        mock_storage.return_value = (None, [
            DeploymentPlanIssue(
                category="storage",
                message="Cannot resolve storage URI",
                blocking=True,
                suggestion="Provide storage_uri",
            )
        ])
        mock_preflight.return_value = ([], [])

        plan = await plan_deployment(
            self._make_server(),
            self._make_recommendation(),
            namespace="prod",
        )

        assert plan.ready is False
        assert len(plan.issues) == 1
        assert plan.issues[0].blocking is True
        assert plan.steps[0].action == "fix_issues"

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_no_runtime_found_triggers_creation_step(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        from rhoai_mcp.composites.planner.models import DeploymentPlanIssue

        mock_runtime.return_value = (None, [
            DeploymentPlanIssue(
                category="runtime",
                message="No runtime found",
                blocking=False,
            )
        ])
        mock_storage.return_value = ("oci://model", [])
        mock_preflight.return_value = ([], [])

        plan = await plan_deployment(
            self._make_server(),
            self._make_recommendation(),
            namespace="prod",
        )

        assert plan.ready is True
        actions = [s.action for s in plan.steps]
        assert "create_runtime" in actions

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_gpu_resource_profiles_applied(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        mock_runtime.return_value = ("vllm-runtime", [])
        mock_storage.return_value = ("oci://model", [])
        mock_preflight.return_value = ([], [])

        plan = await plan_deployment(
            self._make_server(),
            self._make_recommendation(),
            namespace="prod",
        )

        expected = GPU_RESOURCE_PROFILES["A100-80"]
        assert plan.resolved_params.cpu_request == expected[0]
        assert plan.resolved_params.cpu_limit == expected[1]
        assert plan.resolved_params.memory_request == expected[2]
        assert plan.resolved_params.memory_limit == expected[3]

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_recommendation_summary_includes_cost(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        mock_runtime.return_value = ("vllm-runtime", [])
        mock_storage.return_value = ("oci://model", [])
        mock_preflight.return_value = ([], [])

        plan = await plan_deployment(
            self._make_server(),
            self._make_recommendation(),
            namespace="prod",
        )

        assert plan.recommendation_summary["predicted_cost_month"] == 1200
        assert plan.recommendation_summary["meets_slo"] is True

    @patch("rhoai_mcp.composites.planner.deployment._run_preflight_checks")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_storage_uri")
    @patch("rhoai_mcp.composites.planner.deployment._resolve_runtime")
    async def test_tensor_parallel_warning(
        self,
        mock_runtime: MagicMock,
        mock_storage: MagicMock,
        mock_preflight: MagicMock,
    ) -> None:
        mock_runtime.return_value = ("vllm-runtime", [])
        mock_storage.return_value = ("oci://model", [])
        mock_preflight.return_value = (
            [],
            ["Tensor parallelism=2 requires NVLink/NVSwitch between GPUs"],
        )

        rec = self._make_recommendation()
        rec["gpu_config"]["tensor_parallel"] = 2

        plan = await plan_deployment(self._make_server(), rec, namespace="prod")

        assert any("NVLink" in w for w in plan.warnings)


class TestResolveRuntime:
    """Tests for _resolve_runtime."""

    async def test_override_returns_immediately(self) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_runtime

        runtime, issues = await _resolve_runtime(MagicMock(), "ns", "my-override")
        assert runtime == "my-override"
        assert issues == []

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_finds_vllm_runtime_by_name(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_runtime

        mock_cls.return_value.list_serving_runtimes.return_value = [
            {"name": "vllm-cuda-runtime", "supported_formats": ["pytorch"]},
        ]

        runtime, issues = await _resolve_runtime(MagicMock(), "ns", None)
        assert runtime == "vllm-cuda-runtime"
        assert issues == []

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_falls_back_to_format_match(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_runtime

        mock_cls.return_value.list_serving_runtimes.return_value = [
            {"name": "custom-runtime", "supported_formats": ["pytorch"]},
        ]

        runtime, issues = await _resolve_runtime(MagicMock(), "ns", None)
        assert runtime == "custom-runtime"

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_no_matching_runtime_returns_issue(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_runtime

        mock_cls.return_value.list_serving_runtimes.return_value = [
            {"name": "other-runtime", "supported_formats": ["onnx"]},
        ]

        runtime, issues = await _resolve_runtime(MagicMock(), "ns", None)
        assert runtime is None
        assert len(issues) == 1
        assert issues[0].category == "runtime"

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    async def test_exception_returns_non_blocking_issue(self, mock_cls: MagicMock) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_runtime

        mock_cls.return_value.list_serving_runtimes.side_effect = RuntimeError("fail")

        runtime, issues = await _resolve_runtime(MagicMock(), "ns", None)
        assert runtime is None
        assert len(issues) == 1
        assert issues[0].blocking is False


class TestResolveStorageUri:
    """Tests for _resolve_storage_uri."""

    async def test_override_returns_immediately(self) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_storage_uri

        uri, issues = await _resolve_storage_uri(MagicMock(), "model-id", None, "s3://bucket")
        assert uri == "s3://bucket"
        assert issues == []

    async def test_model_uri_from_recommendation(self) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_storage_uri

        uri, issues = await _resolve_storage_uri(
            MagicMock(), "model-id", "oci://registry/model", None
        )
        assert uri == "oci://registry/model"
        assert issues == []

    @patch("rhoai_mcp.domains.inference.tools._resolve_catalog_storage_uri")
    async def test_catalog_lookup_success(self, mock_catalog: MagicMock) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_storage_uri

        mock_catalog.return_value = "oci://catalog/model"
        server = MagicMock()

        uri, issues = await _resolve_storage_uri(server, "model-id", None, None)
        assert uri == "oci://catalog/model"
        assert issues == []

    @patch("rhoai_mcp.domains.inference.tools._resolve_catalog_storage_uri")
    async def test_catalog_returns_none_creates_blocking_issue(
        self, mock_catalog: MagicMock
    ) -> None:
        from rhoai_mcp.composites.planner.deployment import _resolve_storage_uri

        mock_catalog.return_value = None
        server = MagicMock()

        uri, issues = await _resolve_storage_uri(server, "model-id", None, None)
        assert uri is None
        assert len(issues) == 1
        assert issues[0].blocking is True
        assert issues[0].category == "storage"


class TestRunPreflightChecks:
    """Tests for _run_preflight_checks."""

    @patch("rhoai_mcp.domains.training.client.TrainingClient")
    @patch("rhoai_mcp.domains.projects.client.ProjectClient")
    async def test_namespace_missing_creates_blocking_issue(
        self, mock_proj_cls: MagicMock, mock_train_cls: MagicMock
    ) -> None:
        from rhoai_mcp.composites.planner.deployment import _run_preflight_checks

        project = MagicMock()
        project.metadata.name = "other-ns"
        mock_proj_cls.return_value.list_projects.return_value = [project]
        mock_train_cls.return_value.get_cluster_resources.return_value = MagicMock(
            has_gpus=False, gpu_info=None
        )

        issues, warnings = await _run_preflight_checks(
            MagicMock(), "missing-ns", "A100-80", 1, 1
        )

        assert any(i.category == "namespace" and i.blocking for i in issues)

    @patch("rhoai_mcp.domains.training.client.TrainingClient")
    @patch("rhoai_mcp.domains.projects.client.ProjectClient")
    async def test_no_gpus_creates_issue(
        self, mock_proj_cls: MagicMock, mock_train_cls: MagicMock
    ) -> None:
        from rhoai_mcp.composites.planner.deployment import _run_preflight_checks

        project = MagicMock()
        project.metadata.name = "prod"
        mock_proj_cls.return_value.list_projects.return_value = [project]
        mock_train_cls.return_value.get_cluster_resources.return_value = MagicMock(
            has_gpus=False, gpu_info=None
        )

        issues, warnings = await _run_preflight_checks(
            MagicMock(), "prod", "H100", 2, 1
        )

        assert any(i.category == "gpu" for i in issues)

    @patch("rhoai_mcp.domains.training.client.TrainingClient")
    @patch("rhoai_mcp.domains.projects.client.ProjectClient")
    async def test_gpu_type_not_on_cluster(
        self, mock_proj_cls: MagicMock, mock_train_cls: MagicMock
    ) -> None:
        from rhoai_mcp.composites.planner.deployment import _run_preflight_checks

        project = MagicMock()
        project.metadata.name = "prod"
        mock_proj_cls.return_value.list_projects.return_value = [project]

        gpu_info = MagicMock()
        gpu_info.products = ["NVIDIA-A100-SXM4-80GB"]
        gpu_info.available = 4
        mock_train_cls.return_value.get_cluster_resources.return_value = MagicMock(
            has_gpus=True, gpu_info=gpu_info
        )

        issues, warnings = await _run_preflight_checks(
            MagicMock(), "prod", "H100", 2, 1
        )

        assert any(i.category == "gpu" and "not found" in i.message for i in issues)

    @patch("rhoai_mcp.domains.training.client.TrainingClient")
    @patch("rhoai_mcp.domains.projects.client.ProjectClient")
    async def test_insufficient_gpus(
        self, mock_proj_cls: MagicMock, mock_train_cls: MagicMock
    ) -> None:
        from rhoai_mcp.composites.planner.deployment import _run_preflight_checks

        project = MagicMock()
        project.metadata.name = "prod"
        mock_proj_cls.return_value.list_projects.return_value = [project]

        gpu_info = MagicMock()
        gpu_info.products = ["NVIDIA-A100-80GB"]
        gpu_info.available = 1
        gpu_info.nodes_with_gpu = 2
        mock_train_cls.return_value.get_cluster_resources.return_value = MagicMock(
            has_gpus=True, gpu_info=gpu_info
        )

        issues, _ = await _run_preflight_checks(
            MagicMock(), "prod", "A100-80", 4, 1
        )

        assert any(i.category == "gpu" and "only" in i.message for i in issues)

    @patch("rhoai_mcp.domains.training.client.TrainingClient")
    @patch("rhoai_mcp.domains.projects.client.ProjectClient")
    async def test_tensor_parallel_adds_warning(
        self, mock_proj_cls: MagicMock, mock_train_cls: MagicMock
    ) -> None:
        from rhoai_mcp.composites.planner.deployment import _run_preflight_checks

        project = MagicMock()
        project.metadata.name = "prod"
        mock_proj_cls.return_value.list_projects.return_value = [project]
        mock_train_cls.return_value.get_cluster_resources.return_value = MagicMock(
            has_gpus=False, gpu_info=None
        )

        _, warnings = await _run_preflight_checks(
            MagicMock(), "prod", "H100", 4, 4
        )

        assert any("NVLink" in w for w in warnings)
