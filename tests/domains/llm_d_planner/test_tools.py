"""Tests for llm-d-planner MCP tools."""

from unittest.mock import MagicMock, patch

from rhoai_mcp.domains.llm_d_planner.models import DeploymentIntent
from rhoai_mcp.domains.llm_d_planner.tools import register_tools
from rhoai_mcp.utils.workflow_token import sign_step, verify_step


def _make_mock_mcp() -> MagicMock:
    """Create a mock FastMCP server that captures tool registrations."""
    mock = MagicMock()
    registered_tools: dict = {}

    def capture_tool():
        def decorator(f):
            registered_tools[f.__name__] = f
            return f

        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    return mock


def _make_mock_server() -> MagicMock:
    """Create a mock RHOAIServer."""
    server = MagicMock()
    server.config.planner_url = "http://localhost:8000"
    server.config.planner_timeout = 120
    return server


SAMPLE_INTENT = DeploymentIntent(
    use_case="chatbot_conversational",
    user_count=1000,
    experience_class="conversational",
    preferred_gpu_types=[],
    accuracy_priority="medium",
    cost_priority="medium",
    latency_priority="medium",
    complexity_priority="medium",
    domain_specialization=["general"],
)

SAMPLE_INTENT_DATA = {
    "use_case": "chatbot_conversational",
    "user_count": 1000,
    "experience_class": "conversational",
    "preferred_gpu_types": [],
    "accuracy_priority": "medium",
    "cost_priority": "medium",
    "latency_priority": "medium",
    "complexity_priority": "medium",
    "domain_specialization": ["general"],
    "additional_context": None,
}

SAMPLE_SLO_DEFAULTS = {
    "success": True,
    "slo_defaults": {
        "use_case": "chatbot_conversational",
        "ttft_ms": {"min": 50, "max": 200, "default": 150},
        "itl_ms": {"min": 20, "max": 80, "default": 65},
        "e2e_ms": {"min": 500, "max": 3000, "default": 2000},
    },
}

SAMPLE_WORKLOAD_PROFILE = {
    "success": True,
    "use_case": "chatbot_conversational",
    "workload_profile": {
        "prompt_tokens": 512,
        "output_tokens": 256,
        "peak_multiplier": 2.0,
        "distribution": "poisson",
        "active_fraction": 0.3,
        "requests_per_active_user_per_min": 2,
    },
}

SAMPLE_EXPECTED_RPS = {
    "success": True,
    "expected_rps": 10.0,
    "peak_rps": 20.0,
}

SAMPLE_RECOMMENDATION_RAW = {
    "model_id": "meta-llama/Llama-3.1-70B-Instruct",
    "model_name": "Llama 3.1 70B",
    "gpu_config": {"gpu_type": "NVIDIA-H100", "gpu_count": 2, "tensor_parallel": 2, "replicas": 1},
    "predicted_ttft_p95_ms": 140,
    "predicted_itl_p95_ms": 50,
    "predicted_e2e_p95_ms": 1200,
    "predicted_throughput_qps": 100.0,
    "cost_per_hour_usd": 3.98,
    "cost_per_month_usd": 2872.32,
    "meets_slo": True,
    "reasoning": "Selected Llama 3.1 70B for chatbot use case",
    "scores": {
        "accuracy_score": 78, "price_score": 65, "latency_score": 95,
        "complexity_score": 90, "balanced_score": 75.3, "slo_status": "compliant",
    },
}

SAMPLE_RANKED_RESPONSE = {
    "balanced": [SAMPLE_RECOMMENDATION_RAW],
    "lowest_cost": [SAMPLE_RECOMMENDATION_RAW],
    "lowest_latency": [SAMPLE_RECOMMENDATION_RAW],
    "total_configs_evaluated": 2847,
    "configs_after_filters": 542,
}

SAMPLE_SPEC_DATA = {
    "specification": {
        "use_case": "chatbot_conversational",
        "user_count": 1000,
        "slo_targets": {"ttft_ms": 150, "itl_ms": 65, "e2e_ms": 2000},
        "traffic_profile": {"prompt_tokens": 512, "output_tokens": 256, "expected_qps": 10.0},
        "preferred_gpu_types": [],
    },
}

SAMPLE_DEPLOY_RESPONSE = {
    "deployment_id": "chatbot-llama-3-1-70b-20260322143022",
    "namespace": "default",
    "files": {
        "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService",
        "autoscaling": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler",
        "servicemonitor": "apiVersion: monitoring.coreos.com/v1\nkind: ServiceMonitor",
    },
    "success": True,
    "message": "Deployment files generated successfully",
}

SAMPLE_MODELS_RECOMMENDED_DATA = {
    "specification": SAMPLE_SPEC_DATA["specification"],
    "recommendations": {
        "top_balanced": {"model": "Llama 3.1 70B", "gpu": "2x NVIDIA-H100"},
        "top_cost": {"model": "Llama 3.1 70B", "gpu": "2x NVIDIA-H100"},
        "top_performance": {"model": "Llama 3.1 70B", "gpu": "2x NVIDIA-H100"},
    },
    "_ranked_response": SAMPLE_RANKED_RESPONSE,
}


class TestExtractIntentTool:
    """Tests for extract_intent tool."""

    def test_tool_registration(self) -> None:
        """extract_intent tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "extract_intent" in mock_mcp._registered_tools

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_successful_extraction(self, mock_client_class: MagicMock) -> None:
        """Successful extraction returns intent fields and workflow token."""
        mock_client_class.return_value.extract_intent.return_value = SAMPLE_INTENT
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        extract_intent = mock_mcp._registered_tools["extract_intent"]

        result = extract_intent(text="I need a chatbot for 1000 users")

        assert result["use_case"] == "chatbot_conversational"
        assert result["user_count"] == 1000
        assert "workflow_token" in result
        # Verify token is valid and carries intent data
        data = verify_step(result["workflow_token"], "intent_extracted")
        assert "error" not in data
        assert data["use_case"] == "chatbot_conversational"

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_empty_text_returns_error(self, mock_client_class: MagicMock) -> None:
        """Empty text returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        extract_intent = mock_mcp._registered_tools["extract_intent"]

        result = extract_intent(text="")

        assert "error" in result
        assert "non-empty" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_text_exceeds_max_length(self, mock_client_class: MagicMock) -> None:
        """Text exceeding max length returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        extract_intent = mock_mcp._registered_tools["extract_intent"]

        result = extract_intent(text="x" * 4001)

        assert "error" in result
        assert "max length" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """Connection error returns error dict."""
        from rhoai_mcp.domains.llm_d_planner.client import PlannerConnectionError

        mock_client_class.return_value.extract_intent.side_effect = PlannerConnectionError(
            "llm-d-planner service unavailable at http://localhost:8000"
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        extract_intent = mock_mcp._registered_tools["extract_intent"]

        result = extract_intent(text="I need a chatbot")

        assert "error" in result
        assert "unavailable" in result["error"].lower()

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """API error returns error dict with status code."""
        from rhoai_mcp.domains.llm_d_planner.client import PlannerAPIError

        mock_client_class.return_value.extract_intent.side_effect = PlannerAPIError(
            status_code=500,
            detail="Internal server error"
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        extract_intent = mock_mcp._registered_tools["extract_intent"]

        result = extract_intent(text="I need a chatbot")

        assert "error" in result
        assert "api error" in result["error"].lower()
        assert result["status_code"] == 500


class TestPrepareModelTechSpecs:
    """Tests for prepare_model_tech_specs tool."""

    def test_tool_registration(self) -> None:
        """prepare_model_tech_specs tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "prepare_model_tech_specs" in mock_mcp._registered_tools

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_successful_spec_building(self, mock_client_class: MagicMock) -> None:
        """Builds specification from intent token and API defaults."""
        mock_client = mock_client_class.return_value
        mock_client.get_slo_defaults.return_value = SAMPLE_SLO_DEFAULTS
        mock_client.get_workload_profile.return_value = SAMPLE_WORKLOAD_PROFILE
        mock_client.get_expected_rps.return_value = SAMPLE_EXPECTED_RPS

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        prepare = mock_mcp._registered_tools["prepare_model_tech_specs"]

        token = sign_step("intent_extracted", SAMPLE_INTENT_DATA)
        result = prepare(workflow_token=token)

        assert "error" not in result
        spec = result["specification"]
        assert spec["use_case"] == "chatbot_conversational"
        assert spec["user_count"] == 1000
        assert spec["slo_targets"]["ttft_ms"] == 150
        assert spec["traffic_profile"]["prompt_tokens"] == 512
        assert "workflow_token" in result
        # Verify output token carries spec data
        data = verify_step(result["workflow_token"], "specs_prepared")
        assert "error" not in data

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_user_overrides_applied(self, mock_client_class: MagicMock) -> None:
        """User overrides replace intent/default values."""
        mock_client = mock_client_class.return_value
        mock_client.get_slo_defaults.return_value = SAMPLE_SLO_DEFAULTS
        mock_client.get_workload_profile.return_value = SAMPLE_WORKLOAD_PROFILE
        mock_client.get_expected_rps.return_value = SAMPLE_EXPECTED_RPS

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        prepare = mock_mcp._registered_tools["prepare_model_tech_specs"]

        token = sign_step("intent_extracted", SAMPLE_INTENT_DATA)
        result = prepare(
            workflow_token=token,
            user_count=5000,
            ttft_max_ms=100,
        )

        assert "error" not in result
        spec = result["specification"]
        assert spec["user_count"] == 5000  # overridden
        assert spec["slo_targets"]["ttft_ms"] == 100  # overridden
        assert spec["slo_targets"]["itl_ms"] == 65  # default

    def test_missing_workflow_token(self) -> None:
        """Missing workflow_token returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        prepare = mock_mcp._registered_tools["prepare_model_tech_specs"]

        result = prepare(workflow_token="")

        assert "error" in result

    def test_wrong_step_token(self) -> None:
        """Token from wrong step returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        prepare = mock_mcp._registered_tools["prepare_model_tech_specs"]

        token = sign_step("specs_prepared", {"some": "data"})
        result = prepare(workflow_token=token)

        assert "error" in result
        assert "Wrong workflow order" in result["error"]


class TestGetRecommendedModels:
    """Tests for get_recommended_models tool."""

    def test_tool_registration(self) -> None:
        """get_recommended_models tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "get_recommended_models" in mock_mcp._registered_tools

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_successful_recommendations(self, mock_client_class: MagicMock) -> None:
        """Returns recommendations from specification."""
        mock_client_class.return_value.get_recommendations.return_value = SAMPLE_RANKED_RESPONSE
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_recs = mock_mcp._registered_tools["get_recommended_models"]

        token = sign_step("specs_prepared", SAMPLE_SPEC_DATA)
        result = get_recs(workflow_token=token)

        assert "error" not in result
        assert "recommendations" in result
        assert "specification" in result
        recs = result["recommendations"]
        assert "top_balanced" in recs
        assert recs["top_balanced"]["model"] == "Llama 3.1 70B"
        assert "workflow_token" in result
        data = verify_step(result["workflow_token"], "models_recommended")
        assert "error" not in data

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_with_optimization_profile(self, mock_client_class: MagicMock) -> None:
        """Optimization profile is resolved to weights."""
        mock_client_class.return_value.get_recommendations.return_value = SAMPLE_RANKED_RESPONSE
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_recs = mock_mcp._registered_tools["get_recommended_models"]

        token = sign_step("specs_prepared", SAMPLE_SPEC_DATA)
        result = get_recs(workflow_token=token, optimization_profile="optimize_cost")

        assert "error" not in result
        call_kwargs = mock_client_class.return_value.get_recommendations.call_args.kwargs
        assert call_kwargs["weights"] == {"accuracy": 2, "price": 8, "latency": 1, "complexity": 1}

    def test_missing_workflow_token(self) -> None:
        """Missing workflow_token returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_recs = mock_mcp._registered_tools["get_recommended_models"]

        result = get_recs(workflow_token="")

        assert "error" in result

    def test_invalid_optimization_profile(self) -> None:
        """Invalid optimization profile returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_recs = mock_mcp._registered_tools["get_recommended_models"]

        token = sign_step("specs_prepared", SAMPLE_SPEC_DATA)
        result = get_recs(workflow_token=token, optimization_profile="invalid")

        assert "error" in result
        assert "optimization_profile" in result["error"]


class TestGetDeploymentConfig:
    """Tests for get_deployment_config tool."""

    def test_tool_registration(self) -> None:
        """get_deployment_config tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "get_deployment_config" in mock_mcp._registered_tools

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_successful_deployment_config(self, mock_client_class: MagicMock) -> None:
        """Generates deployment configs from recommendations."""
        # Mock the deploy response to return ml-prod namespace
        deploy_response = SAMPLE_DEPLOY_RESPONSE.copy()
        deploy_response["namespace"] = "ml-prod"
        mock_client_class.return_value.deploy.return_value = deploy_response
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        token = sign_step("models_recommended", SAMPLE_MODELS_RECOMMENDED_DATA)
        result = get_config(workflow_token=token, category="balanced", namespace="ml-prod")

        assert "error" not in result
        assert result["deployment_id"] == "chatbot-llama-3-1-70b-20260322143022"
        assert result["namespace"] == "ml-prod"
        assert "inferenceservice" in result["configs"]
        # Terminal step — no workflow_token in output
        assert "workflow_token" not in result

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_invalid_category(self, mock_client_class: MagicMock) -> None:
        """Invalid category returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        token = sign_step("models_recommended", SAMPLE_MODELS_RECOMMENDED_DATA)
        result = get_config(workflow_token=token, category="fastest", namespace="default")

        assert "error" in result
        assert "category" in result["error"]
        mock_client_class.assert_not_called()

    def test_missing_workflow_token(self) -> None:
        """Missing workflow_token returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(workflow_token="", category="balanced", namespace="default")

        assert "error" in result

    @patch("rhoai_mcp.domains.llm_d_planner.tools.PlannerClient")
    def test_empty_namespace(self, mock_client_class: MagicMock) -> None:
        """Empty namespace returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        token = sign_step("models_recommended", SAMPLE_MODELS_RECOMMENDED_DATA)
        result = get_config(workflow_token=token, category="balanced", namespace="  ")

        assert "error" in result
        assert "namespace" in result["error"]
        mock_client_class.assert_not_called()
