"""Tests for NeuralNav recommend_model MCP tool."""

from unittest.mock import MagicMock, patch

from rhoai_mcp.composites.neuralnav.models import (
    DeploymentConfigResult,
    ModelRecommendation,
    RecommendationResult,
)
from rhoai_mcp.composites.neuralnav.tools import register_tools


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
    server.config.neuralnav_url = "http://localhost:8000"
    server.config.neuralnav_timeout = 120
    return server


SAMPLE_REC = ModelRecommendation(
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    model_name="Llama 3.1 70B",
    gpu_config={
        "gpu_type": "NVIDIA-H100",
        "gpu_count": 2,
        "tensor_parallel": 2,
        "replicas": 1,
    },
    predicted_ttft_p95_ms=140,
    predicted_itl_p95_ms=50,
    predicted_e2e_p95_ms=1200,
    predicted_throughput_qps=100.0,
    cost_per_hour_usd=3.98,
    cost_per_month_usd=2872.32,
    meets_slo=True,
    reasoning="Selected for chatbot",
    scores={
        "accuracy_score": 78,
        "price_score": 65,
        "latency_score": 95,
        "complexity_score": 90,
        "balanced_score": 75.3,
        "slo_status": "compliant",
    },
)

SAMPLE_RESULT = RecommendationResult(
    specification={
        "use_case": "chatbot_conversational",
        "user_count": 1000,
        "slo_targets": {"ttft_ms": 150, "itl_ms": 65, "e2e_ms": 2000},
        "traffic_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256,
            "expected_qps": 10.0,
        },
    },
    top_performance=SAMPLE_REC,
    top_cost=SAMPLE_REC,
    top_balanced=SAMPLE_REC,
    total_configs_evaluated=2847,
    configs_after_filters=542,
)

SAMPLE_CONFIG_RESULT = DeploymentConfigResult(
    deployment_id="chatbot-llama-3-1-70b-20260322143022",
    namespace="default",
    model_name="Llama 3.1 70B",
    configs={
        "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService",
        "autoscaling": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler",
        "servicemonitor": "apiVersion: monitoring.coreos.com/v1\nkind: ServiceMonitor",
    },
)


class TestRecommendModelTool:
    """Tests for recommend_model tool."""

    def test_tool_registration(self) -> None:
        """recommend_model tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "recommend_model" in mock_mcp._registered_tools

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_successful_recommendation(self, mock_client_class: MagicMock) -> None:
        """Successful recommendation returns formatted result."""
        mock_client_class.return_value.recommend.return_value = SAMPLE_RESULT
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot for 1000 users")

        assert "specification" in result
        assert "recommendations" in result
        recs = result["recommendations"]
        assert "top_balanced" in recs
        assert recs["top_balanced"]["model"] == "Llama 3.1 70B"
        assert recs["top_balanced"]["score"] == 75.3
        assert "top_performance" in recs
        assert "top_cost" in recs

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_with_overrides(self, mock_client_class: MagicMock) -> None:
        """Overrides are passed to the client."""
        mock_client_class.return_value.recommend.return_value = SAMPLE_RESULT
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        recommend_model(
            text="I need a model",
            use_case="code_completion",
            user_count=5000,
            preferred_gpu_types=["H100"],
        )

        mock_client_class.return_value.recommend.assert_called_once_with(
            "I need a model",
            use_case_override="code_completion",
            user_count_override=5000,
            gpu_types_override=["H100"],
            ttft_override_ms=None,
            itl_override_ms=None,
            e2e_override_ms=None,
            min_accuracy=None,
            max_cost=None,
            weights=None,
            percentile=None,
        )

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """Connection error returns error dict."""
        from rhoai_mcp.composites.neuralnav.client import NeuralNavConnectionError

        mock_client_class.return_value.recommend.side_effect = NeuralNavConnectionError(
            "Neural Navigator service unavailable at http://localhost:8000"
        )
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert "error" in result
        assert "unavailable" in result["error"].lower()
        assert "hint" in result

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """API error returns error dict with status code."""
        from rhoai_mcp.composites.neuralnav.client import NeuralNavAPIError

        mock_client_class.return_value.recommend.side_effect = NeuralNavAPIError(
            status_code=500,
            detail="Internal Server Error",
        )
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert "error" in result
        assert result["status_code"] == 500

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_with_slo_overrides(self, mock_client_class: MagicMock) -> None:
        """SLO override parameters are passed to the client."""
        mock_client_class.return_value.recommend.return_value = SAMPLE_RESULT
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        recommend_model(
            text="I need a chatbot",
            ttft_max_ms=100,
            itl_max_ms=30,
            e2e_max_ms=1500,
        )

        mock_client_class.return_value.recommend.assert_called_once_with(
            "I need a chatbot",
            use_case_override=None,
            user_count_override=None,
            gpu_types_override=None,
            ttft_override_ms=100,
            itl_override_ms=30,
            e2e_override_ms=1500,
            min_accuracy=None,
            max_cost=None,
            weights=None,
            percentile=None,
        )

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_with_optimization_profile(self, mock_client_class: MagicMock) -> None:
        """Optimization profile is resolved to weights dict."""
        mock_client_class.return_value.recommend.return_value = SAMPLE_RESULT
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        recommend_model(
            text="I need a chatbot",
            optimization_profile="optimize_latency",
        )

        call_kwargs = mock_client_class.return_value.recommend.call_args.kwargs
        assert call_kwargs["weights"] == {"accuracy": 2, "price": 2, "latency": 8, "complexity": 1}

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_with_all_constraints(self, mock_client_class: MagicMock) -> None:
        """All constraint parameters are forwarded to the client."""
        mock_client_class.return_value.recommend.return_value = SAMPLE_RESULT
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        recommend_model(
            text="I need a chatbot",
            use_case="code_completion",
            user_count=5000,
            preferred_gpu_types=["H100"],
            ttft_max_ms=100,
            itl_max_ms=30,
            e2e_max_ms=1500,
            min_accuracy=70,
            max_cost_per_month=5000.0,
            optimization_profile="optimize_cost",
            percentile="p99",
        )

        mock_client_class.return_value.recommend.assert_called_once_with(
            "I need a chatbot",
            use_case_override="code_completion",
            user_count_override=5000,
            gpu_types_override=["H100"],
            ttft_override_ms=100,
            itl_override_ms=30,
            e2e_override_ms=1500,
            min_accuracy=70,
            max_cost=5000.0,
            weights={"accuracy": 2, "price": 8, "latency": 1, "complexity": 1},
            percentile="p99",
        )

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_optimization_profile(self, mock_client_class: MagicMock) -> None:
        """Invalid optimization profile returns error dict."""
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(
            text="I need a chatbot",
            optimization_profile="invalid_profile",
        )

        assert "error" in result
        assert "optimization_profile" in result["error"]
        # Client should NOT have been called
        mock_client_class.assert_not_called()
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_use_case(self, mock_client_class: MagicMock) -> None:
        """Invalid use_case returns error dict without calling the client."""
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(
            text="I need a chatbot",
            use_case="document_summarization",
        )

        assert "error" in result
        assert "use_case" in result["error"]
        assert "document_summarization" in result["error"]
        # Client should NOT have been called
        mock_client_class.assert_not_called()
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_valid_use_case_accepted(self, mock_client_class: MagicMock) -> None:
        """Valid use_case is passed through to the client."""
        mock_client_class.return_value.recommend.return_value = SAMPLE_RESULT
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(
            text="I need a chatbot",
            use_case="chatbot_conversational",
        )

        assert "error" not in result
        call_kwargs = mock_client_class.return_value.recommend.call_args.kwargs
        assert call_kwargs["use_case_override"] == "chatbot_conversational"

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_percentile(self, mock_client_class: MagicMock) -> None:
        """Invalid percentile returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", percentile="p50")

        assert "error" in result
        assert "percentile" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_user_count(self, mock_client_class: MagicMock) -> None:
        """Non-positive user_count returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", user_count=0)

        assert "error" in result
        assert "user_count" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_slo_target(self, mock_client_class: MagicMock) -> None:
        """Non-positive SLO target returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", ttft_max_ms=-1)

        assert "error" in result
        assert "ttft_max_ms" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_min_accuracy(self, mock_client_class: MagicMock) -> None:
        """Out-of-range min_accuracy returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", min_accuracy=101)

        assert "error" in result
        assert "min_accuracy" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_max_cost(self, mock_client_class: MagicMock) -> None:
        """Negative max_cost_per_month returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", max_cost_per_month=-100.0)

        assert "error" in result
        assert "max_cost_per_month" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_gpu_types(self, mock_client_class: MagicMock) -> None:
        """Invalid GPU types return error listing valid options."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", preferred_gpu_types=["H100", "V100"])

        assert "error" in result
        assert "V100" in result["error"]
        assert "preferred_gpu_types" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_empty_recommendations(self, mock_client_class: MagicMock) -> None:
        """Empty recommendations returns message."""
        empty_result = RecommendationResult(
            specification={
                "use_case": "chatbot_conversational",
                "user_count": 1000,
                "slo_targets": {},
                "traffic_profile": {},
            },
            total_configs_evaluated=2847,
            configs_after_filters=0,
        )
        mock_client_class.return_value.recommend.return_value = empty_result
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert result["recommendations"] == {}
        assert "message" in result

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_partial_recommendations(self, mock_client_class: MagicMock) -> None:
        """When some categories are None, only populated ones appear."""
        partial_result = RecommendationResult(
            specification={
                "use_case": "chatbot_conversational",
                "user_count": 1000,
                "slo_targets": {},
                "traffic_profile": {},
            },
            top_balanced=SAMPLE_REC,
            # top_performance and top_cost are None
            total_configs_evaluated=2847,
            configs_after_filters=100,
        )
        mock_client_class.return_value.recommend.return_value = partial_result
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert "top_balanced" in result["recommendations"]
        assert "top_performance" not in result["recommendations"]
        assert "top_cost" not in result["recommendations"]
        assert "message" not in result

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_empty_text_returns_error(self, mock_client_class: MagicMock) -> None:
        """Empty text returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="")

        assert "error" in result
        assert "non-empty" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_whitespace_text_returns_error(self, mock_client_class: MagicMock) -> None:
        """Whitespace-only text returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="   \n\t  ")

        assert "error" in result
        assert "non-empty" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_text_exceeds_max_length(self, mock_client_class: MagicMock) -> None:
        """Text exceeding max length returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="x" * 4001)

        assert "error" in result
        assert "max length" in result["error"]
        mock_client_class.assert_not_called()


class TestDeploymentConfigTool:
    """Tests for get_deployment_config tool."""

    def test_tool_registration(self) -> None:
        """get_deployment_config tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "get_deployment_config" in mock_mcp._registered_tools

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_successful_deployment_config(self, mock_client_class: MagicMock) -> None:
        """Successful config generation returns formatted result."""
        mock_client_class.return_value.generate_config.return_value = SAMPLE_CONFIG_RESULT
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["deployment_id"] == "chatbot-llama-3-1-70b-20260322143022"
        assert result["namespace"] == "default"
        assert result["model"] == "Llama 3.1 70B"
        assert "inferenceservice" in result["configs"]
        assert "autoscaling" in result["configs"]
        assert "servicemonitor" in result["configs"]

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_deployment_config_no_model_key(self, mock_client_class: MagicMock) -> None:
        """When model_name is None, model key is omitted from output."""
        no_model = DeploymentConfigResult(
            deployment_id="chatbot-unknown-20260322",
            namespace="default",
            configs={"inferenceservice": "yaml-content"},
        )
        mock_client_class.return_value.generate_config.return_value = no_model
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "model" not in result
        assert result["deployment_id"] == "chatbot-unknown-20260322"

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_deployment_config_model_name_fallback(self, mock_client_class: MagicMock) -> None:
        """When model_name is a model_id fallback value, it is passed through."""
        fallback_result = DeploymentConfigResult(
            deployment_id="chatbot-llama-20260322",
            namespace="default",
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            configs={"inferenceservice": "yaml-content"},
        )
        mock_client_class.return_value.generate_config.return_value = fallback_result
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["model"] == "meta-llama/Llama-3.1-70B-Instruct"

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_empty_category_result(self, mock_client_class: MagicMock) -> None:
        """When no recommendations exist for category, returns error dict."""
        from rhoai_mcp.composites.neuralnav.client import NeuralNavAPIError

        mock_client_class.return_value.generate_config.side_effect = NeuralNavAPIError(
            status_code=404, detail="No recommendation found for category 'cost'"
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="cost",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert result["status_code"] == 404

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_category(self, mock_client_class: MagicMock) -> None:
        """Invalid category returns error dict."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="fastest",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "category" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_use_case(self, mock_client_class: MagicMock) -> None:
        """Invalid use_case returns error dict."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="invalid_case",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "use_case" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_user_count(self, mock_client_class: MagicMock) -> None:
        """user_count <= 0 returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=0,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "user_count" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_slo_targets(self, mock_client_class: MagicMock) -> None:
        """SLO targets <= 0 return error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=0,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "ttft_target_ms" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_token_counts(self, mock_client_class: MagicMock) -> None:
        """prompt/output tokens <= 0 return error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=-1,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "prompt_tokens" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_expected_qps(self, mock_client_class: MagicMock) -> None:
        """expected_qps <= 0 returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "expected_qps" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_optimization_profile(self, mock_client_class: MagicMock) -> None:
        """Invalid optimization_profile returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            optimization_profile="turbo",
        )

        assert "error" in result
        assert "optimization_profile" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_gpu_types(self, mock_client_class: MagicMock) -> None:
        """Invalid GPU types return error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            preferred_gpu_types=["V100"],
        )

        assert "error" in result
        assert "V100" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_min_accuracy(self, mock_client_class: MagicMock) -> None:
        """Out-of-range min_accuracy returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            min_accuracy=101,
        )

        assert "error" in result
        assert "min_accuracy" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_max_cost(self, mock_client_class: MagicMock) -> None:
        """Negative max_cost_per_month returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            max_cost_per_month=-50.0,
        )

        assert "error" in result
        assert "max_cost_per_month" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_invalid_percentile(self, mock_client_class: MagicMock) -> None:
        """Invalid percentile returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            percentile="p50",
        )

        assert "error" in result
        assert "percentile" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """NeuralNav connection error returns error dict."""
        from rhoai_mcp.composites.neuralnav.client import NeuralNavConnectionError

        mock_client_class.return_value.generate_config.side_effect = NeuralNavConnectionError(
            "Neural Navigator service unavailable at http://localhost:8000"
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "unavailable" in result["error"].lower()
        assert "hint" in result

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """NeuralNav API error returns error dict with status code."""
        from rhoai_mcp.composites.neuralnav.client import NeuralNavAPIError

        mock_client_class.return_value.generate_config.side_effect = NeuralNavAPIError(
            status_code=500, detail="Internal Server Error"
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert result["status_code"] == 500

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_empty_namespace(self, mock_client_class: MagicMock) -> None:
        """Empty namespace returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        result = get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            namespace="  ",
        )

        assert "error" in result
        assert "namespace" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.neuralnav.tools.NeuralNavClient")
    def test_optimization_profile_resolved_to_weights(self, mock_client_class: MagicMock) -> None:
        """optimization_profile is resolved to weights dict before calling client."""
        mock_client_class.return_value.generate_config.return_value = SAMPLE_CONFIG_RESULT
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        get_config = mock_mcp._registered_tools["get_deployment_config"]

        get_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            optimization_profile="optimize_cost",
        )

        call_kwargs = mock_client_class.return_value.generate_config.call_args.kwargs
        assert call_kwargs["weights"] == {"accuracy": 2, "price": 8, "latency": 1, "complexity": 1}
