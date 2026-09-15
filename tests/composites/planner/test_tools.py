"""Tests for Planner recommend_model MCP tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from rhoai_mcp.composites.planner.models import (
    DeploymentConfigResult,
    ModelRecommendation,
    RecommendationResult,
)
from rhoai_mcp.composites.planner.tools import register_tools
from rhoai_mcp.config import PlannerMode


def _make_mock_mcp() -> MagicMock:
    """Create a mock FastMCP server that captures tool registrations."""
    mock = MagicMock()
    registered_tools: dict = {}
    registered_async_tools: dict = {}

    def capture_tool():
        def decorator(f):
            def sync_tool(*args, **kwargs):
                return asyncio.run(f(*args, **kwargs))

            registered_tools[f.__name__] = sync_tool
            registered_async_tools[f.__name__] = f
            return f

        return decorator

    mock.tool = capture_tool
    mock._registered_tools = registered_tools
    mock._registered_async_tools = registered_async_tools
    return mock


def _make_mock_server(mode: PlannerMode = PlannerMode.REMOTE) -> MagicMock:
    """Create a mock RHOAIServer."""
    server = MagicMock()
    server.config.planner_mode = mode
    server.config.planner_url = "http://localhost:8000"
    server.config.planner_timeout = 120
    server.config.planner_model_catalog_url = None
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
        "quality_score": 78,
        "price_score": 65,
        "latency_score": 95,
        "balanced_score": 75.3,
        "slo_status": "compliant",
    },
)

SAMPLE_RESULT = RecommendationResult(
    specification={
        "use_case": "chatbot_conversational",
        "user_count": 1000,
        "slo_targets": {"ttft_target_ms": 150, "itl_target_ms": 65, "e2e_target_ms": 2000},
        "traffic_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256,
            "expected_qps": 10.0,
        },
    },
    top_performance=SAMPLE_REC,
    top_cost=SAMPLE_REC,
    top_balanced=SAMPLE_REC,
    top_quality=SAMPLE_REC,
    total_configs_evaluated=2847,
    configs_after_filters=542,
)

SAMPLE_CONFIG_RESULT = DeploymentConfigResult(
    deployment_id="chatbot-llama-3-1-70b-20260322143022",
    namespace="default",
    model_name="Llama 3.1 70B",
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    model_uri="oci://quay.io/rhoai/llama-3-1-70b:latest",
    gpu_config={
        "gpu_type": "NVIDIA-H100",
        "gpu_count": 2,
        "tensor_parallel": 2,
        "replicas": 1,
    },
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

    @patch("rhoai_mcp.composites.planner.tools.LocalPlannerClient")
    async def test_concurrent_local_requests_initialize_one_client(
        self, mock_client_class: MagicMock
    ) -> None:
        """Concurrent local requests share one asynchronously initialized client."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server(PlannerMode.LOCAL))
        recommend_model = mock_mcp._registered_async_tools["recommend_model"]

        await asyncio.gather(
            recommend_model(
                text="I need a chatbot",
                use_case="chatbot_conversational",
                user_count=1000,
                preferred_gpu_types=["H100"],
            ),
            recommend_model(
                text="I need a chatbot",
                use_case="chatbot_conversational",
                user_count=1000,
                preferred_gpu_types=["H100"],
            ),
        )

        # asyncio.Lock serialises inside _get_client(); only one LocalPlannerClient is created
        mock_client_class.assert_called_once_with(model_catalog_url=None)

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_successful_recommendation(self, mock_client_class: MagicMock) -> None:
        """Successful recommendation returns formatted result."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
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
        assert recs["top_balanced"]["model_id"] == "meta-llama/Llama-3.1-70B-Instruct"
        assert recs["top_balanced"]["score"] == 75.3
        assert recs["top_balanced"]["ttft_p95_ms"] == 140
        assert recs["top_balanced"]["e2e_p95_ms"] == 1200
        assert recs["top_balanced"]["quality_score"] == 78
        assert recs["top_balanced"]["cost_usd_month"] == 2872.32
        assert "top_performance" in recs
        assert recs["top_performance"]["ttft_p95_ms"] == 140
        assert recs["top_performance"]["quality_score"] == 78
        assert "top_cost" in recs
        assert recs["top_cost"]["cost_usd_month"] == 2872.32
        assert "top_quality" in recs
        assert recs["top_quality"]["score"] == 78
        assert recs["top_quality"]["quality_score"] == 78

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_with_overrides(self, mock_client_class: MagicMock) -> None:
        """Overrides are passed to the client."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
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
            min_quality=None,
            max_cost=None,
            percentile_override=None,
            priority_weights=None,
        )

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """Connection error returns error dict."""
        from rhoai_mcp.composites.planner.client import PlannerConnectionError

        mock_client_class.return_value.recommend = AsyncMock(side_effect=PlannerConnectionError(
            "Planner service unavailable at http://localhost:8000"
        ))
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert "error" in result
        assert "unavailable" in result["error"].lower()
        assert "hint" in result

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """API error returns error dict with status code."""
        from rhoai_mcp.composites.planner.client import PlannerAPIError

        mock_client_class.return_value.recommend = AsyncMock(side_effect=PlannerAPIError(
            status_code=500,
            detail="Internal Server Error",
        ))
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert "error" in result
        assert result["status_code"] == 500

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_with_slo_overrides(self, mock_client_class: MagicMock) -> None:
        """SLO override parameters are passed to the client."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
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
            min_quality=None,
            max_cost=None,
            percentile_override=None,
            priority_weights=None,
        )

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_valid_optimization_profile_accepted(self, mock_client_class: MagicMock) -> None:
        """A valid optimization_profile passes validation and the client is called."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        recommend_model(
            text="I need a chatbot",
            optimization_profile="optimize_latency",
        )

        mock_client_class.return_value.recommend.assert_called_once()
        call_kwargs = mock_client_class.return_value.recommend.call_args.kwargs
        assert call_kwargs["priority_weights"] == {"quality": 2, "price": 2, "latency": 8}

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_with_all_constraints(self, mock_client_class: MagicMock) -> None:
        """All constraint parameters are forwarded to the client."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
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
            min_quality=70,
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
            min_quality=70,
            max_cost=5000.0,
            percentile_override="p99",
            priority_weights={"quality": 2, "price": 8, "latency": 1},
        )

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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
        mock_client_class.assert_not_called()
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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
        mock_client_class.assert_not_called()
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_valid_use_case_accepted(self, mock_client_class: MagicMock) -> None:
        """Valid use_case is passed through to the client."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_percentile(self, mock_client_class: MagicMock) -> None:
        """Invalid percentile returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", percentile="p50")

        assert "error" in result
        assert "percentile" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_user_count(self, mock_client_class: MagicMock) -> None:
        """Non-positive user_count returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", user_count=0)

        assert "error" in result
        assert "user_count" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_slo_target(self, mock_client_class: MagicMock) -> None:
        """Non-positive SLO target returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", ttft_max_ms=-1)

        assert "error" in result
        assert "ttft_max_ms" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_min_quality(self, mock_client_class: MagicMock) -> None:
        """Out-of-range min_quality returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", min_quality=101)

        assert "error" in result
        assert "min_quality" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_max_cost(self, mock_client_class: MagicMock) -> None:
        """Negative max_cost_per_month returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="chatbot", max_cost_per_month=-100.0)

        assert "error" in result
        assert "max_cost_per_month" in result["error"]
        mock_client_class.return_value.recommend.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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
        mock_client_class.return_value.recommend = AsyncMock(return_value=empty_result)
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert result["recommendations"] == {}
        assert "message" in result

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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
            total_configs_evaluated=2847,
            configs_after_filters=100,
        )
        mock_client_class.return_value.recommend = AsyncMock(return_value=partial_result)
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server()

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="I need a chatbot")

        assert "top_balanced" in result["recommendations"]
        assert "top_performance" not in result["recommendations"]
        assert "top_cost" not in result["recommendations"]
        assert "top_quality" not in result["recommendations"]
        assert "message" not in result

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_empty_text_returns_error(self, mock_client_class: MagicMock) -> None:
        """Empty text returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="")

        assert "error" in result
        assert "non-empty" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_whitespace_text_returns_error(self, mock_client_class: MagicMock) -> None:
        """Whitespace-only text returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        result = recommend_model(text="   \n\t  ")

        assert "error" in result
        assert "non-empty" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_successful_deployment_config(self, mock_client_class: MagicMock) -> None:
        """Successful config generation returns formatted result."""
        mock_client_class.return_value.generate_config = AsyncMock(return_value=SAMPLE_CONFIG_RESULT)
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_deployment_config_no_model_key(self, mock_client_class: MagicMock) -> None:
        """When model_name is None, model key is omitted from output."""
        no_model = DeploymentConfigResult(
            deployment_id="chatbot-unknown-20260322",
            namespace="default",
            configs={"inferenceservice": "yaml-content"},
        )
        mock_client_class.return_value.generate_config = AsyncMock(return_value=no_model)
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_deployment_config_model_name_fallback(self, mock_client_class: MagicMock) -> None:
        """When model_name is a model_id fallback value, it is passed through."""
        fallback_result = DeploymentConfigResult(
            deployment_id="chatbot-llama-20260322",
            namespace="default",
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            configs={"inferenceservice": "yaml-content"},
        )
        mock_client_class.return_value.generate_config = AsyncMock(return_value=fallback_result)
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_empty_category_result(self, mock_client_class: MagicMock) -> None:
        """When no recommendations exist for category, returns error dict."""
        from rhoai_mcp.composites.planner.client import PlannerAPIError

        mock_client_class.return_value.generate_config = AsyncMock(side_effect=PlannerAPIError(
            status_code=404, detail="No recommendation found for category 'cost'"
        ))
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_min_quality(self, mock_client_class: MagicMock) -> None:
        """Out-of-range min_quality returns error."""
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
            min_quality=101,
        )

        assert "error" in result
        assert "min_quality" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """Planner connection error returns error dict."""
        from rhoai_mcp.composites.planner.client import PlannerConnectionError

        mock_client_class.return_value.generate_config = AsyncMock(side_effect=PlannerConnectionError(
            "Planner service unavailable at http://localhost:8000"
        ))
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """Planner API error returns error dict with status code."""
        from rhoai_mcp.composites.planner.client import PlannerAPIError

        mock_client_class.return_value.generate_config = AsyncMock(side_effect=PlannerAPIError(
            status_code=500, detail="Internal Server Error"
        ))
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
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

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_valid_optimization_profile_accepted(self, mock_client_class: MagicMock) -> None:
        """A valid optimization_profile passes validation and the client is called."""
        mock_client_class.return_value.generate_config = AsyncMock(return_value=SAMPLE_CONFIG_RESULT)
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

        mock_client_class.return_value.generate_config.assert_called_once()
        call_kwargs = mock_client_class.return_value.generate_config.call_args.kwargs
        assert call_kwargs["priority_weights"] == {"quality": 2, "price": 8, "latency": 1}


class TestGetUseCaseDefaultsTool:
    """Tests for get_use_case_defaults tool."""

    def test_tool_registration(self) -> None:
        """get_use_case_defaults tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "get_use_case_defaults" in mock_mcp._registered_tools

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_successful_response(self, mock_client_class: MagicMock) -> None:
        """Merges SLO and workload data into a single clean response."""
        mock_client_class.return_value.get_slo_defaults = AsyncMock(return_value={
            "success": True,
            "slo_defaults": {
                "use_case": "chatbot_conversational",
                "description": "Conversational AI chatbot",
                "ttft_ms": {"min": 50, "max": 500, "default": 388},
                "itl_ms": {"min": 10, "max": 100, "default": 78},
                "e2e_ms": {"min": 500, "max": 10000, "default": 7875},
            },
        })
        mock_client_class.return_value.get_workload_profile = AsyncMock(return_value={
            "success": True,
            "use_case": "chatbot_conversational",
            "description": "Conversational AI chatbot",
            "workload_profile": {
                "prompt_tokens": 512,
                "output_tokens": 256,
                "active_fraction": 0.1,
                "requests_per_active_user_per_min": 2.0,
            },
        })
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_use_case_defaults"]

        result = tool(use_case="chatbot_conversational")

        assert "error" not in result
        assert result["use_case"] == "chatbot_conversational"
        assert result["description"] == "Conversational AI chatbot"
        assert result["slo_targets"]["ttft_ms"]["default"] == 388
        assert result["workload"]["prompt_tokens"] == 512
        assert result["workload"]["active_fraction"] == 0.1

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_use_case(self, mock_client_class: MagicMock) -> None:
        """Invalid use_case returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_use_case_defaults"]

        result = tool(use_case="invalid_use_case")

        assert "error" in result
        assert "use_case" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """Connection error returns error dict with hint."""
        from rhoai_mcp.composites.planner.client import PlannerConnectionError

        mock_client_class.return_value.get_slo_defaults = AsyncMock(
            side_effect=PlannerConnectionError("Planner service unavailable")
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_use_case_defaults"]

        result = tool(use_case="chatbot_conversational")

        assert "error" in result
        assert "unavailable" in result["error"].lower()
        assert "hint" in result

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """Planner API error returns error dict with status code."""
        from rhoai_mcp.composites.planner.client import PlannerAPIError

        mock_client_class.return_value.get_slo_defaults = AsyncMock(
            side_effect=PlannerAPIError(status_code=404, detail="Use case not found")
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_use_case_defaults"]

        result = tool(use_case="chatbot_conversational")

        assert "error" in result
        assert result["status_code"] == 404


class TestGetExpectedRpsTool:
    """Tests for get_expected_rps tool."""

    def test_tool_registration(self) -> None:
        """get_expected_rps tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "get_expected_rps" in mock_mcp._registered_tools

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_successful_response(self, mock_client_class: MagicMock) -> None:
        """Returns expected and peak RPS for the given use case and user count."""
        mock_client_class.return_value.get_expected_rps = AsyncMock(return_value={
            "success": True,
            "use_case": "chatbot_conversational",
            "user_count": 1000,
            "workload_params": {"active_fraction": 0.1, "requests_per_active_user_per_min": 2.0, "peak_multiplier": 2.0},
            "expected_rps": 3.33,
            "expected_concurrent_users": 100,
            "peak_rps": 6.67,
        })
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_expected_rps"]

        result = tool(use_case="chatbot_conversational", user_count=1000)

        assert "error" not in result
        assert result["use_case"] == "chatbot_conversational"
        assert result["user_count"] == 1000
        assert result["expected_rps"] == 3.33
        assert result["peak_rps"] == 6.67
        assert result["expected_concurrent_users"] == 100

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_use_case(self, mock_client_class: MagicMock) -> None:
        """Invalid use_case returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_expected_rps"]

        result = tool(use_case="not_a_real_use_case", user_count=500)

        assert "error" in result
        assert "use_case" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_invalid_user_count(self, mock_client_class: MagicMock) -> None:
        """user_count <= 0 returns error without calling client."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_expected_rps"]

        result = tool(use_case="chatbot_conversational", user_count=0)

        assert "error" in result
        assert "user_count" in result["error"]
        mock_client_class.assert_not_called()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """Connection error returns error dict with hint."""
        from rhoai_mcp.composites.planner.client import PlannerConnectionError

        mock_client_class.return_value.get_expected_rps = AsyncMock(
            side_effect=PlannerConnectionError("Planner service unavailable")
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_expected_rps"]

        result = tool(use_case="chatbot_conversational", user_count=1000)

        assert "error" in result
        assert "unavailable" in result["error"].lower()
        assert "hint" in result

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """API error returns error dict with status code."""
        from rhoai_mcp.composites.planner.client import PlannerAPIError

        mock_client_class.return_value.get_expected_rps = AsyncMock(
            side_effect=PlannerAPIError(status_code=500, detail="Internal error")
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["get_expected_rps"]

        result = tool(use_case="chatbot_conversational", user_count=1000)

        assert "error" in result
        assert result["status_code"] == 500


class TestListUseCasesTool:
    """Tests for list_use_cases tool."""

    def test_tool_registration(self) -> None:
        """list_use_cases tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        assert "list_use_cases" in mock_mcp._registered_tools

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_successful_response(self, mock_client_class: MagicMock) -> None:
        """Returns use case list with id and description."""
        mock_client_class.return_value.list_use_cases = AsyncMock(return_value={
            "use_cases": {
                "chatbot_conversational": {
                    "use_case_id": "chatbot_conversational",
                    "description": "Conversational AI chatbot",
                },
                "code_completion": {
                    "use_case_id": "code_completion",
                    "description": "Code autocompletion",
                },
            },
            "count": 2,
        })
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["list_use_cases"]

        result = tool()

        assert "error" not in result
        assert result["count"] == 2
        ids = {uc["id"] for uc in result["use_cases"]}
        assert "chatbot_conversational" in ids
        assert "code_completion" in ids
        descriptions = {uc["description"] for uc in result["use_cases"]}
        assert "Conversational AI chatbot" in descriptions

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_connection_error(self, mock_client_class: MagicMock) -> None:
        """Connection error returns error dict with hint."""
        from rhoai_mcp.composites.planner.client import PlannerConnectionError

        mock_client_class.return_value.list_use_cases = AsyncMock(
            side_effect=PlannerConnectionError("Planner service unavailable")
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["list_use_cases"]

        result = tool()

        assert "error" in result
        assert "unavailable" in result["error"].lower()
        assert "hint" in result

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_api_error(self, mock_client_class: MagicMock) -> None:
        """API error returns error dict with status code."""
        from rhoai_mcp.composites.planner.client import PlannerAPIError

        mock_client_class.return_value.list_use_cases = AsyncMock(
            side_effect=PlannerAPIError(status_code=500, detail="Internal error")
        )
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server())
        tool = mock_mcp._registered_tools["list_use_cases"]

        result = tool()

        assert "error" in result
        assert result["status_code"] == 500


# ---------------------------------------------------------------------------
# Helpers for plan_deployment / execute_deployment tests
# ---------------------------------------------------------------------------

def _make_mock_server_with_k8s(mode: PlannerMode = PlannerMode.REMOTE) -> MagicMock:
    """Mock server with a k8s client stub."""
    server = _make_mock_server(mode=mode)
    server.k8s = MagicMock()
    server.config.is_operation_allowed.return_value = (True, "")
    return server


SAMPLE_VLLM_RUNTIME = {
    "name": "vllm-cuda-runtime",
    "display_name": "vLLM CUDA Runtime",
    "source": "namespace",
    "requires_instantiation": False,
    "supported_formats": ["pytorch"],
}

SAMPLE_VLLM_TEMPLATE = {
    "name": "vllm-cuda-runtime-template",
    "display_name": "vLLM CUDA Runtime",
    "source": "template",
    "requires_instantiation": True,
    "template_name": "vllm-cuda-runtime-template",
    "supported_formats": ["pytorch"],
}


class TestPlanDeploymentTool:
    """Tests for plan_deployment tool."""

    def test_tool_registration(self) -> None:
        """plan_deployment tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        assert "plan_deployment" in mock_mcp._registered_tools

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_successful_plan_runtime_available(
        self, mock_inference_cls: MagicMock, mock_planner_cls: MagicMock
    ) -> None:
        """Returns ready=True when vLLM runtime is already present in namespace."""
        mock_planner_cls.return_value.generate_config = AsyncMock(return_value=SAMPLE_CONFIG_RESULT)
        mock_inference_cls.return_value.list_serving_runtimes.return_value = [SAMPLE_VLLM_RUNTIME]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["ready"] is True
        assert result["runtime"] == "vllm-cuda-runtime"
        assert result["runtime_needs_creation"] is False
        assert result["template_to_instantiate"] is None
        assert result["model_id"] == "meta-llama/Llama-3.1-70B-Instruct"
        assert result["gpu_count"] == 2
        assert result["replicas"] == 1
        assert result["storage_uri"] == "oci://quay.io/rhoai/llama-3-1-70b:latest"
        assert "inferenceservice" in result["configs"]
        assert result["issues"] is None

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_plan_runtime_needs_creation(
        self, mock_inference_cls: MagicMock, mock_planner_cls: MagicMock
    ) -> None:
        """Returns runtime_needs_creation=True when runtime is a template."""
        mock_planner_cls.return_value.generate_config = AsyncMock(return_value=SAMPLE_CONFIG_RESULT)
        mock_inference_cls.return_value.list_serving_runtimes.return_value = [SAMPLE_VLLM_TEMPLATE]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["ready"] is True
        assert result["runtime_needs_creation"] is True
        assert result["template_to_instantiate"] == "vllm-cuda-runtime-template"

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_plan_no_vllm_runtime(
        self, mock_inference_cls: MagicMock, mock_planner_cls: MagicMock
    ) -> None:
        """Returns ready=False with issue when no vLLM runtime is found."""
        mock_planner_cls.return_value.generate_config = AsyncMock(return_value=SAMPLE_CONFIG_RESULT)
        mock_inference_cls.return_value.list_serving_runtimes.return_value = [
            {"name": "openvino-runtime", "source": "namespace", "supported_formats": ["onnx"]}
        ]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["ready"] is False
        assert result["runtime"] is None
        assert result["issues"] is not None
        assert any("vLLM" in issue for issue in result["issues"])

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_plan_warns_when_no_storage_uri(
        self, mock_inference_cls: MagicMock, mock_planner_cls: MagicMock
    ) -> None:
        """Warns when model_uri is not returned by planner."""
        config_without_uri = DeploymentConfigResult(
            deployment_id="test-id",
            namespace="my-project",
            model_name="Llama 3.1 70B",
            configs={"inferenceservice": "yaml"},
        )
        mock_planner_cls.return_value.generate_config = AsyncMock(return_value=config_without_uri)
        mock_inference_cls.return_value.list_serving_runtimes.return_value = [SAMPLE_VLLM_RUNTIME]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["storage_uri"] is None
        assert result["warnings"] is not None
        assert any("storage" in w.lower() for w in result["warnings"])

    def test_plan_invalid_category(self) -> None:
        """Invalid category returns error."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="invalid",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=100,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=5.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert "error" in result
        assert "invalid" in result["error"].lower()

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_plan_planner_connection_error(self, mock_planner_cls: MagicMock) -> None:
        """Planner connection error returns error dict."""
        from rhoai_mcp.composites.planner.client import PlannerConnectionError

        mock_planner_cls.return_value.generate_config = AsyncMock(
            side_effect=PlannerConnectionError("Planner unavailable")
        )

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
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


    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_plan_missing_model_id_blocks_ready(
        self, mock_inference_cls: MagicMock, mock_planner_cls: MagicMock
    ) -> None:
        """ready=False and an issue is raised when planner returns no model_id."""
        no_model_id_result = DeploymentConfigResult(
            deployment_id="chatbot-unknown-20260322143022",
            namespace="my-project",
            model_name="Llama 3.1 70B",
            model_id=None,
            model_uri="oci://quay.io/rhoai/llama-3-1-70b:latest",
            gpu_config={"gpu_type": "NVIDIA-H100", "gpu_count": 2, "tensor_parallel": 2, "replicas": 1},
            configs={"inferenceservice": "apiVersion: serving.kserve.io/v1beta1"},
        )
        mock_planner_cls.return_value.generate_config = AsyncMock(return_value=no_model_id_result)
        mock_inference_cls.return_value.list_serving_runtimes.return_value = [SAMPLE_VLLM_RUNTIME]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["ready"] is False
        assert result["issues"] is not None
        assert any("model_id" in issue for issue in result["issues"])

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_suggested_deploy_params_populated_when_ready(
        self, mock_inference_cls: MagicMock, mock_planner_cls: MagicMock
    ) -> None:
        """suggested_deploy_params is populated when model_id, runtime, and storage_uri are set."""
        mock_planner_cls.return_value.generate_config = AsyncMock(return_value=SAMPLE_CONFIG_RESULT)
        mock_inference_cls.return_value.list_serving_runtimes.return_value = [SAMPLE_VLLM_RUNTIME]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        params = result["suggested_deploy_params"]
        assert params is not None
        assert params["model_id"] == "meta-llama/Llama-3.1-70B-Instruct"
        assert params["namespace"] == "my-project"
        assert params["runtime"] == "vllm-cuda-runtime"
        assert params["storage_uri"] == "oci://quay.io/rhoai/llama-3-1-70b:latest"
        assert params["gpu_count"] == 2
        assert params["gpu_type"] == "H100"
        assert params["tensor_parallel"] == 2
        assert params["replicas"] == 1

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_suggested_deploy_params_none_when_storage_missing(
        self, mock_inference_cls: MagicMock, mock_planner_cls: MagicMock
    ) -> None:
        """suggested_deploy_params is None when storage_uri cannot be resolved."""
        no_storage_result = DeploymentConfigResult(
            deployment_id="chatbot-llama-20260322143022",
            namespace="my-project",
            model_name="Llama 3.1 70B",
            model_id="meta-llama/Llama-3.1-70B-Instruct",
            model_uri=None,
            gpu_config={"gpu_type": "NVIDIA-H100", "gpu_count": 2, "tensor_parallel": 2, "replicas": 1},
            configs={"inferenceservice": "apiVersion: serving.kserve.io/v1beta1"},
        )
        mock_planner_cls.return_value.generate_config = AsyncMock(return_value=no_storage_result)
        mock_inference_cls.return_value.list_serving_runtimes.return_value = [SAMPLE_VLLM_RUNTIME]

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["plan_deployment"]

        result = tool(
            category="balanced",
            namespace="my-project",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert result["suggested_deploy_params"] is None


class TestExecuteDeploymentTool:
    """Tests for execute_deployment tool."""

    def test_tool_registration(self) -> None:
        """execute_deployment tool is registered."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        assert "execute_deployment" in mock_mcp._registered_tools

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_successful_deployment(self, mock_inference_cls: MagicMock) -> None:
        """Successful deployment returns deployed=True with name and namespace."""
        mock_isvc = MagicMock()
        mock_isvc.status.value = "Pending"
        mock_isvc.metadata.name = "llama-3-1-70b-instruct"
        mock_isvc.metadata.namespace = "my-project"
        mock_isvc.metadata.to_source_dict.return_value = {
            "kind": "InferenceService",
            "name": "llama-3-1-70b-instruct",
        }
        mock_inference_cls.return_value.deploy_model.return_value = mock_isvc
        # Poll returns Pending then Ready
        mock_updated = MagicMock()
        mock_updated.status.value = "Ready"
        mock_inference_cls.return_value.get_inference_service.return_value = mock_updated

        mock_mcp = _make_mock_mcp()
        server = _make_mock_server_with_k8s()

        with patch("rhoai_mcp.composites.planner.tools.asyncio.sleep", new_callable=AsyncMock):
            register_tools(mock_mcp, server)
            tool = mock_mcp._registered_tools["execute_deployment"]

            result = tool(
                model_id="meta-llama/Llama-3.1-70B-Instruct",
                namespace="my-project",
                runtime="vllm-cuda-runtime",
                storage_uri="oci://quay.io/rhoai/llama:latest",
                gpu_count=2,
                gpu_type="H100",
                replicas=1,
            )

        assert result["deployed"] is True
        assert result["name"] == "llama-3-1-70b-instruct"
        assert result["namespace"] == "my-project"
        assert result["status"] == "Ready"
        assert "Ready" in result["message"]

    def test_execute_operation_not_allowed(self) -> None:
        """Returns error when create operations are not permitted."""
        mock_mcp = _make_mock_mcp()
        server = _make_mock_server_with_k8s()
        server.config.is_operation_allowed.return_value = (False, "Read-only mode is enabled")

        register_tools(mock_mcp, server)
        tool = mock_mcp._registered_tools["execute_deployment"]

        result = tool(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            namespace="my-project",
            runtime="vllm-cuda-runtime",
            storage_uri="oci://quay.io/rhoai/llama:latest",
            gpu_count=1,
            gpu_type="H100",
        )

        assert "error" in result

    def test_execute_invalid_gpu_count(self) -> None:
        """Returns error for gpu_count <= 0."""
        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["execute_deployment"]

        result = tool(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            namespace="my-project",
            runtime="vllm-cuda-runtime",
            storage_uri="oci://quay.io/rhoai/llama:latest",
            gpu_count=0,
            gpu_type="H100",
        )

        assert "error" in result

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_execute_tensor_parallel_forwarded(self, mock_inference_cls: MagicMock) -> None:
        """tensor_parallel is passed to InferenceServiceCreate, not silently dropped."""
        mock_isvc = MagicMock()
        mock_isvc.status.value = "Ready"
        mock_isvc.metadata.name = "llama-3-1-70b-instruct"
        mock_isvc.metadata.namespace = "my-project"
        mock_isvc.metadata.to_source_dict.return_value = {"kind": "InferenceService", "name": "llama-3-1-70b-instruct"}
        mock_inference_cls.return_value.deploy_model.return_value = mock_isvc

        mock_mcp = _make_mock_mcp()
        with patch("rhoai_mcp.composites.planner.tools.asyncio.sleep", new_callable=AsyncMock):
            register_tools(mock_mcp, _make_mock_server_with_k8s())
            tool = mock_mcp._registered_tools["execute_deployment"]
            tool(
                model_id="meta-llama/Llama-3.1-70B-Instruct",
                namespace="my-project",
                runtime="vllm-cuda-runtime",
                storage_uri="oci://quay.io/rhoai/llama:latest",
                gpu_count=4,
                gpu_type="H100",
                tensor_parallel=4,
                replicas=1,
            )

        call_args = mock_inference_cls.return_value.deploy_model.call_args
        request = call_args[0][0]
        assert request.tensor_parallel == 4

    @patch("rhoai_mcp.domains.inference.client.InferenceClient")
    def test_execute_deploy_failure(self, mock_inference_cls: MagicMock) -> None:
        """Returns deployed=False when InferenceClient raises."""
        mock_inference_cls.return_value.deploy_model.side_effect = RuntimeError("K8s API error")

        mock_mcp = _make_mock_mcp()
        register_tools(mock_mcp, _make_mock_server_with_k8s())
        tool = mock_mcp._registered_tools["execute_deployment"]

        result = tool(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            namespace="my-project",
            runtime="vllm-cuda-runtime",
            storage_uri="oci://quay.io/rhoai/llama:latest",
            gpu_count=1,
            gpu_type="H100",
        )

        assert result["deployed"] is False
        assert "error" in result


class TestGpuMemoryHelper:
    """Tests for _gpu_memory_for_type helper."""

    def test_known_gpu_types(self) -> None:
        """Known GPU types return correct memory strings."""
        from rhoai_mcp.composites.planner.tools import _gpu_memory_for_type

        assert _gpu_memory_for_type("H100", 1) == "80Gi"
        assert _gpu_memory_for_type("H100", 2) == "160Gi"
        assert _gpu_memory_for_type("A100-80", 1) == "80Gi"
        assert _gpu_memory_for_type("A100-40", 2) == "80Gi"
        assert _gpu_memory_for_type("L4", 4) == "96Gi"
        assert _gpu_memory_for_type("H200", 1) == "141Gi"
        assert _gpu_memory_for_type("B200", 1) == "192Gi"

    def test_unknown_gpu_type_falls_back(self) -> None:
        """Unknown GPU types return 80Gi per GPU as fallback."""
        from rhoai_mcp.composites.planner.tools import _gpu_memory_for_type

        assert _gpu_memory_for_type("NVIDIA-T4", 1) == "80Gi"
        assert _gpu_memory_for_type("NVIDIA-H100", 2) == "160Gi"

    def test_planner_style_names(self) -> None:
        """Planner-style gpu_type strings like 'NVIDIA-H100' are matched correctly."""
        from rhoai_mcp.composites.planner.tools import _gpu_memory_for_type

        assert _gpu_memory_for_type("NVIDIA-H100", 1) == "80Gi"
        assert _gpu_memory_for_type("NVIDIA-A100-80", 1) == "80Gi"


class TestMakeDeploymentName:
    """Tests for _make_deployment_name helper."""

    def test_basic_model_id(self) -> None:
        from rhoai_mcp.composites.planner.tools import _make_deployment_name

        assert _make_deployment_name("meta-llama/Llama-3.1-70B-Instruct") == "llama-3-1-70b-instruct"

    def test_no_trailing_hyphen_after_truncation(self) -> None:
        """Truncation at 50 chars must not leave a trailing hyphen."""
        from rhoai_mcp.composites.planner.tools import _make_deployment_name

        # 'a' * 49 + '-b' → after sanitise = 'a'*49 + '-b', truncate at 50 = 'a'*49 + '-'
        model_id = "a" * 49 + "-b"
        result = _make_deployment_name(model_id)
        assert not result.endswith("-")
        assert len(result) <= 50

    def test_special_chars_replaced(self) -> None:
        from rhoai_mcp.composites.planner.tools import _make_deployment_name

        assert _make_deployment_name("org/My_Model.v2") == "my-model-v2"

    def test_numeric_prefix_gets_model_prefix(self) -> None:
        from rhoai_mcp.composites.planner.tools import _make_deployment_name

        result = _make_deployment_name("7b-instruct")
        assert result.startswith("model-")


class TestClientFactory:
    """Tests for the _get_client factory function."""

    @patch("rhoai_mcp.composites.planner.tools.LocalPlannerClient")
    def test_local_mode_creates_local_client(self, mock_local_cls: MagicMock) -> None:
        """Factory creates LocalPlannerClient when mode is LOCAL."""
        mock_local_cls.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server(mode=PlannerMode.LOCAL)

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        recommend_model(
            text="chatbot",
            use_case="chatbot_conversational",
            user_count=1000,
            preferred_gpu_types=["H100"],
        )

        mock_local_cls.assert_called_once_with(
            model_catalog_url=None,
        )

    @patch("rhoai_mcp.composites.planner.tools.PlannerClient")
    def test_remote_mode_creates_remote_client(self, mock_client_class: MagicMock) -> None:
        """Factory creates PlannerClient when mode is REMOTE."""
        mock_client_class.return_value.recommend = AsyncMock(return_value=SAMPLE_RESULT)
        mock_mcp = _make_mock_mcp()
        mock_server = _make_mock_server(mode=PlannerMode.REMOTE)

        register_tools(mock_mcp, mock_server)
        recommend_model = mock_mcp._registered_tools["recommend_model"]

        recommend_model(text="chatbot")

        mock_client_class.assert_called_once_with(
            "http://localhost:8000",
            timeout=120.0,
        )
