"""Tests for Planner HTTP client."""

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.composites.planner.client import (
    PlannerAPIError,
    PlannerClient,
    PlannerConnectionError,
)
from rhoai_mcp.composites.planner.models import DeploymentConfigResult

SAMPLE_INTENT = {
    "use_case": "chatbot_conversational",
    "user_count": 1000,
    "domain_specialization": ["general"],
    "preferred_gpu_types": [],
    "preferred_models": [],
    "quality_priority": "medium",
    "cost_priority": "medium",
    "latency_priority": "medium",
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

SAMPLE_CONFIGURATION = {
    "model_id": "meta-llama/Llama-3.1-70B-Instruct",
    "model_name": "Llama 3.1 70B",
    "model_uri": None,
    "gpu_config": {
        "gpu_type": "NVIDIA-H100",
        "gpu_count": 2,
        "tensor_parallel": 2,
        "replicas": 1,
    },
    "use_case": "chatbot_conversational",
    "expected_qps": 10.0,
    "prompt_tokens": 512,
    "output_tokens": 256,
    "e2e_target_ms": 2000,
}

SAMPLE_RECOMMENDATION = {
    "model_id": "meta-llama/Llama-3.1-70B-Instruct",
    "model_name": "Llama 3.1 70B",
    "model_uri": None,
    "gpu_config": {
        "gpu_type": "NVIDIA-H100",
        "gpu_count": 2,
        "tensor_parallel": 2,
        "replicas": 1,
    },
    "predicted_ttft_p95_ms": 140,
    "predicted_itl_p95_ms": 50,
    "predicted_e2e_p95_ms": 1200,
    "predicted_throughput_qps": 100.0,
    "cost_per_hour_usd": 3.98,
    "cost_per_month_usd": 2872.32,
    "meets_slo": True,
    "reasoning": "Selected Llama 3.1 70B for chatbot use case",
    "scores": {
        "quality_score": 78,
        "price_score": 65,
        "latency_score": 95,
        "balanced_score": 75.3,
        "slo_status": "compliant",
    },
    "configuration": SAMPLE_CONFIGURATION,
}

_SAMPLE_SPECIFICATION: dict[str, Any] = {
    "intent": SAMPLE_INTENT,
    "slo_targets": {
        "ttft_target_ms": 150,
        "itl_target_ms": 65,
        "e2e_target_ms": 2000,
        "percentile": "p95",
    },
    "workload_profile": {
        "prompt_tokens": 512,
        "output_tokens": 256,
        "expected_qps": 10.0,
    },
    "priorities": {
        "quality": {"priority": "medium", "weight": 4},
        "cost": {"priority": "medium", "weight": 4},
        "latency": {"priority": "medium", "weight": 4},
    },
}


def sample_specification() -> dict[str, Any]:
    """Return a fresh copy to prevent cross-test mutation."""
    return copy.deepcopy(_SAMPLE_SPECIFICATION)


SAMPLE_RANKED_RESPONSE = {
    "balanced": [SAMPLE_RECOMMENDATION],
    "best_quality": [SAMPLE_RECOMMENDATION],
    "lowest_cost": [SAMPLE_RECOMMENDATION],
    "lowest_latency": [SAMPLE_RECOMMENDATION],
    "total_configs_evaluated": 2847,
    "configs_after_filters": 542,
    "specification": _SAMPLE_SPECIFICATION,
}

SAMPLE_DEPLOYMENT_BUNDLE = {
    "deployment_id": "chatbot-llama-3-1-70b-20260322143022",
    "namespace": "default",
    "stack": "vllm",
    "configuration": SAMPLE_CONFIGURATION,
    "files": {
        "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService",
        "autoscaling": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler",
        "servicemonitor": "apiVersion: monitoring.coreos.com/v1\nkind: ServiceMonitor",
    },
}


class TestPlannerClientExtractIntent:
    """Tests for intent extraction."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_extract_intent_success(self, mock_httpx: MagicMock) -> None:
        """Successful intent extraction returns DeploymentIntent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_INTENT
        mock_response.raise_for_status = MagicMock()
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_httpx.Client.return_value.__enter__.return_value.post.return_value = mock_response
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        intent = client.extract_intent("I need a chatbot for 1000 users")

        assert intent.use_case == "chatbot_conversational"
        assert intent.user_count == 1000

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_extract_intent_connection_error(self, mock_httpx: MagicMock) -> None:
        """Connection failure raises PlannerConnectionError."""
        import httpx as real_httpx

        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.TimeoutException = real_httpx.TimeoutException
        mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
        mock_client = MagicMock()
        mock_client.post.side_effect = real_httpx.ConnectError("Connection refused")
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerConnectionError):
            client.extract_intent("test")

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_extract_intent_malformed_response(self, mock_httpx: MagicMock) -> None:
        """Malformed intent response raises PlannerAPIError."""
        import httpx as real_httpx

        mock_httpx.TimeoutException = real_httpx.TimeoutException
        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.RequestError = real_httpx.RequestError
        mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected_field": "value"}
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError, match="invalid intent response"):
            client.extract_intent("test")


class TestPlannerClientGetDefaults:
    """Tests for fetching SLO/workload defaults."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_get_slo_defaults(self, mock_httpx: MagicMock) -> None:
        """SLO defaults are fetched and parsed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_SLO_DEFAULTS
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        defaults = client.get_slo_defaults("chatbot_conversational")

        assert defaults["slo_defaults"]["ttft_ms"]["default"] == 150

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_get_workload_profile(self, mock_httpx: MagicMock) -> None:
        """Workload profile is fetched and parsed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_WORKLOAD_PROFILE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        profile = client.get_workload_profile("chatbot_conversational")

        assert profile["workload_profile"]["prompt_tokens"] == 512

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_get_expected_rps(self, mock_httpx: MagicMock) -> None:
        """Expected RPS is fetched and parsed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_EXPECTED_RPS
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        rps = client.get_expected_rps("chatbot_conversational", 1000)

        assert rps["expected_rps"] == 10.0


class TestPlannerClientGenerateSpecification:
    """Tests for generate_specification method."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_specification(self, mock_httpx: MagicMock) -> None:
        """Specification is generated from intent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _SAMPLE_SPECIFICATION
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        from rhoai_mcp.composites.planner.models import DeploymentIntent

        client = PlannerClient("http://localhost:8000")
        intent = DeploymentIntent(use_case="chatbot_conversational", user_count=1000)
        spec = client.generate_specification(intent)

        assert spec["slo_targets"]["ttft_target_ms"] == 150
        call_args = mock_client.post.call_args
        assert "/api/v1/generate-specification" in call_args.args[0]


class TestPlannerClientGenerateRecommendations:
    """Tests for generate_recommendations method."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_recommendations(self, mock_httpx: MagicMock) -> None:
        """Ranked recommendations are generated from specification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_RANKED_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.generate_recommendations(_SAMPLE_SPECIFICATION)

        assert len(result["balanced"]) == 1
        assert result["total_configs_evaluated"] == 2847
        call_args = mock_client.post.call_args
        assert "/api/v1/generate-recommendations" in call_args.args[0]

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_recommendations_with_constraints(self, mock_httpx: MagicMock) -> None:
        """Constraint parameters are included in the POST payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_RANKED_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        client.generate_recommendations(
            _SAMPLE_SPECIFICATION,
            min_quality=70,
            max_cost=5000.0,
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["min_quality"] == 70
        assert payload["max_cost"] == 5000.0

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_recommendations_without_constraints(self, mock_httpx: MagicMock) -> None:
        """When no constraints are provided, they are omitted from payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_RANKED_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        client.generate_recommendations(_SAMPLE_SPECIFICATION)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "min_quality" not in payload
        assert "max_cost" not in payload


class TestPlannerClientRecommend:
    """Tests for the full recommendation flow."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_full_flow(self, mock_httpx: MagicMock) -> None:
        """Full recommend() chains extract -> generate-specification -> generate-recommendations."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        # 3 POST calls: extract, generate-specification, generate-recommendations
        mock_client.post.side_effect = [extract_resp, spec_resp, ranked_resp]

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.recommend("I need a chatbot for 1000 users")

        assert result.top_balanced is not None
        assert result.top_balanced.model_id == "meta-llama/Llama-3.1-70B-Instruct"
        assert result.top_cost is not None
        assert result.top_performance is not None
        assert result.top_quality is not None
        assert result.specification["use_case"] == "chatbot_conversational"
        assert result.total_configs_evaluated == 2847
        # 3 POSTs, 0 GETs
        assert mock_client.post.call_count == 3
        assert mock_client.get.call_count == 0

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_with_overrides(self, mock_httpx: MagicMock) -> None:
        """When all overrides are provided, extraction is skipped."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        # Only 2 POST calls: generate-specification, generate-recommendations
        mock_client.post.side_effect = [spec_resp, ranked_resp]

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.recommend(
            "I need a chatbot",
            use_case_override="code_completion",
            user_count_override=5000,
            gpu_types_override=["H100"],
        )

        # Extraction skipped — only 2 POST calls
        assert mock_client.post.call_count == 2
        assert result.specification["use_case"] == "code_completion"

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_api_error(self, mock_httpx: MagicMock) -> None:
        """API error during recommendation raises PlannerAPIError."""
        import httpx as real_httpx

        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.TimeoutException = real_httpx.TimeoutException
        mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
        mock_httpx.RequestError = real_httpx.RequestError
        mock_client = MagicMock()

        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"
        error_response.raise_for_status.side_effect = real_httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=error_response,
        )
        mock_client.post.return_value = error_response

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError):
            client.extract_intent("test")

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_with_slo_overrides(self, mock_httpx: MagicMock) -> None:
        """SLO overrides replace generated specification values."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp, ranked_resp]

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.recommend(
            "I need a chatbot",
            ttft_override_ms=100,
            itl_override_ms=30,
            e2e_override_ms=1500,
        )

        assert result.specification["slo_targets"]["ttft_target_ms"] == 100
        assert result.specification["slo_targets"]["itl_target_ms"] == 30
        assert result.specification["slo_targets"]["e2e_target_ms"] == 1500

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_with_partial_slo_overrides(self, mock_httpx: MagicMock) -> None:
        """Partial SLO overrides only replace the specified values."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp, ranked_resp]

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.recommend(
            "I need a chatbot",
            ttft_override_ms=100,
        )

        # TTFT overridden, ITL and E2E use generated specification defaults
        assert result.specification["slo_targets"]["ttft_target_ms"] == 100
        assert result.specification["slo_targets"]["itl_target_ms"] == 65
        assert result.specification["slo_targets"]["e2e_target_ms"] == 2000

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_forwards_constraints(self, mock_httpx: MagicMock) -> None:
        """min_quality and max_cost are forwarded to generate_recommendations."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp, ranked_resp]

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        client.recommend(
            "I need a chatbot",
            min_quality=70,
            max_cost=5000.0,
        )

        # Verify constraints were forwarded to the generate-recommendations POST
        ranked_call = mock_client.post.call_args_list[2]
        payload = ranked_call.kwargs.get("json") or ranked_call[1].get("json")
        assert payload["min_quality"] == 70
        assert payload["max_cost"] == 5000.0


    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_forwards_percentile_override(self, mock_httpx: MagicMock) -> None:
        """percentile_override is applied to the specification's slo_targets."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp, ranked_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        client.recommend("I need a chatbot", percentile_override="p99")

        ranked_call = mock_client.post.call_args_list[2]
        payload = ranked_call.kwargs.get("json") or ranked_call[1].get("json")
        assert payload["specification"]["slo_targets"]["percentile"] == "p99"

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_forwards_priority_weights(self, mock_httpx: MagicMock) -> None:
        """priority_weights override the specification's priorities."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp, ranked_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        weights = {"quality": 8, "price": 2, "latency": 1}
        client.recommend("I need a chatbot", priority_weights=weights)

        ranked_call = mock_client.post.call_args_list[2]
        payload = ranked_call.kwargs.get("json") or ranked_call[1].get("json")
        priorities = payload["specification"]["priorities"]
        assert priorities["quality"]["weight"] == 8
        assert priorities["cost"]["weight"] == 2
        assert priorities["latency"]["weight"] == 1


    @pytest.mark.parametrize("bad_priorities", [None, []])
    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_invalid_priorities(
        self, mock_httpx: MagicMock, bad_priorities: Any
    ) -> None:
        """Non-dict priorities raises PlannerAPIError(502)."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec = sample_specification()
        spec["priorities"] = bad_priorities
        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = spec
        spec_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError) as exc_info:
            client.recommend(
                "I need a chatbot",
                priority_weights={"quality": 8, "price": 2, "latency": 1},
            )
        assert exc_info.value.status_code == 502

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_invalid_priority_entry(self, mock_httpx: MagicMock) -> None:
        """Null priority entry raises PlannerAPIError(502)."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec = sample_specification()
        spec["priorities"]["quality"] = None
        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = spec
        spec_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError) as exc_info:
            client.recommend(
                "I need a chatbot",
                priority_weights={"quality": 8, "price": 2, "latency": 1},
            )
        assert exc_info.value.status_code == 502

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_missing_priority_category(self, mock_httpx: MagicMock) -> None:
        """Missing priority category in spec raises PlannerAPIError(502)."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec = sample_specification()
        del spec["priorities"]["quality"]
        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = spec
        spec_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError) as exc_info:
            client.recommend(
                "I need a chatbot",
                priority_weights={"quality": 8, "price": 2, "latency": 1},
            )
        assert exc_info.value.status_code == 502
        assert "priorities.quality" in exc_info.value.detail


class TestPlannerClientRecommendExtractionBypass:
    """Tests for skipping extraction when overrides are sufficient."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_skips_extraction_when_all_overrides_provided(
        self, mock_httpx: MagicMock
    ) -> None:
        """When all overrides are provided, extraction is skipped."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        # Only 2 POST calls: generate-specification, generate-recommendations
        mock_client.post.side_effect = [spec_resp, ranked_resp]

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.recommend(
            "I need a chatbot for 1000 users",
            use_case_override="chatbot_conversational",
            user_count_override=1000,
            gpu_types_override=["A100"],
        )

        # Only two POST calls (generate-specification + generate-recommendations)
        assert mock_client.post.call_count == 2
        assert result.specification["use_case"] == "chatbot_conversational"

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_recommend_still_extracts_when_only_use_case_override(
        self, mock_httpx: MagicMock
    ) -> None:
        """When only use_case override is provided, extraction still runs for user_count."""
        mock_client = MagicMock()

        extract_resp = MagicMock()
        extract_resp.status_code = 200
        extract_resp.json.return_value = SAMPLE_INTENT
        extract_resp.raise_for_status = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [extract_resp, spec_resp, ranked_resp]

        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.recommend(
            "I need a chatbot",
            use_case_override="code_completion",
        )

        # Three POST calls: extract + generate-specification + generate-recommendations
        assert mock_client.post.call_count == 3
        assert result.specification["use_case"] == "code_completion"


class TestPlannerClientRequestErrors:
    """Tests for _request error handling edge cases."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_invalid_json_response(self, mock_httpx: MagicMock) -> None:
        """Non-JSON response raises PlannerAPIError."""
        import httpx as real_httpx

        mock_httpx.TimeoutException = real_httpx.TimeoutException
        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.RequestError = real_httpx.RequestError
        mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError, match="invalid JSON"):
            client.get_slo_defaults("chatbot_conversational")

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generic_request_error(self, mock_httpx: MagicMock) -> None:
        """Other httpx.RequestError subtypes raise PlannerConnectionError."""
        import httpx as real_httpx

        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.TimeoutException = real_httpx.TimeoutException
        mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
        mock_httpx.RequestError = real_httpx.RequestError
        mock_client = MagicMock()
        mock_client.get.side_effect = real_httpx.RequestError("protocol error")
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerConnectionError, match="request failed"):
            client.get_slo_defaults("chatbot_conversational")


class TestPlannerClientHealthCheck:
    """Tests for health check."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_health_check_healthy(self, mock_httpx: MagicMock) -> None:
        """Health check returns True when service is reachable."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        healthy, msg = client.health_check()

        assert healthy is True
        assert "available" in msg.lower()
        mock_client.get.assert_called_once_with("http://localhost:8000/health", params=None)

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_health_check_unhealthy(self, mock_httpx: MagicMock) -> None:
        """Health check returns False when service is unreachable."""
        import httpx as real_httpx

        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.TimeoutException = real_httpx.TimeoutException
        mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
        mock_client = MagicMock()
        mock_client.get.side_effect = real_httpx.ConnectError("Connection refused")
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        healthy, msg = client.health_check()

        assert healthy is False
        assert "unavailable" in msg.lower()


class TestPlannerClientGenerateDeployment:
    """Tests for generate_deployment() method."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_deployment(self, mock_httpx: MagicMock) -> None:
        """generate_deployment() sends configuration + namespace and returns bundle."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_DEPLOYMENT_BUNDLE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.generate_deployment(SAMPLE_CONFIGURATION, namespace="ml-prod")

        call_args = mock_client.post.call_args
        assert "/api/v1/generate-deployment" in call_args.args[0]
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["configuration"] == SAMPLE_CONFIGURATION
        assert payload["namespace"] == "ml-prod"

        assert result["deployment_id"] == "chatbot-llama-3-1-70b-20260322143022"
        assert "inferenceservice" in result["files"]


class TestPlannerClientGenerateConfig:
    """Tests for generate_config() method."""

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_balanced(self, mock_httpx: MagicMock) -> None:
        """generate_config with category='balanced' picks from balanced list."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.json.return_value = SAMPLE_DEPLOYMENT_BUNDLE
        deploy_resp.raise_for_status = MagicMock()

        # 3 POSTs: generate-specification, generate-recommendations, generate-deployment
        mock_client.post.side_effect = [spec_resp, ranked_resp, deploy_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.generate_config(
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

        assert isinstance(result, DeploymentConfigResult)
        assert result.deployment_id == "chatbot-llama-3-1-70b-20260322143022"
        assert result.model_name == "Llama 3.1 70B"
        assert "inferenceservice" in result.configs

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_applies_workload_overrides(self, mock_httpx: MagicMock) -> None:
        """Caller's prompt_tokens/output_tokens/expected_qps override the specification."""
        mock_client = MagicMock()

        spec = sample_specification()
        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = spec
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.json.return_value = SAMPLE_DEPLOYMENT_BUNDLE
        deploy_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp, ranked_resp, deploy_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        client.generate_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=1024,
            output_tokens=512,
            expected_qps=25.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        # The second POST is generate-recommendations; its payload should
        # contain the overridden workload profile values
        recommend_call = mock_client.post.call_args_list[1]
        sent_spec = recommend_call.kwargs.get("json", recommend_call[1].get("json", {}))
        workload = sent_spec["specification"]["workload_profile"]
        assert workload["prompt_tokens"] == 1024
        assert workload["output_tokens"] == 512
        assert workload["expected_qps"] == 25.0

    @pytest.mark.parametrize("bad_workload", [None, []])
    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_invalid_workload_profile(
        self, mock_httpx: MagicMock, bad_workload: Any
    ) -> None:
        """Non-dict workload_profile raises PlannerAPIError(502)."""
        mock_client = MagicMock()

        spec = sample_specification()
        spec["workload_profile"] = bad_workload
        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = spec
        spec_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError) as exc_info:
            client.generate_config(
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
        assert exc_info.value.status_code == 502

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_cost(self, mock_httpx: MagicMock) -> None:
        """category='cost' maps to 'lowest_cost' ranking list."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.json.return_value = SAMPLE_DEPLOYMENT_BUNDLE
        deploy_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp, ranked_resp, deploy_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.generate_config(
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

        assert isinstance(result, DeploymentConfigResult)
        assert result.deployment_id == "chatbot-llama-3-1-70b-20260322143022"

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_performance(self, mock_httpx: MagicMock) -> None:
        """category='performance' maps to 'lowest_latency' ranking list."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.json.return_value = SAMPLE_DEPLOYMENT_BUNDLE
        deploy_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp, ranked_resp, deploy_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.generate_config(
            category="performance",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert isinstance(result, DeploymentConfigResult)

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_quality(self, mock_httpx: MagicMock) -> None:
        """category='quality' maps to 'best_quality' ranking list."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.json.return_value = SAMPLE_DEPLOYMENT_BUNDLE
        deploy_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp, ranked_resp, deploy_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.generate_config(
            category="quality",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert isinstance(result, DeploymentConfigResult)

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_empty_category(self, mock_httpx: MagicMock) -> None:
        """Empty category list raises PlannerAPIError."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        empty_ranked = {
            **SAMPLE_RANKED_RESPONSE,
            "balanced": [],
        }
        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = empty_ranked
        ranked_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp, ranked_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError, match="No recommendation found"):
            client.generate_config(
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

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_deploy_error(self, mock_httpx: MagicMock) -> None:
        """Deploy failure after ranking succeeds raises PlannerAPIError."""
        import httpx as real_httpx

        mock_httpx.ConnectError = real_httpx.ConnectError
        mock_httpx.TimeoutException = real_httpx.TimeoutException
        mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError
        mock_httpx.RequestError = real_httpx.RequestError
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"
        error_response.raise_for_status.side_effect = real_httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=error_response,
        )

        mock_client.post.side_effect = [spec_resp, ranked_resp, error_response]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError):
            client.generate_config(
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

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_empty_files(self, mock_httpx: MagicMock) -> None:
        """Deploy returns empty files dict raises PlannerAPIError."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = SAMPLE_RANKED_RESPONSE
        ranked_resp.raise_for_status = MagicMock()

        empty_bundle = {**SAMPLE_DEPLOYMENT_BUNDLE, "files": {}}
        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.json.return_value = empty_bundle
        deploy_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp, ranked_resp, deploy_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        with pytest.raises(PlannerAPIError, match="no config files"):
            client.generate_config(
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

    @patch("rhoai_mcp.composites.planner.client.httpx")
    def test_generate_config_model_id_fallback(self, mock_httpx: MagicMock) -> None:
        """When model_name is None, model_id is used instead."""
        mock_client = MagicMock()

        spec_resp = MagicMock()
        spec_resp.status_code = 200
        spec_resp.json.return_value = sample_specification()
        spec_resp.raise_for_status = MagicMock()

        rec_no_name = {**SAMPLE_RECOMMENDATION, "model_name": None}
        ranked_no_name = {**SAMPLE_RANKED_RESPONSE, "balanced": [rec_no_name]}
        ranked_resp = MagicMock()
        ranked_resp.status_code = 200
        ranked_resp.json.return_value = ranked_no_name
        ranked_resp.raise_for_status = MagicMock()

        deploy_resp = MagicMock()
        deploy_resp.status_code = 200
        deploy_resp.json.return_value = SAMPLE_DEPLOYMENT_BUNDLE
        deploy_resp.raise_for_status = MagicMock()

        mock_client.post.side_effect = [spec_resp, ranked_resp, deploy_resp]
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.generate_config(
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

        assert result.model_name == "meta-llama/Llama-3.1-70B-Instruct"
