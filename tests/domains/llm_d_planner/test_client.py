"""Tests for llm-d-planner HTTP client."""

from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.domains.llm_d_planner.client import (
    PlannerAPIError,
    PlannerClient,
    PlannerConnectionError,
)

SAMPLE_INTENT = {
    "use_case": "chatbot_conversational",
    "experience_class": "conversational",
    "user_count": 1000,
    "domain_specialization": ["general"],
    "preferred_gpu_types": [],
    "accuracy_priority": "medium",
    "cost_priority": "medium",
    "latency_priority": "medium",
    "complexity_priority": "medium",
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

SAMPLE_RECOMMENDATION = {
    "model_id": "meta-llama/Llama-3.1-70B-Instruct",
    "model_name": "Llama 3.1 70B",
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
        "accuracy_score": 78,
        "price_score": 65,
        "latency_score": 95,
        "complexity_score": 90,
        "balanced_score": 75.3,
        "slo_status": "compliant",
    },
    "intent": SAMPLE_INTENT,
    "traffic_profile": {
        "prompt_tokens": 512,
        "output_tokens": 256,
        "expected_qps": 10.0,
    },
    "slo_targets": {
        "ttft_p95_target_ms": 150,
        "itl_p95_target_ms": 65,
        "e2e_p95_target_ms": 2000,
        "percentile": "p95",
    },
}

SAMPLE_RANKED_RESPONSE = {
    "balanced": [SAMPLE_RECOMMENDATION],
    "best_accuracy": [SAMPLE_RECOMMENDATION],
    "lowest_cost": [SAMPLE_RECOMMENDATION],
    "lowest_latency": [SAMPLE_RECOMMENDATION],
    "simplest": [SAMPLE_RECOMMENDATION],
    "total_configs_evaluated": 2847,
    "configs_after_filters": 542,
    "specification": {
        "intent": SAMPLE_INTENT,
        "traffic_profile": {
            "prompt_tokens": 512,
            "output_tokens": 256,
            "expected_qps": 10.0,
        },
        "slo_targets": {
            "ttft_p95_target_ms": 150,
            "itl_p95_target_ms": 65,
            "e2e_p95_target_ms": 2000,
            "percentile": "p95",
        },
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


class TestPlannerClientExtractIntent:
    """Tests for intent extraction."""

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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


class TestPlannerClientGetRecommendations:
    """Tests for getting recommendations."""

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
    def test_get_recommendations(self, mock_httpx: MagicMock) -> None:
        """Ranked recommendations are fetched and parsed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_RANKED_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.get_recommendations(
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        assert len(result["balanced"]) == 1
        assert result["total_configs_evaluated"] == 2847

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
    def test_get_recommendations_with_constraints(self, mock_httpx: MagicMock) -> None:
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
        client.get_recommendations(
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=100,
            itl_target_ms=30,
            e2e_target_ms=1500,
            min_accuracy=70,
            max_cost=5000.0,
            weights={"accuracy": 2, "price": 2, "latency": 8, "complexity": 1},
            percentile="p99",
        )

        # Verify the payload sent to the API
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["min_accuracy"] == 70
        assert payload["max_cost"] == 5000.0
        assert payload["weights"] == {"accuracy": 2, "price": 2, "latency": 8, "complexity": 1}
        assert payload["percentile"] == "p99"

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
    def test_get_recommendations_without_constraints(self, mock_httpx: MagicMock) -> None:
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
        client.get_recommendations(
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
        )

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "min_accuracy" not in payload
        assert "max_cost" not in payload
        assert "weights" not in payload
        assert payload["percentile"] == "p95"


class TestPlannerClientRequestErrors:
    """Tests for _request error handling edge cases."""

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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
        assert "llm-d-planner available" in msg
        # Verify health check uses /health endpoint (not /api/v1/)
        mock_client.get.assert_called_once_with("http://localhost:8000/health", params=None)

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
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
        assert "llm-d-planner unavailable" in msg


class TestPlannerClientDeploy:
    """Tests for deploy() method."""

    @patch("rhoai_mcp.domains.llm_d_planner.client.httpx")
    def test_deploy(self, mock_httpx: MagicMock) -> None:
        """deploy() sends recommendation + namespace and returns response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_DEPLOY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)

        client = PlannerClient("http://localhost:8000")
        result = client.deploy(SAMPLE_RECOMMENDATION, namespace="ml-prod")

        # Verify correct endpoint and payload
        call_args = mock_client.post.call_args
        assert "/api/v1/deploy" in call_args.args[0]
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["recommendation"] == SAMPLE_RECOMMENDATION
        assert payload["namespace"] == "ml-prod"

        # Verify response passthrough
        assert result["deployment_id"] == "chatbot-llama-3-1-70b-20260322143022"
        assert "inferenceservice" in result["files"]
