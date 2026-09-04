"""Tests for LocalPlannerClient (embedded Planner library)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rhoai_mcp.composites.planner.client import PlannerAPIError
from rhoai_mcp.composites.planner.local_client import LocalPlannerClient


def _make_mock_planner() -> MagicMock:
    """Create a mock Planner instance with standard responses."""
    mock = MagicMock()

    mock.generate_specification.return_value.model_dump.return_value = {
        "intent": {
            "use_case": "chatbot_conversational",
            "user_count": 1000,
            "preferred_gpu_types": ["H100"],
        },
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

    mock.generate_recommendations.return_value.model_dump.return_value = {
        "balanced": [
            {
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
                "reasoning": "Best balanced option",
                "scores": {
                    "quality_score": 78,
                    "price_score": 65,
                    "latency_score": 95,
                    "balanced_score": 75.3,
                    "slo_status": "compliant",
                },
                "configuration": {
                    "model_id": "meta-llama/Llama-3.1-70B-Instruct",
                    "model_name": "Llama 3.1 70B",
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
                },
            }
        ],
        "lowest_cost": [],
        "lowest_latency": [],
        "best_quality": [],
    }
    mock.generate_recommendations.return_value.total_configs_evaluated = 2847
    mock.generate_recommendations.return_value.configs_after_filters = 542

    bundle_mock = MagicMock()
    bundle_mock.deployment_id = "chatbot-llama-20260322"
    bundle_mock.namespace = "default"
    bundle_mock.files = {
        "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService",
    }
    mock.generate_deployment.return_value = bundle_mock

    return mock


class TestLocalPlannerClientInit:
    """Tests for LocalPlannerClient initialization."""

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_init_loads_bundled_benchmarks_when_no_catalog(
        self, mock_planner_cls: MagicMock
    ) -> None:
        """Bundled benchmarks are loaded as fallback when no catalog URL."""
        LocalPlannerClient()

        mock_planner_cls.assert_called_once()
        mock_planner_cls.return_value.load_bundled_benchmarks.assert_called_once()
        mock_planner_cls.return_value.sync_model_catalog.assert_not_called()

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_init_with_catalog_sync(self, mock_planner_cls: MagicMock) -> None:
        """Model Catalog sync is used and bundled benchmarks are skipped."""
        mock_planner_cls.return_value.sync_model_catalog.return_value = {
            "benchmarks_added": 10,
            "models_added": 5,
            "errors": [],
        }

        LocalPlannerClient(
            model_catalog_url="https://catalog.example.com",
        )

        mock_planner_cls.return_value.sync_model_catalog.assert_called_once_with(
            url="https://catalog.example.com",
        )
        mock_planner_cls.return_value.load_bundled_benchmarks.assert_not_called()

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_init_catalog_sync_import_error(self, mock_planner_cls: MagicMock) -> None:
        """ImportError from sync_model_catalog is mapped to PlannerAPIError."""
        mock_planner_cls.return_value.sync_model_catalog.side_effect = ImportError(
            "httpx required"
        )

        with pytest.raises(PlannerAPIError, match="extra dependency"):
            LocalPlannerClient(model_catalog_url="https://catalog.example.com")

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_init_catalog_sync_value_error(self, mock_planner_cls: MagicMock) -> None:
        """ValueError from sync_model_catalog is mapped to PlannerAPIError(502)."""
        mock_planner_cls.return_value.sync_model_catalog.side_effect = ValueError("bad url")

        with pytest.raises(PlannerAPIError) as exc_info:
            LocalPlannerClient(model_catalog_url="https://catalog.example.com")

        assert exc_info.value.status_code == 502
        assert "Model Catalog sync failed" in exc_info.value.detail

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_init_catalog_sync_planner_error(self, mock_planner_cls: MagicMock) -> None:
        """PlannerError from sync_model_catalog is mapped to PlannerAPIError(502)."""
        from planner import PlannerError

        mock_planner_cls.return_value.sync_model_catalog.side_effect = PlannerError(
            "unreachable"
        )

        with pytest.raises(PlannerAPIError) as exc_info:
            LocalPlannerClient(model_catalog_url="https://catalog.example.com")

        assert exc_info.value.status_code == 502
        assert "Model Catalog sync failed" in exc_info.value.detail


class TestLocalPlannerRecommend:
    """Tests for LocalPlannerClient.recommend()."""

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_full_flow(self, mock_planner_cls: MagicMock) -> None:
        """Full recommend flow with all overrides works."""
        mock_planner = _make_mock_planner()
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
        result = client.recommend(
            text="ignored in local mode",
            use_case_override="chatbot_conversational",
            user_count_override=1000,
            gpu_types_override=["H100"],
        )

        assert result.specification["use_case"] == "chatbot_conversational"
        assert result.specification["user_count"] == 1000
        assert result.top_balanced is not None
        assert result.top_balanced.model_name == "Llama 3.1 70B"
        assert result.total_configs_evaluated == 2847

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_missing_overrides_raises(self, mock_planner_cls: MagicMock) -> None:
        """Missing required overrides raises PlannerAPIError(400)."""
        client = LocalPlannerClient()

        with pytest.raises(PlannerAPIError) as exc_info:
            client.recommend(text="chatbot for 1000 users")

        assert exc_info.value.status_code == 400
        assert "use_case" in exc_info.value.detail

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_missing_use_case_raises(self, mock_planner_cls: MagicMock) -> None:
        """Missing use_case override raises PlannerAPIError(400)."""
        client = LocalPlannerClient()

        with pytest.raises(PlannerAPIError) as exc_info:
            client.recommend(
                text="chatbot",
                user_count_override=1000,
                gpu_types_override=["H100"],
            )

        assert exc_info.value.status_code == 400

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_slo_overrides_applied(self, mock_planner_cls: MagicMock) -> None:
        """SLO overrides are applied to the specification."""
        mock_planner = _make_mock_planner()
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
        client.recommend(
            text="test",
            use_case_override="chatbot_conversational",
            user_count_override=1000,
            gpu_types_override=["H100"],
            ttft_override_ms=100,
            itl_override_ms=30,
            e2e_override_ms=1500,
        )

        spec_call = mock_planner.generate_recommendations.call_args
        spec_arg = spec_call[0][0] if spec_call[0] else spec_call[1]["spec"]
        assert spec_arg.slo_targets.ttft_target_ms == 100
        assert spec_arg.slo_targets.itl_target_ms == 30
        assert spec_arg.slo_targets.e2e_target_ms == 1500

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_priority_overrides_applied(self, mock_planner_cls: MagicMock) -> None:
        """Priority weight overrides are applied to the specification."""
        mock_planner = _make_mock_planner()
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
        client.recommend(
            text="test",
            use_case_override="chatbot_conversational",
            user_count_override=1000,
            gpu_types_override=["H100"],
            priority_weights={"quality": 8, "price": 2, "latency": 1},
        )

        spec_call = mock_planner.generate_recommendations.call_args
        spec_arg = spec_call[0][0] if spec_call[0] else spec_call[1]["spec"]
        assert spec_arg.priorities.quality.weight == 8
        assert spec_arg.priorities.cost.weight == 2
        assert spec_arg.priorities.latency.weight == 1

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_constraints_forwarded(self, mock_planner_cls: MagicMock) -> None:
        """min_quality and max_cost are forwarded to generate_recommendations."""
        mock_planner = _make_mock_planner()
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
        client.recommend(
            text="test",
            use_case_override="chatbot_conversational",
            user_count_override=1000,
            gpu_types_override=["H100"],
            min_quality=70,
            max_cost=5000.0,
        )

        call_kwargs = mock_planner.generate_recommendations.call_args[1]
        assert call_kwargs["min_quality"] == 70
        assert call_kwargs["max_cost"] == 5000.0

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_planner_error_mapped(self, mock_planner_cls: MagicMock) -> None:
        """PlannerError is mapped to PlannerAPIError(502)."""
        from planner import PlannerError

        mock_planner = _make_mock_planner()
        mock_planner.generate_specification.side_effect = PlannerError("No benchmarks loaded")
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
        with pytest.raises(PlannerAPIError) as exc_info:
            client.recommend(
                text="test",
                use_case_override="chatbot_conversational",
                user_count_override=1000,
                gpu_types_override=["H100"],
            )

        assert exc_info.value.status_code == 502

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_recommend_value_error_mapped(self, mock_planner_cls: MagicMock) -> None:
        """ValueError is mapped to PlannerAPIError(400)."""
        mock_planner = _make_mock_planner()
        mock_planner.generate_specification.side_effect = ValueError("Unknown use case")
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
        with pytest.raises(PlannerAPIError) as exc_info:
            client.recommend(
                text="test",
                use_case_override="chatbot_conversational",
                user_count_override=1000,
                gpu_types_override=["H100"],
            )

        assert exc_info.value.status_code == 400


class TestLocalPlannerGenerateConfig:
    """Tests for LocalPlannerClient.generate_config()."""

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_generate_config_full_flow(self, mock_planner_cls: MagicMock) -> None:
        """Full generate_config flow returns deployment configs."""
        mock_planner = _make_mock_planner()
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
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

        assert result.deployment_id == "chatbot-llama-20260322"
        assert result.namespace == "default"
        assert result.model_name == "Llama 3.1 70B"
        assert "inferenceservice" in result.configs

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_generate_config_invalid_category(self, mock_planner_cls: MagicMock) -> None:
        """Invalid category raises PlannerAPIError(400)."""
        client = LocalPlannerClient()

        with pytest.raises(PlannerAPIError) as exc_info:
            client.generate_config(
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

        assert exc_info.value.status_code == 400
        assert "category" in exc_info.value.detail

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_generate_config_empty_category(self, mock_planner_cls: MagicMock) -> None:
        """Empty category list raises PlannerAPIError(404)."""
        mock_planner = _make_mock_planner()
        mock_planner.generate_recommendations.return_value.model_dump.return_value = {
            "balanced": [],
            "lowest_cost": [],
            "lowest_latency": [],
            "best_quality": [],
        }
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
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

        assert exc_info.value.status_code == 404

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_generate_config_no_files_raises(self, mock_planner_cls: MagicMock) -> None:
        """Empty files in deployment bundle raises PlannerAPIError(502)."""
        mock_planner = _make_mock_planner()
        mock_planner.generate_deployment.return_value.files = {}
        mock_planner_cls.return_value = mock_planner

        client = LocalPlannerClient()
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


class TestLocalPlannerHealthCheck:
    """Tests for LocalPlannerClient.health_check()."""

    @patch("rhoai_mcp.composites.planner.local_client.Planner")
    def test_health_check_always_healthy(self, mock_planner_cls: MagicMock) -> None:
        """Health check always returns True."""
        client = LocalPlannerClient()
        healthy, msg = client.health_check()

        assert healthy is True
        assert "local" in msg.lower()
