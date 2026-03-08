"""Tests for NeuralNav composite models."""

from rhoai_mcp.composites.neuralnav.models import (
    DeploymentIntent,
    GPUConfig,
    ModelRecommendation,
    RecommendationResult,
    RecommendationScores,
    SLOTargets,
    TrafficProfile,
)


class TestDeploymentIntent:
    """Tests for DeploymentIntent model."""

    def test_minimal_intent(self) -> None:
        """Intent with only required fields."""
        intent = DeploymentIntent(
            use_case="chatbot_conversational",
            user_count=1000,
        )
        assert intent.use_case == "chatbot_conversational"
        assert intent.user_count == 1000
        assert intent.preferred_gpu_types == []
        assert intent.accuracy_priority == "medium"

    def test_full_intent(self) -> None:
        """Intent with all fields populated."""
        intent = DeploymentIntent(
            use_case="code_completion",
            user_count=5000,
            experience_class="instant",
            preferred_gpu_types=["H100", "A100-80"],
            accuracy_priority="high",
            cost_priority="low",
        )
        assert intent.preferred_gpu_types == ["H100", "A100-80"]
        assert intent.accuracy_priority == "high"


class TestModelRecommendation:
    """Tests for ModelRecommendation model."""

    def test_recommendation_from_dict(self) -> None:
        """Recommendation can be built from API response dict."""
        rec = ModelRecommendation(
            model_id="meta-llama/Llama-3.1-70B-Instruct",
            model_name="Llama 3.1 70B",
            gpu_config=GPUConfig(
                gpu_type="NVIDIA-H100",
                gpu_count=2,
                tensor_parallel=2,
                replicas=1,
            ),
            predicted_ttft_p95_ms=140,
            predicted_itl_p95_ms=50,
            predicted_e2e_p95_ms=1200,
            predicted_throughput_qps=100.0,
            cost_per_hour_usd=3.98,
            cost_per_month_usd=2872.32,
            meets_slo=True,
            reasoning="Selected for chatbot use case",
            scores=RecommendationScores(
                accuracy_score=78,
                price_score=65,
                latency_score=95,
                complexity_score=90,
                balanced_score=75.3,
                slo_status="compliant",
            ),
        )
        assert rec.model_id == "meta-llama/Llama-3.1-70B-Instruct"
        assert rec.gpu_config.gpu_count == 2
        assert rec.scores.balanced_score == 75.3


class TestRecommendationResult:
    """Tests for RecommendationResult model."""

    def test_empty_result(self) -> None:
        """Result with no recommendations."""
        result = RecommendationResult(
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
            recommendations=[],
            total_configs_evaluated=2847,
            configs_after_filters=0,
        )
        assert len(result.recommendations) == 0
        assert result.total_configs_evaluated == 2847


class TestSLOTargets:
    """Tests for SLOTargets model."""

    def test_slo_targets(self) -> None:
        """SLO targets can be constructed."""
        slo = SLOTargets(ttft_ms=150, itl_ms=65, e2e_ms=2000)
        assert slo.ttft_ms == 150


class TestTrafficProfile:
    """Tests for TrafficProfile model."""

    def test_traffic_profile(self) -> None:
        """Traffic profile can be constructed."""
        tp = TrafficProfile(prompt_tokens=512, output_tokens=256, expected_qps=10.0)
        assert tp.expected_qps == 10.0
