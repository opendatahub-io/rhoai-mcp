"""Tests for Planner composite models."""

from typing import get_args

import pytest
from pydantic import ValidationError

from rhoai_mcp.composites.planner.models import (
    ConfigurationScores,
    DeploymentBundle,
    DeploymentConfigResult,
    DeploymentConfiguration,
    DeploymentIntent,
    DeploymentSpecification,
    GPUConfig,
    ModelRecommendation,
    Priorities,
    PriorityEntry,
    RecommendationResult,
    SLORange,
    SLOTargets,
    TrafficProfile,
    UseCaseType,
    WorkloadProfile,
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
        assert intent.quality_priority == "medium"
        assert intent.preferred_models == []

    def test_full_intent(self) -> None:
        """Intent with all fields populated."""
        intent = DeploymentIntent(
            use_case="code_completion",
            user_count=5000,
            preferred_gpu_types=["H100", "A100-80"],
            preferred_models=["meta-llama/Llama-3.1-70B-Instruct"],
            quality_priority="high",
            cost_priority="low",
        )
        assert intent.preferred_gpu_types == ["H100", "A100-80"]
        assert intent.quality_priority == "high"
        assert intent.preferred_models == ["meta-llama/Llama-3.1-70B-Instruct"]

    def test_invalid_use_case_rejected(self) -> None:
        """Invalid use_case values are rejected by Pydantic validation."""
        with pytest.raises(ValidationError, match="use_case"):
            DeploymentIntent(use_case="summarization", user_count=1000)

    def test_invalid_use_case_text_summarization_rejected(self) -> None:
        """LLM-hallucinated 'text_summarization' is rejected."""
        with pytest.raises(ValidationError, match="use_case"):
            DeploymentIntent(use_case="text_summarization", user_count=1000)

    def test_invalid_priority_rejected(self) -> None:
        """Invalid priority values are rejected."""
        with pytest.raises(ValidationError, match="quality_priority"):
            DeploymentIntent(
                use_case="chatbot_conversational",
                user_count=1000,
                quality_priority="critical",
            )

    def test_all_valid_use_cases_accepted(self) -> None:
        """All valid use_case values are accepted."""
        valid_use_cases = list(get_args(UseCaseType))
        assert len(valid_use_cases) > 0
        for uc in valid_use_cases:
            intent = DeploymentIntent(use_case=uc, user_count=100)
            assert intent.use_case == uc


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
            scores=ConfigurationScores(
                quality_score=78,
                price_score=65,
                latency_score=95,
                balanced_score=75.3,
                slo_status="compliant",
            ),
        )
        assert rec.model_id == "meta-llama/Llama-3.1-70B-Instruct"
        assert rec.gpu_config.gpu_count == 2
        assert rec.scores.balanced_score == 75.3
        assert rec.scores.quality_score == 78


class TestConfigurationScores:
    """Tests for ConfigurationScores model."""

    def test_scores(self) -> None:
        """Scores can be constructed with quality_score."""
        scores = ConfigurationScores(
            quality_score=85.5,
            price_score=70,
            latency_score=90,
            balanced_score=80.0,
            slo_status="compliant",
        )
        assert scores.quality_score == 85.5
        assert scores.slo_status == "compliant"

    def test_slo_status_validates_literal(self) -> None:
        """slo_status rejects values outside the allowed set."""
        with pytest.raises(ValidationError):
            ConfigurationScores(
                quality_score=85.5,
                price_score=70,
                latency_score=90,
                balanced_score=80.0,
                slo_status="invalid_status",
            )


class TestRecommendationResult:
    """Tests for RecommendationResult model."""

    def test_empty_result(self) -> None:
        """Result with no recommendations."""
        result = RecommendationResult(
            specification={
                "use_case": "chatbot_conversational",
                "user_count": 1000,
                "slo_targets": {
                    "ttft_target_ms": 150,
                    "itl_target_ms": 65,
                    "e2e_target_ms": 2000,
                },
                "traffic_profile": {
                    "prompt_tokens": 512,
                    "output_tokens": 256,
                    "expected_qps": 10.0,
                },
            },
            total_configs_evaluated=2847,
            configs_after_filters=0,
        )
        assert result.top_performance is None
        assert result.top_cost is None
        assert result.top_balanced is None
        assert result.top_quality is None
        assert result.total_configs_evaluated == 2847

    def test_result_with_top_quality(self) -> None:
        """Result includes the top_quality field."""
        rec = ModelRecommendation(
            model_id="test-model",
            reasoning="test",
        )
        result = RecommendationResult(
            specification={"use_case": "chatbot_conversational"},
            top_quality=rec,
        )
        assert result.top_quality is not None
        assert result.top_quality.model_id == "test-model"


class TestSLOTargets:
    """Tests for SLOTargets model."""

    def test_slo_targets(self) -> None:
        """SLO targets can be constructed with new field names."""
        slo = SLOTargets(ttft_target_ms=150, itl_target_ms=65, e2e_target_ms=2000)
        assert slo.ttft_target_ms == 150
        assert slo.percentile == "p95"

    def test_slo_targets_with_ranges(self) -> None:
        """SLO targets with optional ranges."""
        slo = SLOTargets(
            ttft_target_ms=150,
            itl_target_ms=65,
            e2e_target_ms=2000,
            ttft_range=SLORange(min=50, max=200),
        )
        assert slo.ttft_range is not None
        assert slo.ttft_range.min == 50


class TestSLORange:
    """Tests for SLORange model."""

    def test_slo_range(self) -> None:
        """SLO range can be constructed."""
        r = SLORange(min=50, max=200)
        assert r.min == 50
        assert r.max == 200


class TestTrafficProfile:
    """Tests for TrafficProfile model."""

    def test_traffic_profile(self) -> None:
        """Traffic profile can be constructed."""
        tp = TrafficProfile(prompt_tokens=512, output_tokens=256, expected_qps=10.0)
        assert tp.expected_qps == 10.0


class TestDeploymentSpecification:
    """Tests for DeploymentSpecification model."""

    def test_specification(self) -> None:
        """Specification can be constructed."""
        spec = DeploymentSpecification(
            intent=DeploymentIntent(use_case="chatbot_conversational", user_count=1000),
            slo_targets=SLOTargets(ttft_target_ms=150, itl_target_ms=65, e2e_target_ms=2000),
            workload_profile=WorkloadProfile(
                prompt_tokens=512, output_tokens=256, expected_qps=10.0
            ),
            priorities=Priorities(
                quality=PriorityEntry(priority="medium", weight=4),
                cost=PriorityEntry(priority="medium", weight=4),
                latency=PriorityEntry(priority="medium", weight=4),
            ),
        )
        assert spec.intent.use_case == "chatbot_conversational"
        assert spec.slo_targets.ttft_target_ms == 150
        assert spec.quality_weights is None


class TestDeploymentConfiguration:
    """Tests for DeploymentConfiguration model."""

    def test_configuration(self) -> None:
        """Configuration can be constructed."""
        config = DeploymentConfiguration(
            model_id="meta-llama/Llama-3.1-70B-Instruct",
            model_name="Llama 3.1 70B",
            gpu_config=GPUConfig(gpu_type="NVIDIA-H100", gpu_count=2),
            use_case="chatbot_conversational",
            expected_qps=10.0,
            prompt_tokens=512,
            output_tokens=256,
            e2e_target_ms=2000,
        )
        assert config.model_id == "meta-llama/Llama-3.1-70B-Instruct"
        assert config.gpu_config.gpu_count == 2


class TestDeploymentBundle:
    """Tests for DeploymentBundle model."""

    def test_bundle(self) -> None:
        """Bundle can be constructed."""
        bundle = DeploymentBundle(
            deployment_id="chatbot-llama-20260322",
            namespace="default",
            stack="vllm",
            configuration=DeploymentConfiguration(
                model_id="meta-llama/Llama-3.1-70B-Instruct",
                gpu_config=GPUConfig(gpu_type="NVIDIA-H100", gpu_count=2),
                use_case="chatbot_conversational",
                expected_qps=10.0,
                prompt_tokens=512,
                output_tokens=256,
                e2e_target_ms=2000,
            ),
            files={"inferenceservice.yaml": "apiVersion: serving.kserve.io/v1beta1"},
        )
        assert bundle.deployment_id == "chatbot-llama-20260322"
        assert bundle.stack == "vllm"
        assert len(bundle.files) == 1


class TestDeploymentConfigResult:
    """Tests for DeploymentConfigResult model."""

    def test_full_result(self) -> None:
        """Result with all fields populated."""
        result = DeploymentConfigResult(
            deployment_id="chatbot-llama-3-1-70b-20260322143022",
            namespace="default",
            model_name="Llama 3.1 70B",
            configs={
                "inferenceservice": "apiVersion: serving.kserve.io/v1beta1\nkind: InferenceService",
                "autoscaling": "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler",
                "servicemonitor": "apiVersion: monitoring.coreos.com/v1\nkind: ServiceMonitor",
            },
        )
        assert result.deployment_id == "chatbot-llama-3-1-70b-20260322143022"
        assert result.namespace == "default"
        assert result.model_name == "Llama 3.1 70B"
        assert len(result.configs) == 3
        assert "InferenceService" in result.configs["inferenceservice"]

    def test_result_without_model_name(self) -> None:
        """Result with model_name as None."""
        result = DeploymentConfigResult(
            deployment_id="chatbot-unknown-20260322",
            namespace="ml-prod",
            configs={"inferenceservice": "yaml-content"},
        )
        assert result.model_name is None
        assert result.namespace == "ml-prod"
