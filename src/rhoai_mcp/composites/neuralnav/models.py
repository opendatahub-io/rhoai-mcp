"""Pydantic models for NeuralNav API request/response types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DeploymentIntent(BaseModel):
    """Extracted deployment intent from natural language."""

    use_case: str = Field(..., description="Primary use case type")
    user_count: int = Field(..., description="Number of users or scale")
    experience_class: str = Field(default="conversational", description="User experience class")
    preferred_gpu_types: list[str] = Field(
        default_factory=list, description="Preferred GPU types (empty = any)"
    )
    accuracy_priority: str = Field(default="medium", description="Accuracy importance")
    cost_priority: str = Field(default="medium", description="Cost sensitivity")
    latency_priority: str = Field(default="medium", description="Latency importance")
    complexity_priority: str = Field(default="medium", description="Simplicity preference")
    domain_specialization: list[str] = Field(
        default_factory=lambda: ["general"], description="Domain requirements"
    )
    additional_context: str | None = Field(None, description="Extra context")


class GPUConfig(BaseModel):
    """GPU configuration for a recommendation."""

    gpu_type: str = Field(..., description="GPU type (e.g., NVIDIA-H100)")
    gpu_count: int = Field(..., description="Total number of GPUs")
    tensor_parallel: int = Field(1, description="Tensor parallelism degree")
    replicas: int = Field(1, description="Number of replicas")


class RecommendationScores(BaseModel):
    """Multi-criteria scores for a recommendation (0-100 scale)."""

    accuracy_score: int = Field(..., description="Model capability score")
    price_score: int = Field(..., description="Cost efficiency score")
    latency_score: int = Field(..., description="SLO headroom score")
    complexity_score: int = Field(..., description="Deployment simplicity score")
    balanced_score: float = Field(..., description="Weighted composite score")
    slo_status: str = Field(..., description="SLO compliance: compliant|near_miss|exceeds")


class ModelRecommendation(BaseModel):
    """A single model recommendation from NeuralNav."""

    model_id: str | None = Field(None, description="Model identifier")
    model_name: str | None = Field(None, description="Human-readable model name")
    gpu_config: GPUConfig | None = Field(None, description="GPU configuration")
    predicted_ttft_p95_ms: int | None = Field(None, description="Predicted TTFT p95 (ms)")
    predicted_itl_p95_ms: int | None = Field(None, description="Predicted ITL p95 (ms)")
    predicted_e2e_p95_ms: int | None = Field(None, description="Predicted E2E p95 (ms)")
    predicted_throughput_qps: float | None = Field(None, description="Predicted throughput")
    cost_per_hour_usd: float | None = Field(None, description="Cost per hour (USD)")
    cost_per_month_usd: float | None = Field(None, description="Cost per month (USD)")
    meets_slo: bool = Field(False, description="Whether config meets SLO targets")
    reasoning: str = Field(..., description="Recommendation reasoning")
    scores: RecommendationScores | None = Field(None, description="Multi-criteria scores")


class SLOTargets(BaseModel):
    """SLO targets used for the recommendation."""

    ttft_ms: int = Field(..., description="Time to First Token target (ms)")
    itl_ms: int = Field(..., description="Inter-Token Latency target (ms)")
    e2e_ms: int = Field(..., description="End-to-end latency target (ms)")


class TrafficProfile(BaseModel):
    """Traffic profile used for the recommendation."""

    prompt_tokens: int = Field(..., description="Target prompt length in tokens")
    output_tokens: int = Field(..., description="Target output length in tokens")
    expected_qps: float = Field(..., description="Expected queries per second")


class RecommendationResult(BaseModel):
    """Complete recommendation result returned by the tool."""

    specification: dict[str, Any] = Field(
        ...,
        description="Assembled specification (use_case, SLO targets, traffic profile)",
    )
    recommendations: list[ModelRecommendation] = Field(
        ..., description="Top-5 balanced model recommendations"
    )
    total_configs_evaluated: int = Field(0, description="Total configs evaluated")
    configs_after_filters: int = Field(0, description="Configs after filtering")
