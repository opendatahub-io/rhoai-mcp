"""Pydantic models for Planner composite tools.

Shared types are re-exported from the planner package (canonical source).
Tool-specific output types are defined locally.
"""

from __future__ import annotations

from typing import Any, Literal

# Re-export from planner's public API
from planner import (
    DeploymentBundle,
    DeploymentConfiguration,
    DeploymentIntent,
    DeploymentSpecification,
    GPUConfig,
    GpuPreference,
    SLOTargets,
    WorkloadProfile,
)

# Re-export from planner's shared schemas (not in planner.__all__)
from planner.shared.schemas import (
    Priorities,
    PriorityEntry,
    QualityWeights,
    SLORange,
    TrafficProfile,
)
from pydantic import BaseModel, Field

SloStatusType = Literal["compliant", "near_miss", "exceeds"]

UseCaseType = Literal[
    "chatbot_conversational",
    "code_completion",
    "code_generation_detailed",
    "translation",
    "content_generation",
    "summarization_short",
    "document_analysis_rag",
    "long_document_summarization",
    "research_legal_analysis",
]

PriorityType = Literal["low", "medium", "high"]


class ConfigurationScores(BaseModel):
    """Multi-criteria scores for a recommendation (0-100 scale)."""

    quality_score: float = Field(..., ge=0, le=100, description="Model quality/capability score")
    price_score: float = Field(..., ge=0, le=100, description="Cost efficiency score")
    latency_score: float = Field(..., ge=0, le=100, description="SLO headroom score")
    balanced_score: float = Field(..., ge=0, le=100, description="Weighted composite score")
    slo_status: SloStatusType = Field(..., description="SLO compliance status")


class ModelRecommendation(BaseModel):
    """A single model recommendation from Planner."""

    model_config = {"protected_namespaces": ()}

    model_id: str | None = Field(None, description="Model identifier")
    model_name: str | None = Field(None, description="Human-readable model name")
    model_uri: str | None = Field(None, description="Model artifact URI")
    gpu_config: GPUConfig | None = Field(None, description="GPU configuration")
    predicted_ttft_p95_ms: int | None = Field(None, description="Predicted TTFT p95 (ms)")
    predicted_itl_p95_ms: int | None = Field(None, description="Predicted ITL p95 (ms)")
    predicted_e2e_p95_ms: int | None = Field(None, description="Predicted E2E p95 (ms)")
    predicted_throughput_qps: float | None = Field(None, description="Predicted throughput")
    benchmark_metrics: dict[str, Any] | None = Field(None, description="Benchmark metrics")
    cost_per_hour_usd: float | None = Field(None, description="Cost per hour (USD)")
    cost_per_month_usd: float | None = Field(None, description="Cost per month (USD)")
    meets_slo: bool = Field(False, description="Whether config meets SLO targets")
    reasoning: str = Field(..., description="Recommendation reasoning")
    alternative_options: list[dict[str, Any]] | None = Field(
        None, description="Alternative configurations"
    )
    scores: ConfigurationScores | None = Field(None, description="Multi-criteria scores")
    configuration: DeploymentConfiguration | None = Field(
        None, description="Deployment configuration for YAML generation"
    )


class RecommendationResult(BaseModel):
    """Complete recommendation result returned by the tool."""

    specification: dict[str, Any] = Field(
        ...,
        description="Assembled specification (use_case, SLO targets, traffic profile)",
    )
    top_performance: ModelRecommendation | None = Field(
        None, description="Top model for lowest latency"
    )
    top_cost: ModelRecommendation | None = Field(None, description="Top model for lowest cost")
    top_balanced: ModelRecommendation | None = Field(
        None, description="Top model for balanced score"
    )
    top_quality: ModelRecommendation | None = Field(None, description="Top model for best quality")
    total_configs_evaluated: int = Field(0, description="Total configs evaluated")
    configs_after_filters: int = Field(0, description="Configs after filtering")


class DeploymentConfigResult(BaseModel):
    """Result of deployment config generation."""

    deployment_id: str = Field(..., description="Generated deployment identifier")
    namespace: str = Field(..., description="Target Kubernetes namespace")
    model_name: str | None = Field(None, description="Human-readable model name")
    configs: dict[str, str] = Field(..., description="Config type to YAML content mapping")


__all__ = [
    "ConfigurationScores",
    "DeploymentBundle",
    "DeploymentConfigResult",
    "DeploymentConfiguration",
    "DeploymentIntent",
    "DeploymentSpecification",
    "GPUConfig",
    "GpuPreference",
    "ModelRecommendation",
    "Priorities",
    "PriorityEntry",
    "PriorityType",
    "QualityWeights",
    "RecommendationResult",
    "SLORange",
    "SLOTargets",
    "SloStatusType",
    "TrafficProfile",
    "UseCaseType",
    "WorkloadProfile",
]
