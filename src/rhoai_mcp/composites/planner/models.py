"""Pydantic models for Planner API request/response types."""

from __future__ import annotations

from typing import Any, Literal

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


class GpuPreference(BaseModel):
    """GPU preference with optional count constraint."""

    gpu_type: str = Field(..., description="GPU type name (e.g., H100, L4)")
    max_count: int | None = Field(None, description="Maximum GPU count for this type")


class DeploymentIntent(BaseModel):
    """Extracted deployment intent from natural language."""

    use_case: UseCaseType = Field(..., description="Primary use case type")
    user_count: int = Field(..., description="Number of users or scale")
    domain_specialization: list[str] = Field(
        default_factory=lambda: ["general"], description="Domain requirements"
    )
    preferred_gpu_types: list[str | GpuPreference] = Field(
        default_factory=list, description="Preferred GPU types (empty = any)"
    )
    preferred_models: list[str] = Field(
        default_factory=list, description="Preferred model identifiers"
    )
    quality_priority: PriorityType = Field(default="medium", description="Quality importance")
    cost_priority: PriorityType = Field(default="medium", description="Cost sensitivity")
    latency_priority: PriorityType = Field(default="medium", description="Latency importance")


class GPUConfig(BaseModel):
    """GPU configuration for a recommendation."""

    gpu_type: str = Field(..., description="GPU type (e.g., NVIDIA-H100)")
    gpu_count: int = Field(..., description="Total number of GPUs")
    tensor_parallel: int = Field(1, description="Tensor parallelism degree")
    replicas: int = Field(1, description="Number of replicas")


class SLORange(BaseModel):
    """Range for an SLO metric."""

    min: int = Field(..., description="Minimum value")
    max: int = Field(..., description="Maximum value")


class SLOTargets(BaseModel):
    """SLO targets used for the recommendation."""

    ttft_target_ms: int = Field(..., description="Time to First Token target (ms)")
    itl_target_ms: int = Field(..., description="Inter-Token Latency target (ms)")
    e2e_target_ms: int = Field(..., description="End-to-end latency target (ms)")
    percentile: str = Field(default="p95", description="Percentile for SLO comparison")
    ttft_range: SLORange | None = Field(None, description="Recommended TTFT range")
    itl_range: SLORange | None = Field(None, description="Recommended ITL range")
    e2e_range: SLORange | None = Field(None, description="Recommended E2E range")


class TrafficProfile(BaseModel):
    """Traffic profile used for the recommendation."""

    prompt_tokens: int = Field(..., description="Target prompt length in tokens")
    output_tokens: int = Field(..., description="Target output length in tokens")
    expected_qps: float = Field(..., description="Expected queries per second")


class WorkloadProfile(BaseModel):
    """Workload profile from the specification endpoint."""

    prompt_tokens: int = Field(..., description="Mean input token length per request")
    output_tokens: int = Field(..., description="Mean output token length per request")
    expected_qps: float = Field(..., description="Expected queries per second")


class QualityWeights(BaseModel):
    """Per-use-case category weights for quality scoring."""

    categories: dict[str, int] = Field(..., description="Category name to weight mapping")


class PriorityEntry(BaseModel):
    """A priority with its resolved numeric weight."""

    priority: PriorityType = Field(..., description="Priority level")
    weight: int = Field(..., description="Resolved numeric weight")


class Priorities(BaseModel):
    """Resolved priority weights for scoring."""

    quality: PriorityEntry = Field(..., description="Quality priority and weight")
    cost: PriorityEntry = Field(..., description="Cost priority and weight")
    latency: PriorityEntry = Field(..., description="Latency priority and weight")


class DeploymentSpecification(BaseModel):
    """Complete deployment specification generated from intent."""

    intent: DeploymentIntent = Field(..., description="Original deployment intent")
    slo_targets: SLOTargets = Field(..., description="SLO targets")
    workload_profile: WorkloadProfile = Field(..., description="Workload profile")
    quality_weights: QualityWeights | None = Field(
        None, description="Per-use-case quality scoring weights"
    )
    priorities: Priorities = Field(..., description="Resolved priority weights")


class DeploymentConfiguration(BaseModel):
    """Parameters for generating deployment YAML files."""

    model_config = {"protected_namespaces": ()}

    model_id: str = Field(..., description="Model identifier (HuggingFace format)")
    model_name: str | None = Field(None, description="Human-readable model name")
    model_uri: str | None = Field(None, description="Model artifact URI")
    gpu_config: GPUConfig = Field(..., description="GPU configuration")
    use_case: str = Field(..., description="Use case")
    expected_qps: float = Field(..., description="Expected queries per second")
    prompt_tokens: int = Field(..., description="Mean input token length")
    output_tokens: int = Field(..., description="Mean output token length")
    e2e_target_ms: int = Field(..., description="End-to-end latency target (ms)")


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


class DeploymentBundle(BaseModel):
    """Bundle of generated deployment YAML files."""

    deployment_id: str = Field(..., description="Unique deployment identifier")
    namespace: str = Field(..., description="Kubernetes namespace")
    stack: str = Field(..., description="Deployment stack (vllm or llm-d)")
    configuration: DeploymentConfiguration | None = Field(
        None, description="Configuration used to generate files"
    )
    files: dict[str, str] = Field(
        default_factory=dict, description="Filename to YAML content mapping"
    )


class DeploymentConfigResult(BaseModel):
    """Result of deployment config generation."""

    deployment_id: str = Field(..., description="Generated deployment identifier")
    namespace: str = Field(..., description="Target Kubernetes namespace")
    model_name: str | None = Field(None, description="Human-readable model name")
    configs: dict[str, str] = Field(..., description="Config type to YAML content mapping")
