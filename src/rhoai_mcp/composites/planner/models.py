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


# ---------------------------------------------------------------------------
# Cluster-aware recommendation models (Skill 1 enhancement)
# ---------------------------------------------------------------------------

ClusterFitStatus = Literal["available", "partial", "unavailable"]


class ClusterGPU(BaseModel):
    """A GPU type available on the cluster."""

    product: str = Field(..., description="GPU product name (e.g., NVIDIA-H100-80GB-HBM3)")
    total: int = Field(0, description="Total GPUs of this type")
    available: int = Field(0, description="Available GPUs of this type")
    nodes: int = Field(0, description="Number of nodes with this GPU type")


class ClusterFitResult(BaseModel):
    """Result of checking a recommendation against cluster GPU availability."""

    status: ClusterFitStatus = Field(..., description="Fit status")
    gpu_type: str = Field(..., description="GPU type from the recommendation")
    needed: int = Field(..., description="Total GPUs needed (count * replicas)")
    available: int = Field(0, description="Available GPUs of this type on cluster")
    message: str = Field("", description="Human-readable fit explanation")


# ---------------------------------------------------------------------------
# Deployment planning models (Skill 2 - plan_deployment)
# ---------------------------------------------------------------------------

IssueCategoryType = Literal["runtime", "storage", "gpu", "namespace", "other"]


class DeploymentPlanIssue(BaseModel):
    """An issue discovered during deployment planning."""

    category: IssueCategoryType = Field(..., description="Issue category")
    message: str = Field(..., description="Description of the issue")
    blocking: bool = Field(..., description="Whether this blocks deployment")
    suggestion: str | None = Field(None, description="Suggested fix")


class DeploymentPlanStep(BaseModel):
    """A step in the deployment execution plan."""

    action: str = Field(..., description="Action identifier")
    description: str = Field(..., description="Human-readable description")


class ResolvedDeployParams(BaseModel):
    """Fully resolved parameters for deploy_model."""

    model_config = {"protected_namespaces": ()}

    name: str = Field(..., description="DNS-safe deployment name")
    namespace: str = Field(..., description="Target namespace")
    display_name: str | None = Field(None, description="Human-readable display name")
    runtime: str = Field(..., description="Serving runtime name")
    model_format: str = Field("pytorch", description="Model format")
    storage_uri: str = Field(..., description="Model artifact location")
    min_replicas: int = Field(1, ge=0, description="Minimum replicas")
    max_replicas: int = Field(1, ge=1, description="Maximum replicas")
    cpu_request: str = Field("8", description="CPU request per replica")
    cpu_limit: str = Field("16", description="CPU limit per replica")
    memory_request: str = Field("32Gi", description="Memory request per replica")
    memory_limit: str = Field("64Gi", description="Memory limit per replica")
    gpu_count: int = Field(1, ge=0, description="GPUs per replica")


class DeploymentPlan(BaseModel):
    """A reviewable deployment plan ready for execution."""

    ready: bool = Field(..., description="Whether the plan can be executed (no blocking issues)")
    recommendation_summary: dict[str, Any] = Field(
        ..., description="Summary of the source recommendation"
    )
    resolved_params: ResolvedDeployParams = Field(
        ..., description="Fully resolved deployment parameters"
    )
    steps: list[DeploymentPlanStep] = Field(..., description="Ordered execution steps")
    issues: list[DeploymentPlanIssue] = Field(
        default_factory=list, description="Issues found during planning"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-blocking warnings")


# ---------------------------------------------------------------------------
# Deployment execution models (Skill 3 - execute_deployment)
# ---------------------------------------------------------------------------


class EndpointValidation(BaseModel):
    """Result of testing a deployed model endpoint."""

    reachable: bool = Field(..., description="Whether the endpoint responded")
    response_time_ms: float | None = Field(None, description="Response time in milliseconds")
    status: str | None = Field(None, description="InferenceService status")
    url: str | None = Field(None, description="Endpoint URL")
    message: str = Field("", description="Validation message")


class DeploymentResult(BaseModel):
    """Result of executing a deployment plan."""

    success: bool = Field(..., description="Whether deployment succeeded")
    message: str = Field(..., description="Result summary")
    deployment_name: str | None = Field(None, description="InferenceService name")
    namespace: str | None = Field(None, description="Deployment namespace")
    status: str | None = Field(None, description="Final InferenceService status")
    endpoint_url: str | None = Field(None, description="Inference endpoint URL")
    validation: EndpointValidation | None = Field(None, description="Endpoint validation results")
    slo_comparison: dict[str, Any] | None = Field(
        None, description="Predicted vs. actual SLO comparison"
    )
    issues: list[DeploymentPlanIssue] | None = Field(
        None, description="Issues if deployment failed"
    )
