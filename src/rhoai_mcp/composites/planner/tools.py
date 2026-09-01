"""MCP tools for Planner model recommendations and deployment workflow."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.composites.planner.client import (
    CATEGORY_MAP,
    PlannerAPIError,
    PlannerClient,
    PlannerConnectionError,
)
from rhoai_mcp.composites.planner.models import (
    ClusterFitResult,
    ClusterGPU,
    ModelRecommendation,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

VALID_USE_CASES: set[str] = {
    "chatbot_conversational",
    "code_completion",
    "code_generation_detailed",
    "translation",
    "content_generation",
    "summarization_short",
    "document_analysis_rag",
    "long_document_summarization",
    "research_legal_analysis",
}

VALID_GPU_TYPES: set[str] = {"L4", "A100-40", "A100-80", "H100", "H200", "B200"}
VALID_PERCENTILES: set[str] = {"mean", "p90", "p95", "p99"}
MAX_TEXT_CHARS = 4000

OPTIMIZATION_PROFILES: dict[str, dict[str, int]] = {
    "balanced": {"quality": 4, "price": 4, "latency": 2},
    "optimize_latency": {"quality": 2, "price": 2, "latency": 8},
    "optimize_cost": {"quality": 2, "price": 8, "latency": 1},
    "optimize_quality": {"quality": 8, "price": 2, "latency": 1},
}

VALID_CATEGORIES: set[str] = set(CATEGORY_MAP)
_K8S_NAMESPACE_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")

_GPU_NAME_ALIASES: dict[str, list[str]] = {
    "h100": ["h100"],
    "h200": ["h200"],
    "b200": ["b200"],
    "a100-80": ["a100", "80gb", "a100-sxm"],
    "a100-40": ["a100", "40gb"],
    "l4": ["l4"],
}


def _check_cluster_gpu_fit(
    server: RHOAIServer,
    recommendations: dict[str, Any],
) -> tuple[dict[str, ClusterFitResult], list[ClusterGPU]]:
    """Cross-reference recommendations against actual cluster GPUs.

    Returns a mapping of slot name -> ClusterFitResult, plus the list of
    cluster GPU types.
    """
    try:
        from rhoai_mcp.domains.training.client import TrainingClient

        training_client = TrainingClient(server.k8s)
        resources = training_client.get_cluster_resources()
    except Exception as e:
        logger.debug("Cluster GPU check failed: %s", e)
        return {}, []

    cluster_gpus: list[ClusterGPU] = []
    if resources.gpu_info:
        for product in resources.gpu_info.products:
            cluster_gpus.append(
                ClusterGPU(
                    product=product,
                    total=resources.gpu_info.total,
                    available=resources.gpu_info.available,
                    nodes=resources.gpu_info.nodes_with_gpu,
                )
            )
    if not resources.has_gpus:
        cluster_gpus = []

    fit_results: dict[str, ClusterFitResult] = {}
    for slot, rec in recommendations.items():
        if not isinstance(rec, dict):
            continue
        gpu_str = rec.get("gpu", "")
        if not gpu_str:
            continue

        gpu_type = _extract_gpu_type(gpu_str)
        needed = _extract_gpu_count(gpu_str)
        matched_available = _match_gpu_on_cluster(gpu_type, resources)

        if matched_available is None:
            fit_results[slot] = ClusterFitResult(
                status="unavailable",
                gpu_type=gpu_type,
                needed=needed,
                available=0,
                message=f"{gpu_type} not found on cluster",
            )
        elif matched_available >= needed:
            fit_results[slot] = ClusterFitResult(
                status="available",
                gpu_type=gpu_type,
                needed=needed,
                available=matched_available,
                message=f"{needed}x {gpu_type} available ({matched_available} total)",
            )
        else:
            fit_results[slot] = ClusterFitResult(
                status="partial",
                gpu_type=gpu_type,
                needed=needed,
                available=matched_available,
                message=f"Need {needed}x {gpu_type} but only {matched_available} available",
            )

    return fit_results, cluster_gpus


def _extract_gpu_type(gpu_str: str) -> str:
    """Extract GPU type from a formatted string like '4x H100'."""
    parts = gpu_str.split()
    return parts[-1] if parts else gpu_str


def _extract_gpu_count(gpu_str: str) -> int:
    """Extract GPU count from a formatted string like '4x H100'."""
    match = re.match(r"(\d+)x", gpu_str)
    return int(match.group(1)) if match else 1


def _match_gpu_on_cluster(
    gpu_type: str,
    resources: Any,
) -> int | None:
    """Check if a GPU type is available on the cluster. Returns available count or None."""
    if not resources.has_gpus or not resources.gpu_info:
        return None

    gpu_lower = gpu_type.lower().replace("-", "")
    aliases = _GPU_NAME_ALIASES.get(gpu_type.lower(), [gpu_lower])

    for product in resources.gpu_info.products:
        product_lower = product.lower().replace("-", "")
        if any(alias in product_lower for alias in aliases):
            return int(resources.gpu_info.available)

    return None


def _format_recommendation(rec: ModelRecommendation, slot: str) -> dict[str, Any]:
    """Format a single recommendation compactly for LLM context."""
    compact: dict[str, Any] = {}
    if rec.model_name:
        compact["model"] = rec.model_name
    elif rec.model_id:
        compact["model"] = rec.model_id
    if rec.gpu_config:
        gpu = rec.gpu_config
        compact["gpu"] = f"{gpu.gpu_count}x {gpu.gpu_type}"
    if rec.cost_per_month_usd is not None:
        compact["cost_usd_month"] = rec.cost_per_month_usd
    compact["meets_slo"] = rec.meets_slo
    if slot == "top_balanced" and rec.scores:
        compact["score"] = rec.scores.balanced_score
    if slot == "top_quality" and rec.scores:
        compact["score"] = rec.scores.quality_score
    if rec.reasoning:
        compact["reasoning"] = rec.reasoning
    return compact


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register Planner composite tools with the MCP server."""

    @mcp.tool()
    def recommend_model(
        text: str,
        use_case: str | None = None,
        user_count: int | None = None,
        preferred_gpu_types: list[str] | None = None,
        ttft_max_ms: int | None = None,
        itl_max_ms: int | None = None,
        e2e_max_ms: int | None = None,
        min_quality: int | None = None,
        max_cost_per_month: float | None = None,
        optimization_profile: str | None = None,
        percentile: str | None = None,
        check_cluster: bool = True,
    ) -> dict[str, Any]:
        """Get LLM model recommendations from Planner.

        Runs the full Planner recommendation flow: extracts intent from
        natural language, builds technical specifications, and returns
        four named recommendations: top_performance (lowest latency),
        top_cost (cheapest), top_balanced (weighted composite), and
        top_quality (highest quality score).

        When check_cluster is True (default), cross-references the
        recommended GPU configurations against actual cluster GPU
        availability, so you can see which recommendations are
        immediately deployable.

        Args:
            text: Natural language description of the use case
                (e.g., "I need a chatbot for 5000 users with low latency").
            use_case: Override the extracted use case. Valid values:
                chatbot_conversational, code_completion, code_generation_detailed,
                translation, content_generation, summarization_short,
                document_analysis_rag, long_document_summarization,
                research_legal_analysis.
            user_count: Override the extracted user count.
            preferred_gpu_types: Override GPU preferences.
                Valid: L4, A100-40, A100-80, H100, H200, B200.
            ttft_max_ms: Maximum time-to-first-token in milliseconds.
                Overrides the default SLO target for the use case.
            itl_max_ms: Maximum inter-token latency in milliseconds.
                Overrides the default SLO target for the use case.
            e2e_max_ms: Maximum end-to-end latency in milliseconds.
                Overrides the default SLO target for the use case.
            min_quality: Minimum model quality score (0-100).
                Filters out models below this quality threshold.
            max_cost_per_month: Maximum monthly cost in USD.
                Filters out configurations exceeding this budget.
            optimization_profile: Scoring profile that controls how
                recommendations are ranked. Valid values:
                balanced (default), optimize_latency, optimize_cost,
                optimize_quality.
            percentile: Percentile for SLO evaluation. Valid values:
                mean, p90, p95 (default), p99.
            check_cluster: Cross-reference GPU recommendations against
                actual cluster GPU availability. Default True.

        Returns:
            Four top model recommendations (top_performance, top_cost,
            top_balanced, top_quality) with assembled specification
            and optional cluster GPU fit information,
            or error dict if the request fails.
        """
        # Validate text input
        if not text or not text.strip():
            return {"error": "text must be a non-empty prompt"}

        if len(text) > MAX_TEXT_CHARS:
            return {"error": f"text exceeds max length ({MAX_TEXT_CHARS} chars)"}

        # Validate use_case if provided
        if use_case is not None and use_case not in VALID_USE_CASES:
            valid = ", ".join(sorted(VALID_USE_CASES))
            return {
                "error": f"Invalid use_case '{use_case}'. Valid values: {valid}",
            }

        # Validate percentile
        if percentile is not None and percentile not in VALID_PERCENTILES:
            valid = ", ".join(sorted(VALID_PERCENTILES))
            return {"error": f"Invalid percentile '{percentile}'. Valid values: {valid}"}

        # Validate user_count
        if user_count is not None and user_count <= 0:
            return {"error": "user_count must be > 0"}

        # Validate SLO targets
        for field_name, value in {
            "ttft_max_ms": ttft_max_ms,
            "itl_max_ms": itl_max_ms,
            "e2e_max_ms": e2e_max_ms,
        }.items():
            if value is not None and value <= 0:
                return {"error": f"{field_name} must be > 0"}

        # Validate min_quality
        if min_quality is not None and not 0 <= min_quality <= 100:
            return {"error": "min_quality must be between 0 and 100"}

        # Validate max_cost_per_month
        if max_cost_per_month is not None and max_cost_per_month < 0:
            return {"error": "max_cost_per_month must be >= 0"}

        # Validate preferred_gpu_types
        if preferred_gpu_types:
            invalid = sorted(set(preferred_gpu_types) - VALID_GPU_TYPES)
            if invalid:
                valid = ", ".join(sorted(VALID_GPU_TYPES))
                return {
                    "error": f"Invalid preferred_gpu_types {invalid}. Valid values: {valid}",
                }

        # Validate optimization_profile if provided
        if optimization_profile is not None and optimization_profile not in OPTIMIZATION_PROFILES:
            valid = ", ".join(sorted(OPTIMIZATION_PROFILES))
            return {
                "error": f"Invalid optimization_profile '{optimization_profile}'. "
                f"Valid values: {valid}",
            }

        client = PlannerClient(
            server.config.planner_url,
            timeout=float(server.config.planner_timeout),
        )

        weights = OPTIMIZATION_PROFILES.get(optimization_profile) if optimization_profile else None

        try:
            result = client.recommend(
                text,
                use_case_override=use_case,
                user_count_override=user_count,
                gpu_types_override=preferred_gpu_types,
                ttft_override_ms=ttft_max_ms,
                itl_override_ms=itl_max_ms,
                e2e_override_ms=e2e_max_ms,
                min_quality=min_quality,
                max_cost=max_cost_per_month,
                percentile_override=percentile,
                priority_weights=weights,
            )
        except PlannerConnectionError as e:
            logger.warning("Planner connection error")
            logger.debug("Planner connection error detail: %s", e)
            return {
                "error": "Planner unavailable",
                "hint": "Planner may be warming up. Retry shortly.",
            }
        except PlannerAPIError as e:
            logger.warning("Planner API error status=%s", e.status_code)
            logger.debug("Planner API error detail (truncated): %s", str(e.detail)[:512])
            return {
                "error": "Planner API error",
                "status_code": e.status_code,
            }

        # Format recommendations as 4 named categories
        recommendations: dict[str, Any] = {}
        for key, rec in [
            ("top_performance", result.top_performance),
            ("top_cost", result.top_cost),
            ("top_balanced", result.top_balanced),
            ("top_quality", result.top_quality),
        ]:
            if rec is not None:
                recommendations[key] = _format_recommendation(rec, slot=key)

        response: dict[str, Any] = {
            "specification": result.specification,
            "recommendations": recommendations,
        }

        if not recommendations:
            response["message"] = "No configurations matched the requirements"

        # Cluster GPU cross-reference
        if check_cluster and recommendations:
            fit_results, cluster_gpus = _check_cluster_gpu_fit(server, recommendations)
            if fit_results:
                response["cluster_fit"] = {
                    slot: fit.model_dump() for slot, fit in fit_results.items()
                }
            if cluster_gpus:
                response["cluster_gpus"] = [g.model_dump() for g in cluster_gpus]

        return response

    @mcp.tool()
    def get_deployment_config(
        category: str,
        use_case: str,
        user_count: int,
        prompt_tokens: int,
        output_tokens: int,
        expected_qps: float,
        ttft_target_ms: int,
        itl_target_ms: int,
        e2e_target_ms: int,
        namespace: str = "default",
        optimization_profile: str | None = None,
        preferred_gpu_types: list[str] | None = None,
        min_quality: int | None = None,
        max_cost_per_month: float | None = None,
        percentile: str | None = None,
    ) -> dict[str, Any]:
        """Generate Kubernetes deployment YAML configs for a recommended model.

        Takes the specification values from recommend_model output plus a
        category name, and returns InferenceService, HPA, and ServiceMonitor
        YAML configurations.

        Typical workflow:
        1. Call recommend_model to get recommendations with specification
        2. Call get_deployment_config with specification values + category
        3. Review or apply the generated YAML configs

        Args:
            category: Which recommendation to deploy. Valid values:
                balanced, cost, performance, quality.
            use_case: Use case from recommend_model specification.
                Valid values: chatbot_conversational, code_completion,
                code_generation_detailed, translation, content_generation,
                summarization_short, document_analysis_rag,
                long_document_summarization, research_legal_analysis.
            user_count: User count from recommend_model specification.
            prompt_tokens: Prompt tokens from recommend_model specification.
            output_tokens: Output tokens from recommend_model specification.
            expected_qps: Expected QPS from recommend_model specification.
            ttft_target_ms: TTFT target (ms) from recommend_model specification.
            itl_target_ms: ITL target (ms) from recommend_model specification.
            e2e_target_ms: E2E target (ms) from recommend_model specification.
            namespace: Kubernetes namespace for the generated config.
            optimization_profile: Scoring profile for ranking. Valid values:
                balanced, optimize_latency, optimize_cost, optimize_quality.
            preferred_gpu_types: GPU type filter.
                Valid: L4, A100-40, A100-80, H100, H200, B200.
            min_quality: Minimum quality score (0-100).
            max_cost_per_month: Maximum monthly cost in USD.
            percentile: Percentile for SLO evaluation.
                Valid: mean, p90, p95, p99.

        Returns:
            Deployment config with deployment_id, namespace, model name,
            and YAML configs (inferenceservice, autoscaling, servicemonitor),
            or error dict if the request fails.
        """
        # Validate category
        if category not in VALID_CATEGORIES:
            valid = ", ".join(sorted(VALID_CATEGORIES))
            return {"error": f"Invalid category '{category}'. Valid values: {valid}"}

        # Validate use_case
        if use_case not in VALID_USE_CASES:
            valid = ", ".join(sorted(VALID_USE_CASES))
            return {"error": f"Invalid use_case '{use_case}'. Valid values: {valid}"}

        # Validate user_count
        if user_count <= 0:
            return {"error": "user_count must be > 0"}

        # Validate token counts
        for field_name, value in {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }.items():
            if value <= 0:
                return {"error": f"{field_name} must be > 0"}

        # Validate expected_qps
        if expected_qps <= 0:
            return {"error": "expected_qps must be > 0"}

        # Validate SLO targets
        for field_name, value in {
            "ttft_target_ms": ttft_target_ms,
            "itl_target_ms": itl_target_ms,
            "e2e_target_ms": e2e_target_ms,
        }.items():
            if value <= 0:
                return {"error": f"{field_name} must be > 0"}

        # Validate namespace (must be a valid DNS-1123 label)
        if not _K8S_NAMESPACE_RE.match(namespace):
            return {
                "error": "namespace must be a valid DNS-1123 label "
                "(lowercase alphanumeric or '-', 1-63 chars, start/end alphanumeric)",
            }

        # Validate percentile
        if percentile is not None and percentile not in VALID_PERCENTILES:
            valid = ", ".join(sorted(VALID_PERCENTILES))
            return {"error": f"Invalid percentile '{percentile}'. Valid values: {valid}"}

        # Validate min_quality
        if min_quality is not None and not 0 <= min_quality <= 100:
            return {"error": "min_quality must be between 0 and 100"}

        # Validate max_cost_per_month
        if max_cost_per_month is not None and max_cost_per_month < 0:
            return {"error": "max_cost_per_month must be >= 0"}

        # Validate preferred_gpu_types
        if preferred_gpu_types:
            invalid = sorted(set(preferred_gpu_types) - VALID_GPU_TYPES)
            if invalid:
                valid = ", ".join(sorted(VALID_GPU_TYPES))
                return {
                    "error": f"Invalid preferred_gpu_types {invalid}. Valid values: {valid}",
                }

        # Resolve optimization_profile to weights
        if optimization_profile is not None and optimization_profile not in OPTIMIZATION_PROFILES:
            valid = ", ".join(sorted(OPTIMIZATION_PROFILES))
            return {
                "error": f"Invalid optimization_profile '{optimization_profile}'. "
                f"Valid values: {valid}",
            }

        weights = OPTIMIZATION_PROFILES.get(optimization_profile) if optimization_profile else None

        client = PlannerClient(
            server.config.planner_url,
            timeout=float(server.config.planner_timeout),
        )

        try:
            result = client.generate_config(
                category=category,
                use_case=use_case,
                user_count=user_count,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                expected_qps=expected_qps,
                ttft_target_ms=ttft_target_ms,
                itl_target_ms=itl_target_ms,
                e2e_target_ms=e2e_target_ms,
                namespace=namespace,
                preferred_gpu_types=preferred_gpu_types,
                min_quality=min_quality,
                max_cost=max_cost_per_month,
                percentile=percentile,
                priority_weights=weights,
            )
        except PlannerConnectionError as e:
            logger.warning("Planner connection error")
            logger.debug("Planner connection error detail: %s", e)
            return {
                "error": "Planner unavailable",
                "hint": "Planner may be warming up. Retry shortly.",
            }
        except PlannerAPIError as e:
            logger.warning("Planner API error status=%s", e.status_code)
            logger.debug("Planner API error detail (truncated): %s", str(e.detail)[:512])
            return {
                "error": "Planner API error",
                "status_code": e.status_code,
            }

        response: dict[str, Any] = {
            "deployment_id": result.deployment_id,
            "namespace": result.namespace,
            "configs": result.configs,
        }
        if result.model_name:
            response["model"] = result.model_name

        return response

    @mcp.tool()
    async def plan_deployment(
        recommendation_json: str,
        namespace: str,
        name: str | None = None,
        storage_uri: str | None = None,
        runtime: str | None = None,
    ) -> dict[str, Any]:
        """Create a deployment plan from a Planner recommendation.

        Takes a recommendation from recommend_model and resolves all
        parameters needed to deploy the model on OpenShift AI: serving
        runtime, storage URI, GPU resources, and replica count.
        Validates the plan against the live cluster.

        Typical workflow:
        1. Call recommend_model to get recommendations
        2. Call plan_deployment with the chosen recommendation JSON
        3. Review the plan, then call execute_deployment to deploy

        Args:
            recommendation_json: JSON string of a recommendation from
                recommend_model output (e.g., the value of the
                "top_balanced" key from the recommendations dict).
            namespace: Target Kubernetes namespace for deployment.
            name: Override the auto-generated deployment name.
                Must be DNS-1123 compatible if provided.
            storage_uri: Override storage URI resolution. Provide
                the model artifact location (oci://, s3://, pvc://).
            runtime: Override serving runtime selection. Provide
                the exact runtime name (e.g., "vllm-cuda-runtime").

        Returns:
            A deployment plan with resolved parameters, execution steps,
            and any issues that need resolution before deployment.
        """
        # Parse recommendation JSON
        try:
            recommendation = json.loads(recommendation_json)
        except (json.JSONDecodeError, TypeError):
            return {"error": "recommendation_json must be valid JSON"}

        if not isinstance(recommendation, dict):
            return {"error": "recommendation_json must be a JSON object"}

        # Validate namespace
        if not _K8S_NAMESPACE_RE.match(namespace):
            return {
                "error": "namespace must be a valid DNS-1123 label "
                "(lowercase alphanumeric or '-', 1-63 chars, start/end alphanumeric)",
            }

        # Validate name override if provided
        if name is not None and not _K8S_NAMESPACE_RE.match(name):
            return {
                "error": "name must be a valid DNS-1123 label "
                "(lowercase alphanumeric or '-', 1-63 chars, start/end alphanumeric)",
            }

        from rhoai_mcp.composites.planner.deployment import plan_deployment as _plan

        plan = await _plan(
            server,
            recommendation=recommendation,
            namespace=namespace,
            name_override=name,
            storage_uri_override=storage_uri,
            runtime_override=runtime,
        )

        result = plan.model_dump()

        # Add a human-readable next_action hint
        if plan.ready:
            result["next_action"] = (
                "Plan is ready. Call execute_deployment with this plan to deploy."
            )
        else:
            blocking = [i for i in plan.issues if i.blocking]
            result["next_action"] = (
                f"Resolve {len(blocking)} blocking issue(s) before deployment: "
                + "; ".join(i.message for i in blocking)
            )

        return result

    @mcp.tool()
    async def execute_deployment(
        plan_json: str,
    ) -> dict[str, Any]:
        """Execute a deployment plan to deploy a model on OpenShift AI.

        Takes a deployment plan from plan_deployment and executes it:
        creates the KServe InferenceService, waits for the model to
        become Ready, and validates the endpoint.

        This tool creates real resources on the cluster. The plan must
        have ready=true (no blocking issues).

        Typical workflow:
        1. Call recommend_model to get recommendations
        2. Call plan_deployment to create a validated plan
        3. Call execute_deployment with the plan JSON to deploy

        Args:
            plan_json: JSON string of a DeploymentPlan from plan_deployment.
                Must have ready=true (no blocking issues).

        Returns:
            Deployment result with status, endpoint URL, and SLO comparison.
        """
        # Parse plan JSON
        try:
            plan_data = json.loads(plan_json)
        except (json.JSONDecodeError, TypeError):
            return {"error": "plan_json must be valid JSON"}

        if not isinstance(plan_data, dict):
            return {"error": "plan_json must be a JSON object"}

        from rhoai_mcp.composites.planner.models import DeploymentPlan

        try:
            plan = DeploymentPlan(**plan_data)
        except Exception as e:
            return {"error": f"Invalid deployment plan: {e}"}

        if not plan.ready:
            return {
                "error": "Deployment plan has unresolved blocking issues",
                "issues": [i.model_dump() for i in plan.issues if i.blocking],
            }

        from rhoai_mcp.composites.planner.execution import execute_deployment as _execute

        result = await _execute(server, plan)
        return result.model_dump()
