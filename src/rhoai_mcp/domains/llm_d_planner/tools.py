"""MCP tools for llm-d-planner model recommendation workflow.

Provides 4 workflow-token-chained tools:
  extract_intent → prepare_model_tech_specs → get_recommended_models → get_deployment_config
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rhoai_mcp.domains.llm_d_planner.client import (
    PlannerAPIError,
    PlannerClient,
    PlannerConnectionError,
)
from rhoai_mcp.utils.workflow_token import workflow_step

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)

_MAX_TEXT_LENGTH = 4000


def _error_result(
    error: str,
    *,
    hint: str | None = None,
    status_code: int | None = None,
) -> dict[str, Any]:
    """Build a standardized error dict."""
    result: dict[str, Any] = {"error": error}
    if hint:
        result["hint"] = hint
    if status_code is not None:
        result["status_code"] = status_code
    return result


def _handle_planner_error(
    e: PlannerConnectionError | PlannerAPIError,
    server: RHOAIServer,
) -> dict[str, Any]:
    """Convert a planner client error to a tool error dict."""
    if isinstance(e, PlannerConnectionError):
        return _error_result(
            f"llm-d-planner service unavailable: {e}",
            hint=(
                "llm-d-planner may be warming up (first request loads the LLM model). "
                "Check RHOAI_MCP_PLANNER_URL "
                f"(currently: {server.config.planner_url}) and RHOAI_MCP_PLANNER_TIMEOUT "
                f"(currently: {server.config.planner_timeout}s)."
            ),
        )
    return _error_result(
        f"llm-d-planner API error: {e.detail}",
        status_code=e.status_code,
    )


VALID_PERCENTILES = {"p50", "p75", "p90", "p95", "p99"}

VALID_CATEGORIES = {"balanced", "cost", "performance"}

OPTIMIZATION_PROFILES: dict[str, dict[str, int]] = {
    "optimize_cost": {"accuracy": 2, "price": 8, "latency": 1, "complexity": 1},
    "optimize_latency": {"accuracy": 2, "price": 2, "latency": 8, "complexity": 1},
    "optimize_accuracy": {"accuracy": 8, "price": 2, "latency": 1, "complexity": 1},
    "balanced": {"accuracy": 4, "price": 4, "latency": 4, "complexity": 1},
}


def _format_recommendation(rec: Any) -> dict[str, Any]:
    """Format a single recommendation for tool output."""
    from rhoai_mcp.domains.llm_d_planner.client import _parse_recommendation

    parsed = _parse_recommendation(rec) if isinstance(rec, dict) else rec
    result: dict[str, Any] = {
        "model": parsed.model_name or parsed.model_id or "unknown",
    }
    if parsed.gpu_config:
        gpu = parsed.gpu_config
        if isinstance(gpu, dict):
            result["gpu"] = f"{gpu.get('gpu_count', '?')}x {gpu.get('gpu_type', '?')}"
        else:
            result["gpu"] = f"{gpu.gpu_count}x {gpu.gpu_type}"
    if parsed.predicted_ttft_p95_ms is not None:
        result["ttft_p95_ms"] = parsed.predicted_ttft_p95_ms
    if parsed.predicted_itl_p95_ms is not None:
        result["itl_p95_ms"] = parsed.predicted_itl_p95_ms
    if parsed.predicted_e2e_p95_ms is not None:
        result["e2e_p95_ms"] = parsed.predicted_e2e_p95_ms
    if parsed.cost_per_month_usd is not None:
        result["cost_per_month_usd"] = parsed.cost_per_month_usd
    if parsed.meets_slo is not None:
        result["meets_slo"] = parsed.meets_slo
    if parsed.scores and parsed.scores.balanced_score is not None:
        result["score"] = parsed.scores.balanced_score
    if parsed.reasoning:
        result["reasoning"] = parsed.reasoning
    return result


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register llm-d-planner tools with the MCP server."""

    @mcp.tool()
    @workflow_step(produces="intent_extracted")
    def extract_intent(text: str) -> dict[str, Any]:
        """Extract deployment intent from a natural language description.

        Analyzes the user's text to identify use case, scale, priorities,
        and other deployment requirements. Returns structured intent data
        and a workflow_token for the next step (prepare_model_tech_specs).
        """
        if not text or not text.strip():
            return _error_result("'text' must be non-empty")
        if len(text) > _MAX_TEXT_LENGTH:
            return _error_result(f"'text' exceeds max length ({_MAX_TEXT_LENGTH} characters)")

        try:
            client = PlannerClient(
                server.config.planner_url,
                timeout=float(server.config.planner_timeout),
            )
            intent = client.extract_intent(text)
            return intent.model_dump()
        except (PlannerConnectionError, PlannerAPIError) as e:
            return _handle_planner_error(e, server)

    @mcp.tool()
    @workflow_step(requires="intent_extracted", produces="specs_prepared")
    def prepare_model_tech_specs(
        workflow_token: str,  # noqa: ARG001
        use_case: str | None = None,
        user_count: int | None = None,
        preferred_gpu_types: list[str] | None = None,
        ttft_max_ms: int | None = None,
        itl_max_ms: int | None = None,
        e2e_max_ms: int | None = None,
    ) -> dict[str, Any]:
        """Build full technical specification from intent and API defaults.

        Uses the intent extracted in the previous step as defaults. Optional
        parameters override specific values. Fetches SLO defaults, workload
        profile, and expected RPS from the llm-d-planner API.

        Returns the specification and a workflow_token for get_recommended_models.
        """
        prev: dict[str, Any] = workflow_token  # type: ignore[assignment]  # replaced by decorator

        # Apply overrides on intent data
        uc = use_case or prev["use_case"]
        uc_count = user_count if user_count is not None else prev["user_count"]
        gpu_types = (
            preferred_gpu_types
            if preferred_gpu_types is not None
            else prev.get("preferred_gpu_types", [])
        )

        try:
            client = PlannerClient(
                server.config.planner_url,
                timeout=float(server.config.planner_timeout),
            )

            slo_data = client.get_slo_defaults(uc)
            workload_data = client.get_workload_profile(uc)
            rps_data = client.get_expected_rps(uc, uc_count)

            slo_defaults = slo_data["slo_defaults"]
            workload_profile = workload_data["workload_profile"]
            expected_qps = rps_data["expected_rps"]

            # Apply SLO overrides on top of fetched defaults
            ttft = ttft_max_ms if ttft_max_ms is not None else slo_defaults["ttft_ms"]["default"]
            itl = itl_max_ms if itl_max_ms is not None else slo_defaults["itl_ms"]["default"]
            e2e = e2e_max_ms if e2e_max_ms is not None else slo_defaults["e2e_ms"]["default"]

            specification = {
                "use_case": uc,
                "user_count": uc_count,
                "slo_targets": {
                    "ttft_ms": ttft,
                    "itl_ms": itl,
                    "e2e_ms": e2e,
                },
                "traffic_profile": {
                    "prompt_tokens": workload_profile["prompt_tokens"],
                    "output_tokens": workload_profile["output_tokens"],
                    "expected_qps": expected_qps,
                },
                "preferred_gpu_types": gpu_types,
            }

            return {"specification": specification}

        except (PlannerConnectionError, PlannerAPIError) as e:
            return _handle_planner_error(e, server)
        except KeyError as ke:
            return _error_result(f"Planner response missing expected field: {ke}")

    @mcp.tool()
    @workflow_step(requires="specs_prepared", produces="models_recommended")
    def get_recommended_models(
        workflow_token: str,  # noqa: ARG001
        optimization_profile: str | None = None,
        min_accuracy: int | None = None,
        max_cost_per_month: float | None = None,
        percentile: str | None = None,
    ) -> dict[str, Any]:
        """Get ranked model recommendations from the technical specification.

        Uses the specification built in the previous step. Optional parameters
        adjust ranking criteria. Returns top recommendations in 3 categories
        (performance, cost, balanced) and a workflow_token for get_deployment_config.
        """
        prev: dict[str, Any] = workflow_token  # type: ignore[assignment]
        spec = prev["specification"]

        # Validate optimization profile
        weights = None
        if optimization_profile:
            if optimization_profile not in OPTIMIZATION_PROFILES:
                return _error_result(
                    f"Invalid optimization_profile '{optimization_profile}'. "
                    f"Valid: {', '.join(sorted(OPTIMIZATION_PROFILES))}"
                )
            weights = OPTIMIZATION_PROFILES[optimization_profile]

        # Validate constraints
        if min_accuracy is not None and not (0 <= min_accuracy <= 100):
            return _error_result("min_accuracy must be between 0 and 100")
        if max_cost_per_month is not None and max_cost_per_month < 0:
            return _error_result("max_cost_per_month must be non-negative")
        if percentile and percentile not in VALID_PERCENTILES:
            return _error_result(
                f"Invalid percentile '{percentile}'. Valid: {', '.join(sorted(VALID_PERCENTILES))}"
            )

        try:
            client = PlannerClient(
                server.config.planner_url,
                timeout=float(server.config.planner_timeout),
            )

            slo = spec["slo_targets"]
            traffic = spec["traffic_profile"]

            ranked = client.get_recommendations(
                use_case=spec["use_case"],
                user_count=spec["user_count"],
                prompt_tokens=traffic["prompt_tokens"],
                output_tokens=traffic["output_tokens"],
                expected_qps=traffic["expected_qps"],
                ttft_target_ms=slo["ttft_ms"],
                itl_target_ms=slo["itl_ms"],
                e2e_target_ms=slo["e2e_ms"],
                preferred_gpu_types=spec.get("preferred_gpu_types") or None,
                min_accuracy=min_accuracy,
                max_cost=max_cost_per_month,
                weights=weights,
                percentile=percentile,
            )

            # Extract top recommendation from each category
            balanced_list = ranked.get("balanced", [])
            cost_list = ranked.get("lowest_cost", [])
            latency_list = ranked.get("lowest_latency", [])

            recommendations: dict[str, Any] = {}
            if balanced_list:
                recommendations["top_balanced"] = _format_recommendation(balanced_list[0])
            if cost_list:
                recommendations["top_cost"] = _format_recommendation(cost_list[0])
            if latency_list:
                recommendations["top_performance"] = _format_recommendation(latency_list[0])

            result: dict[str, Any] = {
                "specification": spec,
                "recommendations": recommendations,
            }

            if not recommendations:
                result["message"] = (
                    f"No models matched your constraints "
                    f"({ranked.get('total_configs_evaluated', 0)} evaluated, "
                    f"{ranked.get('configs_after_filters', 0)} after filters). "
                    f"Try relaxing SLO targets or cost limits."
                )

            # Store the raw ranked response for get_deployment_config
            result["_ranked_response"] = ranked

            return result

        except (PlannerConnectionError, PlannerAPIError) as e:
            return _handle_planner_error(e, server)

    @mcp.tool()
    @workflow_step(requires="models_recommended")
    def get_deployment_config(
        workflow_token: str,  # noqa: ARG001
        category: str,
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Generate Kubernetes deployment YAML for a recommended model.

        Uses the recommendations from the previous step. Picks the top model
        from the specified category and generates deployment configurations.

        This is the terminal step — no workflow_token in the output.
        """
        prev: dict[str, Any] = workflow_token  # type: ignore[assignment]

        # Validate category
        if category not in VALID_CATEGORIES:
            return _error_result(
                f"Invalid category '{category}'. Valid: {', '.join(sorted(VALID_CATEGORIES))}"
            )

        # Validate namespace
        if not namespace or not namespace.strip():
            return _error_result("namespace must be non-empty")

        from rhoai_mcp.domains.llm_d_planner.client import CATEGORY_MAP

        # Get the raw ranked response from the token data
        ranked = prev.get("_ranked_response", {})
        category_key = CATEGORY_MAP.get(category)
        if not category_key:
            return _error_result(f"Invalid category '{category}'")

        category_list = ranked.get(category_key, [])
        if not category_list:
            return _error_result(
                f"No recommendation found for category '{category}'. "
                f"Try a different category or relax constraints."
            )

        recommendation = category_list[0]
        model_name = recommendation.get("model_name") or recommendation.get("model_id")

        try:
            client = PlannerClient(
                server.config.planner_url,
                timeout=float(server.config.planner_timeout),
            )

            deploy_result = client.deploy(recommendation, namespace=namespace)

            if deploy_result.get("success") is not True:
                return _error_result(
                    deploy_result.get("message", "Deployment config generation failed"),
                )

            files = deploy_result.get("files", {})
            if not files:
                return _error_result("No config files were generated")

            result: dict[str, Any] = {
                "deployment_id": deploy_result["deployment_id"],
                "namespace": deploy_result["namespace"],
                "configs": files,
            }
            if model_name:
                result["model"] = model_name

            return result

        except (PlannerConnectionError, PlannerAPIError) as e:
            return _handle_planner_error(e, server)
