"""MCP tool for Neural Navigator model recommendations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from rhoai_mcp.composites.neuralnav.client import (
    NeuralNavAPIError,
    NeuralNavClient,
    NeuralNavConnectionError,
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
    "balanced": {"accuracy": 4, "price": 4, "latency": 1, "complexity": 1},
    "optimize_latency": {"accuracy": 2, "price": 2, "latency": 8, "complexity": 1},
    "optimize_cost": {"accuracy": 2, "price": 8, "latency": 1, "complexity": 1},
    "optimize_accuracy": {"accuracy": 8, "price": 2, "latency": 1, "complexity": 1},
}


def register_tools(mcp: FastMCP, server: RHOAIServer) -> None:
    """Register NeuralNav composite tools with the MCP server."""

    @mcp.tool()
    def recommend_model(
        text: str,
        use_case: str | None = None,
        user_count: int | None = None,
        preferred_gpu_types: list[str] | None = None,
        ttft_max_ms: int | None = None,
        itl_max_ms: int | None = None,
        e2e_max_ms: int | None = None,
        min_accuracy: int | None = None,
        max_cost_per_month: float | None = None,
        optimization_profile: str | None = None,
        percentile: str | None = None,
    ) -> dict[str, Any]:
        """Get LLM model recommendations from Neural Navigator.

        Runs the full NeuralNav recommendation flow: extracts intent from
        natural language, builds technical specifications, and returns
        the top-5 balanced model recommendations ranked by a weighted
        composite of accuracy, cost, latency, and deployment complexity.

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
            min_accuracy: Minimum model accuracy score (0-100).
                Filters out models below this quality threshold.
            max_cost_per_month: Maximum monthly cost in USD.
                Filters out configurations exceeding this budget.
            optimization_profile: Scoring profile that controls how
                recommendations are ranked. Valid values:
                balanced (default), optimize_latency, optimize_cost,
                optimize_accuracy.
            percentile: Percentile for SLO evaluation. Valid values:
                mean, p90, p95 (default), p99.

        Returns:
            Top-5 balanced model recommendations with assembled specification,
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

        # Validate min_accuracy
        if min_accuracy is not None and not 0 <= min_accuracy <= 100:
            return {"error": "min_accuracy must be between 0 and 100"}

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
        weights: dict[str, int] | None = None
        if optimization_profile is not None:
            if optimization_profile not in OPTIMIZATION_PROFILES:
                valid = ", ".join(sorted(OPTIMIZATION_PROFILES))
                return {
                    "error": f"Invalid optimization_profile '{optimization_profile}'. "
                    f"Valid values: {valid}",
                }
            weights = OPTIMIZATION_PROFILES[optimization_profile]

        client = NeuralNavClient(
            server.config.neuralnav_url,
            timeout=float(server.config.neuralnav_timeout),
        )

        try:
            result = client.recommend(
                text,
                use_case_override=use_case,
                user_count_override=user_count,
                gpu_types_override=preferred_gpu_types,
                ttft_override_ms=ttft_max_ms,
                itl_override_ms=itl_max_ms,
                e2e_override_ms=e2e_max_ms,
                min_accuracy=min_accuracy,
                max_cost=max_cost_per_month,
                weights=weights,
                percentile=percentile,
            )
        except NeuralNavConnectionError as e:
            logger.warning("NeuralNav connection error")
            logger.debug("NeuralNav connection error detail: %s", e)
            return {
                "error": "Neural Navigator unavailable",
                "hint": "Neural Navigator may be warming up. Retry shortly.",
            }
        except NeuralNavAPIError as e:
            logger.warning("NeuralNav API error status=%s", e.status_code)
            logger.debug("NeuralNav API error detail (truncated): %s", str(e.detail)[:512])
            return {
                "error": "Neural Navigator API error",
                "status_code": e.status_code,
            }

        # Format recommendations compactly to fit small LLM context windows
        recommendations = []
        for i, rec in enumerate(result.recommendations[:5], 1):
            compact: dict[str, Any] = {"rank": i}
            if rec.model_name:
                compact["model"] = rec.model_name
            elif rec.model_id:
                compact["model"] = rec.model_id
            if rec.gpu_config:
                gpu = rec.gpu_config
                compact["gpu"] = f"{gpu.gpu_count}x {gpu.gpu_type}"
            if rec.cost_per_month_usd is not None:
                compact["cost_usd_month"] = rec.cost_per_month_usd
            if rec.meets_slo:
                compact["meets_slo"] = True
            if rec.scores:
                compact["score"] = rec.scores.balanced_score
            if rec.reasoning:
                compact["reasoning"] = rec.reasoning
            recommendations.append(compact)

        response: dict[str, Any] = {
            "specification": result.specification,
            "recommendations": recommendations,
        }

        if not recommendations:
            response["message"] = "No configurations matched the requirements"

        return response
