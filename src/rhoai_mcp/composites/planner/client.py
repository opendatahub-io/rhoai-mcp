"""HTTP client for Planner API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rhoai_mcp.composites.planner.models import (
    DeploymentConfigResult,
    DeploymentIntent,
    ModelRecommendation,
    RecommendationResult,
)

logger = logging.getLogger(__name__)


def _parse_recommendation(raw: dict[str, Any]) -> ModelRecommendation:
    """Parse a single recommendation dict into a ModelRecommendation."""
    return ModelRecommendation(
        model_id=raw.get("model_id"),
        model_name=raw.get("model_name"),
        model_uri=raw.get("model_uri"),
        gpu_config=raw.get("gpu_config"),
        predicted_ttft_p95_ms=raw.get("predicted_ttft_p95_ms"),
        predicted_itl_p95_ms=raw.get("predicted_itl_p95_ms"),
        predicted_e2e_p95_ms=raw.get("predicted_e2e_p95_ms"),
        predicted_throughput_qps=raw.get("predicted_throughput_qps"),
        benchmark_metrics=raw.get("benchmark_metrics"),
        cost_per_hour_usd=raw.get("cost_per_hour_usd"),
        cost_per_month_usd=raw.get("cost_per_month_usd"),
        meets_slo=raw.get("meets_slo", False),
        reasoning=raw["reasoning"],
        alternative_options=raw.get("alternative_options"),
        scores=raw.get("scores"),
        configuration=raw.get("configuration"),
    )


CATEGORY_MAP: dict[str, str] = {
    "balanced": "balanced",
    "cost": "lowest_cost",
    "performance": "lowest_latency",
    "quality": "best_quality",
}

_PROFILE_TO_SPEC = {"quality": "quality", "price": "cost", "latency": "latency"}


def _apply_priority_overrides(
    spec_data: dict[str, Any],
    priority_weights: dict[str, int],
) -> None:
    """Apply optimization-profile weights to the specification's priorities."""
    try:
        priorities = spec_data["priorities"]
    except KeyError as e:
        raise PlannerAPIError(
            status_code=502,
            detail=f"Planner specification response missing expected field: {e}",
        ) from e
    if not isinstance(priorities, dict):
        raise PlannerAPIError(
            status_code=502,
            detail="Planner specification 'priorities' is not a valid object",
        )
    priorities = dict(priorities)
    spec_data["priorities"] = priorities
    for profile_key, spec_key in _PROFILE_TO_SPEC.items():
        if profile_key not in priority_weights:
            continue
        if spec_key not in priorities:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner specification missing 'priorities.{spec_key}'",
            )
        entry = priorities[spec_key]
        if not isinstance(entry, dict):
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner specification 'priorities.{spec_key}' is not a valid object",
            )
        priorities[spec_key] = {**entry, "weight": priority_weights[profile_key]}


class PlannerConnectionError(Exception):
    """Raised when Planner service is unreachable."""


class PlannerAPIError(Exception):
    """Raised when Planner API returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Planner API error ({status_code}): {detail}")


class PlannerClient:
    """HTTP client for Planner API.

    Provides methods for each Planner endpoint and a high-level
    `recommend()` method that chains the full flow.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to Planner."""
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                kwargs: dict[str, Any] = {"params": params}
                if method.upper() in ("POST", "PUT", "PATCH"):
                    kwargs["json"] = json
                http_method = getattr(client, method.lower())
                response = http_method(url, **kwargs)
                response.raise_for_status()
                try:
                    return response.json()  # type: ignore[no-any-return]
                except ValueError as e:
                    raise PlannerAPIError(
                        status_code=502,
                        detail="Planner returned invalid JSON",
                    ) from e
        except httpx.TimeoutException as e:
            raise PlannerConnectionError(
                f"Planner request timed out at {self._base_url}{path}"
            ) from e
        except httpx.ConnectError as e:
            raise PlannerConnectionError(f"Planner service unavailable at {self._base_url}") from e
        except httpx.RequestError as e:
            raise PlannerConnectionError(f"Planner request failed: {type(e).__name__}") from e
        except httpx.HTTPStatusError as e:
            raise PlannerAPIError(
                status_code=e.response.status_code,
                detail=e.response.text,
            ) from e

    def extract_intent(self, text: str) -> DeploymentIntent:
        """Extract deployment intent from natural language."""
        data = self._request("POST", "/api/v1/extract", json={"text": text})
        try:
            return DeploymentIntent(**data)
        except Exception as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner returned invalid intent response: {type(e).__name__}",
            ) from e

    def get_slo_defaults(self, use_case: str) -> dict[str, Any]:
        """Get SLO default values for a use case."""
        return self._request("GET", f"/api/v1/slo-defaults/{use_case}")

    def get_workload_profile(self, use_case: str) -> dict[str, Any]:
        """Get workload profile for a use case."""
        return self._request("GET", f"/api/v1/workload-profile/{use_case}")

    def get_expected_rps(self, use_case: str, user_count: int) -> dict[str, Any]:
        """Calculate expected RPS for a use case and user count."""
        return self._request(
            "GET",
            f"/api/v1/expected-rps/{use_case}",
            params={"user_count": user_count},
        )

    def generate_specification(self, intent: DeploymentIntent) -> dict[str, Any]:
        """Generate a deployment specification from a deployment intent."""
        return self._request(
            "POST",
            "/api/v1/generate-specification",
            json=intent.model_dump(),
        )

    def generate_recommendations(
        self,
        specification: dict[str, Any],
        min_quality: float | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
    ) -> dict[str, Any]:
        """Get ranked recommendations from a deployment specification."""
        payload: dict[str, Any] = {
            "specification": specification,
            "include_near_miss": include_near_miss,
        }
        if min_quality is not None:
            payload["min_quality"] = min_quality
        if max_cost is not None:
            payload["max_cost"] = max_cost
        return self._request("POST", "/api/v1/generate-recommendations", json=payload)

    def generate_deployment(
        self,
        configuration: dict[str, Any],
        namespace: str = "default",
        stack: str = "vllm",
    ) -> dict[str, Any]:
        """Generate deployment bundle from a deployment configuration."""
        return self._request(
            "POST",
            "/api/v1/generate-deployment",
            json={
                "configuration": configuration,
                "namespace": namespace,
                "stack": stack,
            },
        )

    def recommend(
        self,
        text: str,
        use_case_override: str | None = None,
        user_count_override: int | None = None,
        gpu_types_override: list[str] | None = None,
        ttft_override_ms: int | None = None,
        itl_override_ms: int | None = None,
        e2e_override_ms: int | None = None,
        min_quality: float | None = None,
        max_cost: float | None = None,
        percentile_override: str | None = None,
        priority_weights: dict[str, int] | None = None,
    ) -> RecommendationResult:
        """Run the full recommendation flow.

        1. Extract intent from text (skipped when overrides cover all needed fields)
        2. Apply overrides to build a DeploymentIntent
        3. Generate specification via the API
        4. Apply SLO overrides on top of generated specification
        5. Get ranked recommendations
        6. Extract top recommendation from each ranking list
        """
        # Step 1: Extract intent (skip when all overrides are provided)
        if (
            use_case_override is not None
            and user_count_override is not None
            and gpu_types_override is not None
        ):
            intent_for_spec = DeploymentIntent(
                use_case=use_case_override,
                user_count=user_count_override,
                preferred_gpu_types=gpu_types_override,
            )
        else:
            intent = self.extract_intent(text)
            intent_for_spec = DeploymentIntent(
                use_case=(use_case_override if use_case_override is not None else intent.use_case),
                user_count=(
                    user_count_override if user_count_override is not None else intent.user_count
                ),
                preferred_gpu_types=(
                    gpu_types_override
                    if gpu_types_override is not None
                    else [
                        g if isinstance(g, str) else g.gpu_type for g in intent.preferred_gpu_types
                    ]
                ),
                preferred_models=intent.preferred_models,
                domain_specialization=intent.domain_specialization,
                quality_priority=intent.quality_priority,
                cost_priority=intent.cost_priority,
                latency_priority=intent.latency_priority,
            )

        # Step 3: Generate specification
        spec_data = self.generate_specification(intent_for_spec)

        # Step 4: Apply SLO overrides on top of generated specification.
        # Shallow-copy mutable sub-dicts so generate_specification's result is not mutated.
        try:
            slo_targets = dict(spec_data["slo_targets"])
            spec_data["slo_targets"] = slo_targets
        except KeyError as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner specification response missing expected field: {e}",
            ) from e

        if ttft_override_ms is not None:
            slo_targets["ttft_target_ms"] = ttft_override_ms
        if itl_override_ms is not None:
            slo_targets["itl_target_ms"] = itl_override_ms
        if e2e_override_ms is not None:
            slo_targets["e2e_target_ms"] = e2e_override_ms
        if percentile_override is not None:
            slo_targets["percentile"] = percentile_override

        if priority_weights is not None:
            _apply_priority_overrides(spec_data, priority_weights)

        # Step 5: Get recommendations
        ranked = self.generate_recommendations(
            specification=spec_data,
            min_quality=min_quality,
            max_cost=max_cost,
        )

        # Step 6: Extract top recommendation from each ranking list
        try:
            balanced_list = ranked.get("balanced", [])
            cost_list = ranked.get("lowest_cost", [])
            latency_list = ranked.get("lowest_latency", [])
            quality_list = ranked.get("best_quality", [])
            total_evaluated = ranked["total_configs_evaluated"]
            after_filters = ranked["configs_after_filters"]
        except KeyError as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner ranking response missing expected field: {e}",
            ) from e

        try:
            top_balanced = _parse_recommendation(balanced_list[0]) if balanced_list else None
            top_cost = _parse_recommendation(cost_list[0]) if cost_list else None
            top_performance = _parse_recommendation(latency_list[0]) if latency_list else None
            top_quality = _parse_recommendation(quality_list[0]) if quality_list else None
        except Exception as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner returned invalid recommendation data: {type(e).__name__}",
            ) from e

        # Build specification summary for the result
        try:
            workload = spec_data["workload_profile"]
            spec_summary = {
                "use_case": intent_for_spec.use_case,
                "user_count": intent_for_spec.user_count,
                "slo_targets": {
                    "ttft_target_ms": slo_targets["ttft_target_ms"],
                    "itl_target_ms": slo_targets["itl_target_ms"],
                    "e2e_target_ms": slo_targets["e2e_target_ms"],
                },
                "traffic_profile": {
                    "prompt_tokens": workload["prompt_tokens"],
                    "output_tokens": workload["output_tokens"],
                    "expected_qps": workload["expected_qps"],
                },
            }
        except KeyError as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner specification response missing expected field: {e}",
            ) from e

        return RecommendationResult(
            specification=spec_summary,
            top_performance=top_performance,
            top_cost=top_cost,
            top_balanced=top_balanced,
            top_quality=top_quality,
            total_configs_evaluated=total_evaluated,
            configs_after_filters=after_filters,
        )

    def generate_config(
        self,
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
        preferred_gpu_types: list[str] | None = None,
        min_quality: float | None = None,
        max_cost: float | None = None,
        percentile: str | None = None,
        priority_weights: dict[str, int] | None = None,
    ) -> DeploymentConfigResult:
        """Generate deployment configs for the top recommendation in a category.

        1. Build a DeploymentIntent from parameters
        2. Generate specification via the API
        3. Get ranked recommendations
        4. Pick the top recommendation from the specified category
        5. Extract configuration and generate deployment bundle
        """
        # Step 1: Validate category before making any API calls
        category_key = CATEGORY_MAP.get(category)
        if category_key is None:
            raise PlannerAPIError(
                status_code=400,
                detail=f"Invalid category '{category}'. Valid: {', '.join(CATEGORY_MAP)}",
            )

        # Step 2: Build intent and generate specification
        intent = DeploymentIntent(
            use_case=use_case,
            user_count=user_count,
            preferred_gpu_types=preferred_gpu_types or [],
        )
        spec_data = self.generate_specification(intent)

        # Apply explicit SLO targets.
        # Shallow-copy mutable sub-dicts so generate_specification's result is not mutated.
        try:
            slo_targets = dict(spec_data["slo_targets"])
            spec_data["slo_targets"] = slo_targets
        except KeyError as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner specification response missing expected field: {e}",
            ) from e
        slo_targets["ttft_target_ms"] = ttft_target_ms
        slo_targets["itl_target_ms"] = itl_target_ms
        slo_targets["e2e_target_ms"] = e2e_target_ms

        # Apply explicit workload profile overrides
        try:
            workload = spec_data["workload_profile"]
        except KeyError as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner specification response missing expected field: {e}",
            ) from e
        if not isinstance(workload, dict):
            raise PlannerAPIError(
                status_code=502,
                detail="Planner specification 'workload_profile' is not a valid object",
            )
        workload = dict(workload)
        spec_data["workload_profile"] = workload
        workload["prompt_tokens"] = prompt_tokens
        workload["output_tokens"] = output_tokens
        workload["expected_qps"] = expected_qps

        if percentile is not None:
            slo_targets["percentile"] = percentile

        if priority_weights is not None:
            _apply_priority_overrides(spec_data, priority_weights)

        # Step 3: Get ranked recommendations
        ranked = self.generate_recommendations(
            specification=spec_data,
            min_quality=min_quality,
            max_cost=max_cost,
        )

        # Step 4: Pick top recommendation from category
        category_list = ranked.get(category_key, [])
        if not category_list:
            raise PlannerAPIError(
                status_code=404,
                detail=f"No recommendation found for category '{category}'",
            )
        recommendation = category_list[0]

        # Extract model name (model_name preferred, model_id as fallback)
        model_name = recommendation.get("model_name") or recommendation.get("model_id")

        # Step 5: Extract configuration and generate deployment bundle
        configuration = recommendation.get("configuration")
        if configuration is None:
            configuration = {
                "model_id": recommendation.get("model_id", "unknown"),
                "model_name": recommendation.get("model_name"),
                "model_uri": recommendation.get("model_uri"),
                "gpu_config": recommendation.get("gpu_config"),
                "use_case": use_case,
                "expected_qps": expected_qps,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "e2e_target_ms": e2e_target_ms,
            }

        bundle = self.generate_deployment(configuration, namespace=namespace)

        files = bundle.get("files", {})
        if not files:
            raise PlannerAPIError(
                status_code=502,
                detail="Planner generated no config files",
            )

        try:
            return DeploymentConfigResult(
                deployment_id=bundle["deployment_id"],
                namespace=bundle["namespace"],
                model_name=model_name,
                configs=files,
            )
        except KeyError as e:
            raise PlannerAPIError(
                status_code=502,
                detail=f"Planner deployment response missing expected field: {e}",
            ) from e

    def health_check(self) -> tuple[bool, str]:
        """Check if Planner service is available."""
        try:
            self._request("GET", "/health")
            return True, "Planner available"
        except (PlannerConnectionError, PlannerAPIError) as e:
            logger.debug("Planner health check failed (%s)", type(e).__name__)
            return False, "Planner unavailable"
