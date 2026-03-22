"""HTTP client for Neural Navigator API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rhoai_mcp.composites.neuralnav.models import (
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
        gpu_config=raw.get("gpu_config"),
        predicted_ttft_p95_ms=raw.get("predicted_ttft_p95_ms"),
        predicted_itl_p95_ms=raw.get("predicted_itl_p95_ms"),
        predicted_e2e_p95_ms=raw.get("predicted_e2e_p95_ms"),
        predicted_throughput_qps=raw.get("predicted_throughput_qps"),
        cost_per_hour_usd=raw.get("cost_per_hour_usd"),
        cost_per_month_usd=raw.get("cost_per_month_usd"),
        meets_slo=raw.get("meets_slo", False),
        reasoning=raw["reasoning"],
        scores=raw.get("scores"),
    )


class NeuralNavConnectionError(Exception):
    """Raised when NeuralNav service is unreachable."""


class NeuralNavAPIError(Exception):
    """Raised when NeuralNav API returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"NeuralNav API error ({status_code}): {detail}")


class NeuralNavClient:
    """HTTP client for Neural Navigator API.

    Provides methods for each NeuralNav endpoint and a high-level
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
        """Make an HTTP request to NeuralNav."""
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
                    raise NeuralNavAPIError(
                        status_code=502,
                        detail="NeuralNav returned invalid JSON",
                    ) from e
        except httpx.TimeoutException as e:
            raise NeuralNavConnectionError(
                f"Neural Navigator request timed out at {self._base_url}{path}"
            ) from e
        except httpx.ConnectError as e:
            raise NeuralNavConnectionError(
                f"Neural Navigator service unavailable at {self._base_url}"
            ) from e
        except httpx.RequestError as e:
            raise NeuralNavConnectionError(
                f"Neural Navigator request failed: {type(e).__name__}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise NeuralNavAPIError(
                status_code=e.response.status_code,
                detail=e.response.text,
            ) from e

    def extract_intent(self, text: str) -> DeploymentIntent:
        """Extract deployment intent from natural language."""
        data = self._request("POST", "/api/v1/extract", json={"text": text})
        try:
            return DeploymentIntent(**data)
        except Exception as e:
            raise NeuralNavAPIError(
                status_code=502,
                detail=f"NeuralNav returned invalid intent response: {type(e).__name__}",
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

    def get_recommendations(
        self,
        use_case: str,
        user_count: int,
        prompt_tokens: int,
        output_tokens: int,
        expected_qps: float,
        ttft_target_ms: int,
        itl_target_ms: int,
        e2e_target_ms: int,
        preferred_gpu_types: list[str] | None = None,
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        weights: dict[str, int] | None = None,
        percentile: str | None = None,
    ) -> dict[str, Any]:
        """Get ranked recommendations from a specification."""
        payload: dict[str, Any] = {
            "use_case": use_case,
            "user_count": user_count,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "expected_qps": expected_qps,
            "ttft_target_ms": ttft_target_ms,
            "itl_target_ms": itl_target_ms,
            "e2e_target_ms": e2e_target_ms,
            "percentile": percentile or "p95",
            "include_near_miss": True,
        }
        if preferred_gpu_types:
            payload["preferred_gpu_types"] = preferred_gpu_types
        if min_accuracy is not None:
            payload["min_accuracy"] = min_accuracy
        if max_cost is not None:
            payload["max_cost"] = max_cost
        if weights is not None:
            payload["weights"] = weights
        return self._request("POST", "/api/v1/ranked-recommend-from-spec", json=payload)

    def recommend(
        self,
        text: str,
        use_case_override: str | None = None,
        user_count_override: int | None = None,
        gpu_types_override: list[str] | None = None,
        ttft_override_ms: int | None = None,
        itl_override_ms: int | None = None,
        e2e_override_ms: int | None = None,
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        weights: dict[str, int] | None = None,
        percentile: str | None = None,
    ) -> RecommendationResult:
        """Run the full recommendation flow.

        1. Extract intent from text
        2. Apply overrides
        3. Fetch SLO defaults + workload profile + expected RPS
        4. Apply SLO overrides on top of fetched defaults
        5. Get ranked recommendations with all constraints
        6. Extract top recommendation from each ranking list
        """
        # Step 1: Extract intent
        intent = self.extract_intent(text)

        # Step 2: Apply overrides
        use_case = use_case_override if use_case_override is not None else intent.use_case
        user_count = user_count_override if user_count_override is not None else intent.user_count
        gpu_types = (
            gpu_types_override if gpu_types_override is not None else intent.preferred_gpu_types
        )

        # Step 3: Fetch defaults
        slo_data = self.get_slo_defaults(use_case)
        workload_data = self.get_workload_profile(use_case)
        rps_data = self.get_expected_rps(use_case, user_count)

        # Extract values
        try:
            slo_defaults = slo_data["slo_defaults"]
            workload_profile = workload_data["workload_profile"]
            expected_qps = rps_data["expected_rps"]
            prompt_tokens = workload_profile["prompt_tokens"]
            output_tokens = workload_profile["output_tokens"]
        except KeyError as e:
            raise NeuralNavAPIError(
                status_code=502,
                detail=f"NeuralNav response missing expected field: {e}",
            ) from e

        # Step 4: Apply SLO overrides on top of fetched defaults
        try:
            ttft_default = slo_defaults["ttft_ms"]["default"]
            itl_default = slo_defaults["itl_ms"]["default"]
            e2e_default = slo_defaults["e2e_ms"]["default"]
        except KeyError as e:
            raise NeuralNavAPIError(
                status_code=502,
                detail=f"NeuralNav response missing expected SLO default field: {e}",
            ) from e

        ttft_target = ttft_override_ms if ttft_override_ms is not None else ttft_default
        itl_target = itl_override_ms if itl_override_ms is not None else itl_default
        e2e_target = e2e_override_ms if e2e_override_ms is not None else e2e_default

        # Step 5: Get recommendations with all constraints
        ranked = self.get_recommendations(
            use_case=use_case,
            user_count=user_count,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            expected_qps=expected_qps,
            ttft_target_ms=ttft_target,
            itl_target_ms=itl_target,
            e2e_target_ms=e2e_target,
            preferred_gpu_types=gpu_types if gpu_types else None,
            min_accuracy=min_accuracy,
            max_cost=max_cost,
            weights=weights,
            percentile=percentile,
        )

        # Step 6: Extract top recommendation from each ranking list
        try:
            balanced_list = ranked.get("balanced", [])
            cost_list = ranked.get("lowest_cost", [])
            latency_list = ranked.get("lowest_latency", [])
            total_evaluated = ranked["total_configs_evaluated"]
            after_filters = ranked["configs_after_filters"]
        except KeyError as e:
            raise NeuralNavAPIError(
                status_code=502,
                detail=f"NeuralNav ranking response missing expected field: {e}",
            ) from e

        try:
            top_balanced = _parse_recommendation(balanced_list[0]) if balanced_list else None
            top_cost = _parse_recommendation(cost_list[0]) if cost_list else None
            top_performance = _parse_recommendation(latency_list[0]) if latency_list else None
        except Exception as e:
            raise NeuralNavAPIError(
                status_code=502,
                detail=f"NeuralNav returned invalid recommendation data: {type(e).__name__}",
            ) from e

        return RecommendationResult(
            specification={
                "use_case": use_case,
                "user_count": user_count,
                "slo_targets": {
                    "ttft_ms": ttft_target,
                    "itl_ms": itl_target,
                    "e2e_ms": e2e_target,
                },
                "traffic_profile": {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "expected_qps": expected_qps,
                },
            },
            top_performance=top_performance,
            top_cost=top_cost,
            top_balanced=top_balanced,
            total_configs_evaluated=total_evaluated,
            configs_after_filters=after_filters,
        )

    def health_check(self) -> tuple[bool, str]:
        """Check if NeuralNav service is available."""
        try:
            self._request("GET", "/health")
            return True, "Neural Navigator available"
        except (NeuralNavConnectionError, NeuralNavAPIError) as e:
            logger.debug("NeuralNav health check failed (%s)", type(e).__name__)
            return False, "Neural Navigator unavailable"
