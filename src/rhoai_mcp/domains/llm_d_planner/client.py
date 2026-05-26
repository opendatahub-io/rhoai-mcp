"""HTTP client for llm-d-planner API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rhoai_mcp.domains.llm_d_planner.models import (
    DeploymentIntent,
    ModelRecommendation,
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


CATEGORY_MAP: dict[str, str] = {
    "balanced": "balanced",
    "cost": "lowest_cost",
    "performance": "lowest_latency",
}


class PlannerConnectionError(Exception):
    """Raised when llm-d-planner service is unreachable."""


class PlannerAPIError(Exception):
    """Raised when llm-d-planner API returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Planner API error ({status_code}): {detail}")


class PlannerClient:
    """HTTP client for llm-d-planner API.

    Provides methods for each llm-d-planner endpoint.
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
        """Make an HTTP request to llm-d-planner."""
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
                f"llm-d-planner request timed out at {self._base_url}{path}"
            ) from e
        except httpx.ConnectError as e:
            raise PlannerConnectionError(
                f"llm-d-planner service unavailable at {self._base_url}"
            ) from e
        except httpx.RequestError as e:
            raise PlannerConnectionError(f"llm-d-planner request failed: {type(e).__name__}") from e
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

    def deploy(
        self,
        recommendation: dict[str, Any],
        namespace: str = "default",
    ) -> dict[str, Any]:
        """Call POST /api/v1/deploy to generate YAML configs.

        The recommendation dict must be a full DeploymentRecommendation as
        returned by the /api/v1/ranked-recommend-from-spec endpoint.

        Returns dict with deployment_id, namespace, files (config type to
        YAML content), success, and message.
        """
        return self._request(
            "POST",
            "/api/v1/deploy",
            json={"recommendation": recommendation, "namespace": namespace},
        )

    def health_check(self) -> tuple[bool, str]:
        """Check if llm-d-planner service is available."""
        try:
            self._request("GET", "/health")
            return True, "llm-d-planner available"
        except (PlannerConnectionError, PlannerAPIError) as e:
            logger.debug("llm-d-planner health check failed (%s)", type(e).__name__)
            return False, "llm-d-planner unavailable"
