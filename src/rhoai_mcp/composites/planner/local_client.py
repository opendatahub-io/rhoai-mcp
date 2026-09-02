"""Embedded local Planner client using llm-d-planner as a library."""

from __future__ import annotations

import logging
from typing import Any

from planner import (
    DeploymentIntent,
    DeploymentSpecification,
    Planner,
    PlannerError,
)

from rhoai_mcp.composites.planner.client import (
    CATEGORY_MAP,
    PlannerAPIError,
    _apply_priority_overrides,
    _parse_recommendation,
)
from rhoai_mcp.composites.planner.models import (
    DeploymentConfigResult,
    RecommendationResult,
)

logger = logging.getLogger(__name__)


class LocalPlannerClient:
    """Embedded Planner client using in-process library calls.

    When a Model Catalog URL is provided, benchmarks are synced from it.
    Otherwise, falls back to bundled synthetic benchmarks (with a warning).
    Intent extraction is NOT supported — all required fields must be
    provided as explicit overrides by the calling agent.
    """

    def __init__(
        self,
        model_catalog_url: str | None = None,
    ) -> None:
        self._planner = Planner()

        # Pre-seed the gpu_normalizer singleton with the Planner's catalog
        # to avoid a redundant ModelCatalog() construction on first
        # normalize_gpu_types() call (llm-d-planner bug workaround).
        import planner.shared.utils.gpu_normalizer as _gpu_norm

        _gpu_norm._catalog_instance = self._planner._model_catalog

        if model_catalog_url:
            try:
                result = self._planner.sync_model_catalog(url=model_catalog_url)
                logger.info(
                    "Model Catalog sync: %d benchmarks added, %d models added",
                    result.get("benchmarks_added", 0),
                    result.get("models_added", 0),
                )
            except ImportError as e:
                raise PlannerAPIError(
                    status_code=502,
                    detail=f"Model Catalog sync requires extra dependency: {e}",
                ) from e
            except (ValueError, PlannerError) as e:
                raise PlannerAPIError(
                    status_code=502,
                    detail=f"Model Catalog sync failed: {e}",
                ) from e
        else:
            logger.warning(
                "No Model Catalog URL configured; loading bundled synthetic benchmarks. "
                "Set RHOAI_MCP_PLANNER_MODEL_CATALOG_URL for accurate recommendations."
            )
            self._planner.load_bundled_benchmarks()

    def recommend(
        self,
        text: str,  # noqa: ARG002
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
        """Run the full recommendation flow using the embedded Planner.

        All overrides (use_case, user_count, gpu_types) must be provided.
        Intent extraction from natural language is not supported in local mode.
        """
        if use_case_override is None or user_count_override is None or gpu_types_override is None:
            raise PlannerAPIError(
                status_code=400,
                detail="Local mode requires explicit overrides: "
                "provide use_case, user_count, and preferred_gpu_types. "
                "Intent extraction from text is not supported in local mode.",
            )

        try:
            intent = DeploymentIntent(
                use_case=use_case_override,
                user_count=user_count_override,
                preferred_gpu_types=gpu_types_override,
            )

            spec = self._planner.generate_specification(intent)
            spec_data = spec.model_dump()

            # Apply SLO overrides
            slo_targets = spec_data["slo_targets"]
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

            spec_with_overrides = DeploymentSpecification(**spec_data)

            ranked = self._planner.generate_recommendations(
                spec_with_overrides,
                min_quality=min_quality,
                max_cost=max_cost,
            )
            ranked_data = ranked.model_dump()

            balanced_list = ranked_data.get("balanced", [])
            cost_list = ranked_data.get("lowest_cost", [])
            latency_list = ranked_data.get("lowest_latency", [])
            quality_list = ranked_data.get("best_quality", [])

            top_balanced = _parse_recommendation(balanced_list[0]) if balanced_list else None
            top_cost = _parse_recommendation(cost_list[0]) if cost_list else None
            top_performance = _parse_recommendation(latency_list[0]) if latency_list else None
            top_quality = _parse_recommendation(quality_list[0]) if quality_list else None

            workload = spec_data["workload_profile"]
            spec_summary: dict[str, Any] = {
                "use_case": use_case_override,
                "user_count": user_count_override,
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

            return RecommendationResult(
                specification=spec_summary,
                top_performance=top_performance,
                top_cost=top_cost,
                top_balanced=top_balanced,
                top_quality=top_quality,
                total_configs_evaluated=ranked.total_configs_evaluated,
                configs_after_filters=ranked.configs_after_filters,
            )

        except PlannerAPIError:
            raise
        except ValueError as e:
            raise PlannerAPIError(status_code=400, detail=str(e)) from e
        except PlannerError as e:
            raise PlannerAPIError(status_code=502, detail=str(e)) from e

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
        """Generate deployment configs for the top recommendation in a category."""
        category_key = CATEGORY_MAP.get(category)
        if category_key is None:
            raise PlannerAPIError(
                status_code=400,
                detail=f"Invalid category '{category}'. Valid: {', '.join(CATEGORY_MAP)}",
            )

        try:
            intent = DeploymentIntent(
                use_case=use_case,
                user_count=user_count,
                preferred_gpu_types=preferred_gpu_types or [],
            )

            spec = self._planner.generate_specification(intent)
            spec_data = spec.model_dump()

            # Apply explicit SLO targets
            slo_targets = spec_data["slo_targets"]
            slo_targets["ttft_target_ms"] = ttft_target_ms
            slo_targets["itl_target_ms"] = itl_target_ms
            slo_targets["e2e_target_ms"] = e2e_target_ms

            # Apply explicit workload profile overrides
            workload = spec_data["workload_profile"]
            workload["prompt_tokens"] = prompt_tokens
            workload["output_tokens"] = output_tokens
            workload["expected_qps"] = expected_qps

            if percentile is not None:
                slo_targets["percentile"] = percentile

            if priority_weights is not None:
                _apply_priority_overrides(spec_data, priority_weights)

            spec_with_overrides = DeploymentSpecification(**spec_data)

            ranked = self._planner.generate_recommendations(
                spec_with_overrides,
                min_quality=min_quality,
                max_cost=max_cost,
            )
            ranked_data = ranked.model_dump()

            category_list = ranked_data.get(category_key, [])
            if not category_list:
                raise PlannerAPIError(
                    status_code=404,
                    detail=f"No recommendation found for category '{category}'",
                )
            recommendation = category_list[0]

            model_name = recommendation.get("model_name") or recommendation.get("model_id")

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

            from planner import DeploymentConfiguration

            if isinstance(configuration, dict):
                config_obj = DeploymentConfiguration(**configuration)
            else:
                config_obj = configuration

            bundle = self._planner.generate_deployment(
                config_obj,
                namespace=namespace,
            )

            if not bundle.files:
                raise PlannerAPIError(
                    status_code=502,
                    detail="Planner generated no config files",
                )

            return DeploymentConfigResult(
                deployment_id=bundle.deployment_id,
                namespace=bundle.namespace,
                model_name=model_name,
                configs=bundle.files,
            )

        except PlannerAPIError:
            raise
        except ValueError as e:
            raise PlannerAPIError(status_code=400, detail=str(e)) from e
        except PlannerError as e:
            raise PlannerAPIError(status_code=502, detail=str(e)) from e

    def health_check(self) -> tuple[bool, str]:
        """Local planner is always available."""
        return True, "Planner available (local)"
