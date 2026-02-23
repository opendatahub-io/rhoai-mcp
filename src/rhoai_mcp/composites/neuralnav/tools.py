"""MCP tools for NeuralNav deployment recommendation (model + GPU only)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)


def _base_url(server: "RHOAIServer") -> str | None:
    url = getattr(server.config, "neuralnav_backend_url", None)
    if url:
        return url.rstrip("/")
    return None


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    @mcp.tool()
    def get_deployment_recommendation(
        use_case: str,
        user_count: int,
        prompt_tokens: int = 512,
        output_tokens: int = 256,
        expected_qps: float = 10.0,
        ttft_target_ms: int = 500,
        itl_target_ms: int = 100,
        e2e_target_ms: int = 5000,
        percentile: str = "p95",
        preferred_gpu_types: list[str] | None = None,
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
    ) -> dict[str, Any]:
        """Get ranked deployment recommendations (model + GPU only) from NeuralNav.

        Returns five ranked views: balanced, best_accuracy, lowest_cost, lowest_latency, simplest.
        Each config includes model name, GPU config, scores, and cost.

        Args:
            use_case: Use case key (e.g. chatbot_conversational, code_completion, translation).
            user_count: Expected number of users (drives QPS and capacity).
            prompt_tokens: Typical prompt length (default 512).
            output_tokens: Typical output length (default 256).
            expected_qps: Expected queries per second (default 10).
            ttft_target_ms: Target time-to-first-token in ms (default 500).
            itl_target_ms: Target inter-token latency in ms (default 100).
            e2e_target_ms: Target end-to-end latency in ms (default 5000).
            percentile: Latency percentile (mean, p90, p95, p99; default p95).
            preferred_gpu_types: Optional list of GPU types to prefer.
            min_accuracy: Optional minimum accuracy score filter.
            max_cost: Optional maximum monthly cost filter (USD).
            include_near_miss: Include configs that nearly meet SLOs (default True).

        Returns:
            Dict with balanced, best_accuracy, lowest_cost, lowest_latency, simplest
            (lists of model+GPU configs), specification, and stats.
        """
        base = _base_url(server)
        if not base:
            return {
                "error": "NeuralNav service URL not configured",
                "hint": "Set RHOAI_MCP_NEURALNAV_BACKEND_URL to the NeuralNav backend URL (e.g. http://localhost:8000)",
            }
        payload: dict[str, Any] = {
            "use_case": use_case,
            "user_count": user_count,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "expected_qps": expected_qps,
            "ttft_target_ms": ttft_target_ms,
            "itl_target_ms": itl_target_ms,
            "e2e_target_ms": e2e_target_ms,
            "percentile": percentile,
            "include_near_miss": include_near_miss,
        }
        if preferred_gpu_types is not None:
            payload["preferred_gpu_types"] = preferred_gpu_types
        if min_accuracy is not None:
            payload["min_accuracy"] = min_accuracy
        if max_cost is not None:
            payload["max_cost"] = max_cost
        try:
            with httpx.Client(timeout=60) as client:
                r = client.post(f"{base}/api/ranked-recommend-from-spec", json=payload)
            r.raise_for_status()
            data = r.json()
            # Return only model + GPU; strip agent/prompt fields
            for key in ("recommended_agent", "recommended_system_prompt", "prompt_eval_dataset_path", "recommended_agent_explanation"):
                data.pop(key, None)
            return data
        except httpx.HTTPStatusError as e:
            logger.warning("Deployment recommendation HTTP error: %s %s", e.response.status_code, e.response.text[:200])
            try:
                err = e.response.json()
                return {"error": err.get("detail", str(e)), "status_code": e.response.status_code}
            except Exception:
                return {"error": e.response.text or str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.exception("Deployment recommendation failed")
            return {"error": str(e)}
