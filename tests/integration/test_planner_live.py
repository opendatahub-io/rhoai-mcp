"""Live E2E tests for the Planner client against a running llm-d-planner instance.

These tests exercise the full composable pipeline — generate-specification,
generate-recommendations, generate-deployment — and the MCP tool layer
(recommend_model, get_deployment_config) WITHOUT intent extraction.  Intent
extraction requires an LLM backend and is intentionally excluded.

Expensive API calls (generate-recommendations takes ~13s with benchmark data)
are made once via module-scoped fixtures and shared across tests.  Assertions
use structural and range-based checks to avoid leaking exact benchmark values.

Requirements:
    - A running llm-d-planner instance reachable over HTTP.
    - The env var RHOAI_MCP_PLANNER_URL set to its base URL
      (e.g. ``http://localhost:8000`` or a port-forwarded cluster service).

Run with::

    pytest tests/integration/test_planner_live.py -v -m live

The ``live`` marker is already registered in pyproject.toml; these tests are
skipped automatically when the planner is not reachable.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.composites.planner.client import (
    CATEGORY_MAP,
    PlannerAPIError,
    PlannerClient,
)
from rhoai_mcp.composites.planner.models import (
    DeploymentConfigResult,
    DeploymentIntent,
    RecommendationResult,
)
from rhoai_mcp.composites.planner.tools import register_tools

PLANNER_URL = os.environ.get("RHOAI_MCP_PLANNER_URL", "")
PLANNER_TIMEOUT = 120.0

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not PLANNER_URL, reason="RHOAI_MCP_PLANNER_URL not set"),
]

# -- Intents -------------------------------------------------------------------

CHATBOT_INTENT = DeploymentIntent(
    use_case="chatbot_conversational",
    user_count=1000,
    quality_priority="medium",
    latency_priority="high",
)

CODE_INTENT = DeploymentIntent(
    use_case="code_completion",
    user_count=500,
    preferred_gpu_types=["H100"],
    quality_priority="high",
)

# -- Shared fixtures (module-scoped to minimise API calls) ---------------------


@pytest.fixture(scope="module")
def planner() -> PlannerClient:
    """Module-scoped client pointing at the live planner instance."""
    client = PlannerClient(PLANNER_URL, timeout=PLANNER_TIMEOUT)
    healthy, msg = client.health_check()
    if not healthy:
        pytest.skip(f"Planner not reachable at {PLANNER_URL}: {msg}")
    return client


@pytest.fixture(scope="module")
def chatbot_spec(planner: PlannerClient) -> dict[str, Any]:
    """Specification generated once for chatbot_conversational."""
    return planner.generate_specification(CHATBOT_INTENT)


@pytest.fixture(scope="module")
def chatbot_ranked(planner: PlannerClient, chatbot_spec: dict[str, Any]) -> dict[str, Any]:
    """Ranked recommendations generated once from the chatbot specification."""
    return planner.generate_recommendations(chatbot_spec)


@pytest.fixture(scope="module")
def chatbot_balanced_config(chatbot_ranked: dict[str, Any]) -> dict[str, Any]:
    """Configuration block from the top balanced recommendation."""
    balanced = chatbot_ranked.get("balanced", [])
    if not balanced:
        pytest.skip("No balanced recommendations with current benchmark data")
    config = balanced[0].get("configuration")
    if config is None:
        pytest.skip("Balanced recommendation has no configuration block")
    return config


@pytest.fixture(scope="module")
def chatbot_bundle(
    planner: PlannerClient, chatbot_balanced_config: dict[str, Any]
) -> dict[str, Any]:
    """Deployment bundle generated once from the top balanced configuration."""
    return planner.generate_deployment(chatbot_balanced_config, namespace="test-ns")


@pytest.fixture(scope="module")
def recommend_result(planner: PlannerClient) -> RecommendationResult:
    """High-level recommend() result generated once with all overrides."""
    return planner.recommend(
        "unused — extraction is skipped",
        use_case_override="chatbot_conversational",
        user_count_override=1000,
        gpu_types_override=["H100", "A100-80"],
    )


@pytest.fixture(scope="module")
def deploy_config_result(planner: PlannerClient) -> DeploymentConfigResult | None:
    """High-level generate_config() result for balanced category, or None."""
    try:
        return planner.generate_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=200,
            itl_target_ms=65,
            e2e_target_ms=5000,
            namespace="live-test",
        )
    except PlannerAPIError as e:
        if "No recommendation found" in str(e):
            return None
        raise


def _register_live_tools() -> dict[str, Any]:
    """Register MCP tools with a mock FastMCP but real planner config."""
    mock_mcp = MagicMock()
    registered: dict[str, Any] = {}

    def capture_tool():
        def decorator(f):  # type: ignore[no-untyped-def]
            registered[f.__name__] = f
            return f

        return decorator

    mock_mcp.tool = capture_tool

    server = MagicMock()
    server.config.planner_url = PLANNER_URL
    server.config.planner_timeout = int(PLANNER_TIMEOUT)

    register_tools(mock_mcp, server)
    return registered


@pytest.fixture(scope="module")
def mcp_tools(planner: PlannerClient) -> dict[str, Any]:
    """Module-scoped dict of registered MCP tool functions backed by the live planner."""
    return _register_live_tools()


@pytest.fixture(scope="module")
def mcp_recommend_result(mcp_tools: dict[str, Any]) -> dict[str, Any]:
    """MCP recommend_model result generated once with all overrides."""
    recommend_model = mcp_tools["recommend_model"]
    return recommend_model(
        text="unused — all overrides provided",
        use_case="chatbot_conversational",
        user_count=1000,
        preferred_gpu_types=["H100", "A100-80"],
    )


@pytest.fixture(scope="module")
def mcp_deploy_result(mcp_tools: dict[str, Any]) -> dict[str, Any]:
    """MCP get_deployment_config result generated once for balanced category."""
    get_config = mcp_tools["get_deployment_config"]
    return get_config(
        category="balanced",
        use_case="chatbot_conversational",
        user_count=1000,
        prompt_tokens=512,
        output_tokens=256,
        expected_qps=10.0,
        ttft_target_ms=200,
        itl_target_ms=65,
        e2e_target_ms=5000,
        namespace="mcp-live-test",
    )


def _skip_if_no_data(ranked: dict[str, Any]) -> None:
    """Skip the current test if the planner returned no configurations."""
    if ranked.get("total_configs_evaluated", 0) == 0:
        pytest.skip("Planner has no benchmark data loaded")


def _skip_if_no_recommendations_in_tool_result(result: dict[str, Any]) -> None:
    """Skip if the MCP tool result indicates no recommendations (404)."""
    if "error" in result and result.get("status_code") == 404:
        pytest.skip("No recommendations with current benchmark data")


# == Health ====================================================================


def test_health_check(planner: PlannerClient) -> None:
    """Planner health endpoint returns healthy."""
    healthy, msg = planner.health_check()
    assert healthy is True
    assert msg


# == generate-specification ====================================================


def test_specification_chatbot_structure(chatbot_spec: dict[str, Any]) -> None:
    """Chatbot specification has all required top-level sections."""
    assert "slo_targets" in chatbot_spec
    assert "workload_profile" in chatbot_spec
    assert "priorities" in chatbot_spec


def test_specification_chatbot_slo_ranges(chatbot_spec: dict[str, Any]) -> None:
    """SLO targets are positive and within plausible ranges."""
    slo = chatbot_spec["slo_targets"]
    assert 1 < slo["ttft_target_ms"] < 10_000
    assert 1 < slo["itl_target_ms"] < 1_000
    assert 1 < slo["e2e_target_ms"] < 100_000


def test_specification_chatbot_workload(chatbot_spec: dict[str, Any]) -> None:
    """Workload profile has positive token counts and QPS."""
    wp = chatbot_spec["workload_profile"]
    assert wp["prompt_tokens"] > 0
    assert wp["output_tokens"] > 0
    assert wp["expected_qps"] > 0


def test_specification_code_completion(planner: PlannerClient) -> None:
    """generate-specification works for code_completion use case."""
    spec = planner.generate_specification(CODE_INTENT)
    assert 1 < spec["slo_targets"]["ttft_target_ms"] < 10_000
    assert "workload_profile" in spec


# == generate-recommendations ==================================================


def test_ranked_response_structure(chatbot_ranked: dict[str, Any]) -> None:
    """Ranked response contains all four category lists and metadata."""
    assert "total_configs_evaluated" in chatbot_ranked
    assert chatbot_ranked["total_configs_evaluated"] >= 0
    assert "configs_after_filters" in chatbot_ranked

    for key in ("balanced", "lowest_cost", "lowest_latency", "best_quality"):
        assert key in chatbot_ranked, f"Missing ranking category: {key}"
        assert isinstance(chatbot_ranked[key], list)


def test_ranked_response_has_results(chatbot_ranked: dict[str, Any]) -> None:
    """With benchmark data loaded, at least some recommendations exist."""
    _skip_if_no_data(chatbot_ranked)
    assert chatbot_ranked["configs_after_filters"] > 0
    assert any(
        len(chatbot_ranked.get(cat, [])) > 0
        for cat in ("balanced", "lowest_cost", "lowest_latency", "best_quality")
    )


def test_recommendation_scores_structure(chatbot_ranked: dict[str, Any]) -> None:
    """Each recommendation has scores with expected fields and plausible ranges."""
    _skip_if_no_data(chatbot_ranked)
    balanced = chatbot_ranked.get("balanced", [])
    if not balanced:
        pytest.skip("No balanced recommendations")

    scores = balanced[0]["scores"]
    for field in ("quality_score", "price_score", "latency_score", "balanced_score"):
        assert field in scores, f"Missing score field: {field}"
        assert isinstance(scores[field], (int, float))
        assert 0 <= scores[field] <= 100


def test_recommendation_has_configuration(chatbot_balanced_config: dict[str, Any]) -> None:
    """The top balanced recommendation carries a configuration block."""
    assert "model_id" in chatbot_balanced_config
    assert "gpu_config" in chatbot_balanced_config


def test_recommendation_gpu_config_structure(chatbot_balanced_config: dict[str, Any]) -> None:
    """GPU config has type, count, and positive values."""
    gpu = chatbot_balanced_config["gpu_config"]
    assert isinstance(gpu["gpu_type"], str)
    assert gpu["gpu_count"] >= 1


def test_constraints_reduce_results(
    planner: PlannerClient,
    chatbot_spec: dict[str, Any],
    chatbot_ranked: dict[str, Any],
) -> None:
    """Tight constraints produce fewer or equal results compared to unconstrained."""
    constrained = planner.generate_recommendations(chatbot_spec, min_quality=95, max_cost=100.0)
    assert constrained["configs_after_filters"] <= chatbot_ranked["configs_after_filters"]


# == generate-deployment =======================================================


def test_bundle_structure(chatbot_bundle: dict[str, Any]) -> None:
    """Deployment bundle has deployment_id, namespace, and files."""
    assert "deployment_id" in chatbot_bundle
    assert chatbot_bundle["namespace"] == "test-ns"
    assert isinstance(chatbot_bundle.get("files"), dict)
    assert len(chatbot_bundle["files"]) > 0


def test_bundle_contains_inferenceservice(chatbot_bundle: dict[str, Any]) -> None:
    """Deployment bundle includes an InferenceService manifest."""
    files = chatbot_bundle["files"]
    assert any("inferenceservice" in k.lower() for k in files), (
        f"Expected an InferenceService file, got keys: {list(files.keys())}"
    )


def test_bundle_files_are_yaml(chatbot_bundle: dict[str, Any]) -> None:
    """Each file in the bundle looks like YAML (contains apiVersion or kind)."""
    for name, content in chatbot_bundle["files"].items():
        assert isinstance(content, str), f"File '{name}' content should be a string"
        assert "apiVersion" in content or "kind" in content, (
            f"File '{name}' doesn't look like a Kubernetes manifest"
        )


# == High-level client: recommend() ===========================================


def test_recommend_returns_result(recommend_result: RecommendationResult) -> None:
    """recommend() with all overrides returns a RecommendationResult."""
    assert isinstance(recommend_result, RecommendationResult)
    assert recommend_result.specification["use_case"] == "chatbot_conversational"
    assert recommend_result.specification["user_count"] == 1000
    assert recommend_result.total_configs_evaluated >= 0


def test_recommend_has_categories(recommend_result: RecommendationResult) -> None:
    """With benchmark data, at least one recommendation category is populated."""
    if recommend_result.total_configs_evaluated == 0:
        pytest.skip("No benchmark data")
    has_any = any([
        recommend_result.top_balanced,
        recommend_result.top_cost,
        recommend_result.top_performance,
        recommend_result.top_quality,
    ])
    assert has_any, "Expected at least one recommendation category"


def test_recommend_slo_overrides(planner: PlannerClient) -> None:
    """SLO overrides are reflected in the returned specification."""
    result = planner.recommend(
        "unused",
        use_case_override="chatbot_conversational",
        user_count_override=1000,
        gpu_types_override=[],
        ttft_override_ms=100,
        itl_override_ms=30,
        e2e_override_ms=1500,
    )
    assert result.specification["slo_targets"]["ttft_target_ms"] == 100
    assert result.specification["slo_targets"]["itl_target_ms"] == 30
    assert result.specification["slo_targets"]["e2e_target_ms"] == 1500


# == High-level client: generate_config() =====================================


def test_generate_config_returns_result(
    deploy_config_result: DeploymentConfigResult | None,
) -> None:
    """generate_config for balanced category returns a DeploymentConfigResult."""
    if deploy_config_result is None:
        pytest.skip("No recommendations with current benchmark data")
    assert isinstance(deploy_config_result, DeploymentConfigResult)
    assert deploy_config_result.deployment_id
    assert deploy_config_result.namespace == "live-test"


def test_generate_config_has_configs(
    deploy_config_result: DeploymentConfigResult | None,
) -> None:
    """generate_config result contains Kubernetes config files."""
    if deploy_config_result is None:
        pytest.skip("No recommendations with current benchmark data")
    assert len(deploy_config_result.configs) > 0
    assert any("inferenceservice" in k.lower() for k in deploy_config_result.configs)


@pytest.mark.parametrize("category", list(CATEGORY_MAP.keys()))
def test_generate_config_all_categories(planner: PlannerClient, category: str) -> None:
    """generate_config works for every valid category."""
    try:
        result = planner.generate_config(
            category=category,
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=200,
            itl_target_ms=65,
            e2e_target_ms=5000,
        )
    except PlannerAPIError as e:
        if "No recommendation found" in str(e):
            pytest.skip(
                f"No recommendations for category '{category}' with current benchmark data"
            )
            return
        raise

    assert isinstance(result, DeploymentConfigResult)
    assert result.deployment_id
    assert len(result.configs) > 0


# == GET endpoints =============================================================


def test_get_slo_defaults(planner: PlannerClient) -> None:
    """GET slo-defaults returns defaults within plausible ranges."""
    defaults = planner.get_slo_defaults("chatbot_conversational")
    slo = defaults["slo_defaults"]
    assert 1 < slo["ttft_ms"]["default"] < 10_000
    assert 1 < slo["itl_ms"]["default"] < 1_000
    assert 1 < slo["e2e_ms"]["default"] < 100_000


def test_get_workload_profile(planner: PlannerClient) -> None:
    """GET workload-profile returns positive token counts."""
    profile = planner.get_workload_profile("chatbot_conversational")
    wp = profile["workload_profile"]
    assert wp["prompt_tokens"] > 0
    assert wp["output_tokens"] > 0


def test_get_expected_rps(planner: PlannerClient) -> None:
    """GET expected-rps returns a positive RPS estimate."""
    rps = planner.get_expected_rps("chatbot_conversational", 1000)
    assert rps["expected_rps"] > 0


# == MCP tool: recommend_model ================================================


def test_mcp_recommend_structure(mcp_recommend_result: dict[str, Any]) -> None:
    """recommend_model MCP tool returns specification and recommendations."""
    assert "error" not in mcp_recommend_result, f"Tool error: {mcp_recommend_result}"
    assert "specification" in mcp_recommend_result
    assert mcp_recommend_result["specification"]["use_case"] == "chatbot_conversational"
    assert "recommendations" in mcp_recommend_result


def test_mcp_recommend_has_entries(mcp_recommend_result: dict[str, Any]) -> None:
    """With benchmark data, recommendations dict is populated."""
    recs = mcp_recommend_result.get("recommendations", {})
    if not recs:
        pytest.skip("No recommendations with current benchmark data")

    for key, entry in recs.items():
        assert "model" in entry, f"'{key}' missing 'model'"
        assert "meets_slo" in entry, f"'{key}' missing 'meets_slo'"
        if "cost_usd_month" in entry:
            assert entry["cost_usd_month"] > 0


def test_mcp_recommend_top_quality_has_score(mcp_recommend_result: dict[str, Any]) -> None:
    """top_quality recommendation includes a quality score in [0, 100]."""
    recs = mcp_recommend_result.get("recommendations", {})
    if "top_quality" not in recs:
        pytest.skip("No top_quality recommendation")
    score = recs["top_quality"].get("score")
    assert score is not None, "top_quality should include quality_score"
    assert 0 <= score <= 100


def test_mcp_recommend_top_balanced_has_score(mcp_recommend_result: dict[str, Any]) -> None:
    """top_balanced recommendation includes a balanced score in [0, 100]."""
    recs = mcp_recommend_result.get("recommendations", {})
    if "top_balanced" not in recs:
        pytest.skip("No top_balanced recommendation")
    score = recs["top_balanced"].get("score")
    assert score is not None, "top_balanced should include balanced_score"
    assert 0 <= score <= 100


def test_mcp_recommend_slo_overrides(mcp_tools: dict[str, Any]) -> None:
    """recommend_model MCP tool honours SLO override parameters."""
    recommend_model = mcp_tools["recommend_model"]
    result = recommend_model(
        text="unused",
        use_case="chatbot_conversational",
        user_count=1000,
        preferred_gpu_types=[],
        ttft_max_ms=100,
        itl_max_ms=30,
        e2e_max_ms=1500,
    )
    assert "error" not in result, f"Tool error: {result}"
    spec = result["specification"]
    assert spec["slo_targets"]["ttft_target_ms"] == 100
    assert spec["slo_targets"]["itl_target_ms"] == 30
    assert spec["slo_targets"]["e2e_target_ms"] == 1500


def test_mcp_recommend_validation_errors(mcp_tools: dict[str, Any]) -> None:
    """recommend_model returns error dicts for invalid inputs without hitting the API."""
    recommend_model = mcp_tools["recommend_model"]

    result = recommend_model(text="test", use_case="invalid_case")
    assert "error" in result and "use_case" in result["error"]

    result = recommend_model(text="test", optimization_profile="turbo")
    assert "error" in result and "optimization_profile" in result["error"]

    result = recommend_model(text="test", min_quality=101)
    assert "error" in result and "min_quality" in result["error"]

    result = recommend_model(text="test", preferred_gpu_types=["V100"])
    assert "error" in result and "V100" in result["error"]


# == MCP tool: get_deployment_config ===========================================


def test_mcp_deploy_config_structure(mcp_deploy_result: dict[str, Any]) -> None:
    """get_deployment_config MCP tool returns deployment_id, namespace, and configs."""
    _skip_if_no_recommendations_in_tool_result(mcp_deploy_result)
    assert "error" not in mcp_deploy_result, f"Tool error: {mcp_deploy_result}"
    assert mcp_deploy_result["deployment_id"]
    assert mcp_deploy_result["namespace"] == "mcp-live-test"


def test_mcp_deploy_config_has_model(mcp_deploy_result: dict[str, Any]) -> None:
    """get_deployment_config result includes a model name."""
    _skip_if_no_recommendations_in_tool_result(mcp_deploy_result)
    assert "error" not in mcp_deploy_result
    assert "model" in mcp_deploy_result
    assert isinstance(mcp_deploy_result["model"], str)


def test_mcp_deploy_config_has_yaml(mcp_deploy_result: dict[str, Any]) -> None:
    """get_deployment_config result includes InferenceService YAML."""
    _skip_if_no_recommendations_in_tool_result(mcp_deploy_result)
    assert "error" not in mcp_deploy_result
    configs = mcp_deploy_result["configs"]
    assert len(configs) > 0
    assert any("inferenceservice" in k.lower() for k in configs)


def test_mcp_deploy_config_validation_errors(mcp_tools: dict[str, Any]) -> None:
    """get_deployment_config returns error dicts for invalid inputs."""
    get_config = mcp_tools["get_deployment_config"]

    base: dict[str, Any] = dict(
        use_case="chatbot_conversational",
        user_count=1000,
        prompt_tokens=512,
        output_tokens=256,
        expected_qps=10.0,
        ttft_target_ms=200,
        itl_target_ms=65,
        e2e_target_ms=5000,
    )

    result = get_config(category="fastest", **base)
    assert "error" in result and "category" in result["error"]

    result = get_config(category="balanced", **{**base, "use_case": "invalid"})
    assert "error" in result and "use_case" in result["error"]

    result = get_config(category="balanced", **{**base, "user_count": 0})
    assert "error" in result and "user_count" in result["error"]

    result = get_config(category="balanced", **{**base, "namespace": "INVALID!"})
    assert "error" in result and "namespace" in result["error"]
