"""Integration tests for the Planner local (embedded library) mode.

These tests exercise the full pipeline — recommend, generate_config — and the
MCP tool layer (recommend_model, get_deployment_config) using a real
``planner.Planner`` instance backed by an in-memory SQLite database with
bundled BLIS benchmark data.  No external service or network access is needed.

Unlike unit tests in ``tests/composites/planner/test_local_client.py`` which
mock the ``Planner`` class entirely, these tests validate real data shapes,
field names, and value ranges produced by the planner library.

Run with::

    pytest tests/integration/test_planner_local.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rhoai_mcp.composites.planner.client import CATEGORY_MAP, PlannerAPIError
from rhoai_mcp.composites.planner.local_client import LocalPlannerClient
from rhoai_mcp.composites.planner.models import (
    DeploymentConfigResult,
    RecommendationResult,
)
from rhoai_mcp.composites.planner.tools import register_tools
from rhoai_mcp.config import PlannerMode

# -- Shared fixtures (module-scoped — load_bundled_benchmarks() runs once) ----


@pytest.fixture(scope="module")
def local_client() -> LocalPlannerClient:
    """Real LocalPlannerClient with bundled benchmarks loaded."""
    return LocalPlannerClient()


@pytest.fixture(scope="module")
def recommend_result(local_client: LocalPlannerClient) -> RecommendationResult:
    """Recommendation result generated once with chatbot overrides."""
    return local_client.recommend(
        "unused — local mode ignores text",
        use_case_override="chatbot_conversational",
        user_count_override=1000,
        gpu_types_override=["H100", "A100-80"],
    )


@pytest.fixture(scope="module")
def deploy_config_result(local_client: LocalPlannerClient) -> DeploymentConfigResult | None:
    """Deployment config result for balanced category, or None if no match."""
    try:
        return local_client.generate_config(
            category="balanced",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=200,
            itl_target_ms=65,
            e2e_target_ms=5000,
            namespace="local-test",
        )
    except PlannerAPIError as e:
        if "No recommendation found" in str(e):
            return None
        raise


def _register_local_tools() -> dict[str, Any]:
    """Register MCP tools with a mock FastMCP backed by a real local planner."""
    mock_mcp = MagicMock()
    registered: dict[str, Any] = {}

    def capture_tool():
        def decorator(f):  # type: ignore[no-untyped-def]
            registered[f.__name__] = f
            return f

        return decorator

    mock_mcp.tool = capture_tool

    server = MagicMock()
    server.config.planner_mode = PlannerMode.LOCAL
    server.config.planner_model_catalog_url = None

    register_tools(mock_mcp, server)
    return registered


@pytest.fixture(scope="module")
def mcp_tools() -> dict[str, Any]:
    """Module-scoped dict of registered MCP tool functions backed by the real local planner."""
    return _register_local_tools()


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
        namespace="mcp-local-test",
    )


def _skip_if_no_recommendations(result: dict[str, Any]) -> None:
    """Skip if the MCP tool result indicates no recommendations."""
    if "error" in result and result.get("status_code") == 404:
        pytest.skip("No recommendations with bundled benchmark data")


# == Health ====================================================================


def test_health_check(local_client: LocalPlannerClient) -> None:
    """Local planner health check always returns healthy."""
    healthy, msg = local_client.health_check()
    assert healthy is True
    assert "local" in msg.lower()


# == High-level client: recommend() ===========================================


def test_recommend_returns_result(recommend_result: RecommendationResult) -> None:
    """recommend() with all overrides returns a RecommendationResult."""
    assert isinstance(recommend_result, RecommendationResult)
    assert recommend_result.specification["use_case"] == "chatbot_conversational"
    assert recommend_result.specification["user_count"] == 1000
    assert recommend_result.total_configs_evaluated >= 0


def test_recommend_specification_structure(recommend_result: RecommendationResult) -> None:
    """Specification summary has SLO targets and traffic profile."""
    spec = recommend_result.specification
    assert "slo_targets" in spec
    slo = spec["slo_targets"]
    assert 1 < slo["ttft_target_ms"] < 10_000
    assert 1 < slo["itl_target_ms"] < 1_000
    assert 1 < slo["e2e_target_ms"] < 100_000

    assert "traffic_profile" in spec
    tp = spec["traffic_profile"]
    assert tp["prompt_tokens"] > 0
    assert tp["output_tokens"] > 0
    assert tp["expected_qps"] > 0


def test_recommend_has_categories(recommend_result: RecommendationResult) -> None:
    """With bundled benchmarks, at least one recommendation category is populated."""
    if recommend_result.total_configs_evaluated == 0:
        pytest.skip("No benchmark data")
    has_any = any([
        recommend_result.top_balanced,
        recommend_result.top_cost,
        recommend_result.top_performance,
        recommend_result.top_quality,
    ])
    assert has_any, "Expected at least one recommendation category"


def test_recommend_scores_in_range(recommend_result: RecommendationResult) -> None:
    """Recommendation scores are within [0, 100]."""
    for rec in [
        recommend_result.top_balanced,
        recommend_result.top_cost,
        recommend_result.top_performance,
        recommend_result.top_quality,
    ]:
        if rec is None or rec.scores is None:
            continue
        for field in ("quality_score", "price_score", "latency_score", "balanced_score"):
            score = getattr(rec.scores, field)
            assert 0 <= score <= 100, f"{field} = {score} out of range"


def test_recommend_slo_overrides(local_client: LocalPlannerClient) -> None:
    """SLO overrides are reflected in the returned specification."""
    result = local_client.recommend(
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


def test_recommend_missing_overrides_raises(local_client: LocalPlannerClient) -> None:
    """Missing required overrides raises PlannerAPIError(400)."""
    with pytest.raises(PlannerAPIError) as exc_info:
        local_client.recommend(text="chatbot for 1000 users")
    assert exc_info.value.status_code == 400
    assert "use_case" in exc_info.value.detail


# == High-level client: generate_config() =====================================


def test_generate_config_returns_result(
    deploy_config_result: DeploymentConfigResult | None,
) -> None:
    """generate_config for balanced category returns a DeploymentConfigResult."""
    if deploy_config_result is None:
        pytest.skip("No recommendations with bundled benchmark data")
    assert isinstance(deploy_config_result, DeploymentConfigResult)
    assert deploy_config_result.deployment_id
    assert deploy_config_result.namespace == "local-test"


def test_generate_config_has_yaml(
    deploy_config_result: DeploymentConfigResult | None,
) -> None:
    """generate_config result contains Kubernetes config files."""
    if deploy_config_result is None:
        pytest.skip("No recommendations with bundled benchmark data")
    assert len(deploy_config_result.configs) > 0
    assert any("inferenceservice" in k.lower() for k in deploy_config_result.configs)


def test_generate_config_yaml_looks_valid(
    deploy_config_result: DeploymentConfigResult | None,
) -> None:
    """Config files look like Kubernetes YAML manifests."""
    if deploy_config_result is None:
        pytest.skip("No recommendations with bundled benchmark data")
    for name, content in deploy_config_result.configs.items():
        assert isinstance(content, str), f"File '{name}' content should be a string"
        assert "apiVersion" in content or "kind" in content, (
            f"File '{name}' doesn't look like a Kubernetes manifest"
        )


@pytest.mark.parametrize("category", list(CATEGORY_MAP.keys()))
def test_generate_config_all_categories(
    local_client: LocalPlannerClient, category: str
) -> None:
    """generate_config works for every valid category."""
    try:
        result = local_client.generate_config(
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
                f"No recommendations for category '{category}' with bundled benchmark data"
            )
            return
        raise

    assert isinstance(result, DeploymentConfigResult)
    assert result.deployment_id
    assert len(result.configs) > 0


def test_generate_config_invalid_category(local_client: LocalPlannerClient) -> None:
    """Invalid category raises PlannerAPIError(400)."""
    with pytest.raises(PlannerAPIError) as exc_info:
        local_client.generate_config(
            category="fastest",
            use_case="chatbot_conversational",
            user_count=1000,
            prompt_tokens=512,
            output_tokens=256,
            expected_qps=10.0,
            ttft_target_ms=200,
            itl_target_ms=65,
            e2e_target_ms=5000,
        )
    assert exc_info.value.status_code == 400


# == MCP tool: recommend_model ================================================


def test_mcp_recommend_structure(mcp_recommend_result: dict[str, Any]) -> None:
    """recommend_model MCP tool returns specification and recommendations."""
    assert "error" not in mcp_recommend_result, f"Tool error: {mcp_recommend_result}"
    assert "specification" in mcp_recommend_result
    assert mcp_recommend_result["specification"]["use_case"] == "chatbot_conversational"
    assert "recommendations" in mcp_recommend_result


def test_mcp_recommend_has_entries(mcp_recommend_result: dict[str, Any]) -> None:
    """With bundled benchmarks, recommendations dict is populated."""
    recs = mcp_recommend_result.get("recommendations", {})
    if not recs:
        pytest.skip("No recommendations with bundled benchmark data")

    for key, entry in recs.items():
        assert "model" in entry, f"'{key}' missing 'model'"
        assert "meets_slo" in entry, f"'{key}' missing 'meets_slo'"
        if "cost_usd_month" in entry:
            assert entry["cost_usd_month"] > 0


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
    """recommend_model returns error dicts for invalid inputs without hitting the planner."""
    recommend_model = mcp_tools["recommend_model"]

    result = recommend_model(text="test", use_case="invalid_case")
    assert "error" in result and "use_case" in result["error"]

    result = recommend_model(text="test", optimization_profile="turbo")
    assert "error" in result and "optimization_profile" in result["error"]

    result = recommend_model(text="test", min_quality=101)
    assert "error" in result and "min_quality" in result["error"]

    result = recommend_model(text="test", preferred_gpu_types=["V100"])
    assert "error" in result and "V100" in result["error"]


# == MCP tool: get_deployment_config ==========================================


def test_mcp_deploy_config_structure(mcp_deploy_result: dict[str, Any]) -> None:
    """get_deployment_config MCP tool returns deployment_id, namespace, and configs."""
    _skip_if_no_recommendations(mcp_deploy_result)
    assert "error" not in mcp_deploy_result, f"Tool error: {mcp_deploy_result}"
    assert mcp_deploy_result["deployment_id"]
    assert mcp_deploy_result["namespace"] == "mcp-local-test"


def test_mcp_deploy_config_has_model(mcp_deploy_result: dict[str, Any]) -> None:
    """get_deployment_config result includes a model name."""
    _skip_if_no_recommendations(mcp_deploy_result)
    assert "error" not in mcp_deploy_result
    assert "model" in mcp_deploy_result
    assert isinstance(mcp_deploy_result["model"], str)


def test_mcp_deploy_config_has_yaml(mcp_deploy_result: dict[str, Any]) -> None:
    """get_deployment_config result includes InferenceService YAML."""
    _skip_if_no_recommendations(mcp_deploy_result)
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
