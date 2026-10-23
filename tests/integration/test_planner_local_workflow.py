"""Local navigator prerequisites using the real planner and mock cluster."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from planner import DeploymentIntent

from rhoai_mcp.composites.planner.client import PlannerAPIError
from rhoai_mcp.composites.planner.local_client import LocalPlannerClient
from rhoai_mcp.composites.planner.tools import VALID_USE_CASES, register_tools
from rhoai_mcp.config import PlannerMode, RHOAIConfig
from rhoai_mcp.mock_k8s.cluster_state import create_default_cluster_state
from rhoai_mcp.mock_k8s.mock_client import MockK8sClient


@pytest.fixture(scope="module", autouse=True)
def offline() -> Iterator[None]:
    """Stub optional GPU discovery and fail on any accidental network connection."""
    with (
        patch("planner.orchestration.workflow.detect_cluster_gpus", return_value=[]),
        patch(
            "socket.socket.connect", side_effect=AssertionError("Unexpected network access")
        ) as connect,
    ):
        yield
        connect.assert_not_called()


@pytest.fixture(scope="module")
def local() -> LocalPlannerClient:
    return LocalPlannerClient()


@pytest.fixture(scope="module")
def tools(local: LocalPlannerClient) -> Iterator[dict[str, Any]]:
    registered: dict[str, Any] = {}
    mcp = MagicMock()
    mcp.tool.side_effect = lambda: lambda fn: registered.setdefault(fn.__name__, fn)
    server = MagicMock()
    server.config = RHOAIConfig(
        mock_cluster=True, planner_mode=PlannerMode.LOCAL, planner_model_catalog_url=None
    )
    server.k8s = MockK8sClient(server.config, state=create_default_cluster_state())
    server.k8s.connect()
    # Exercise the existing local selector with the shared real library instance.
    with patch("rhoai_mcp.composites.planner.tools.LocalPlannerClient", return_value=local):
        register_tools(mcp, server)
        yield registered


@pytest.mark.parametrize("use_case", sorted(VALID_USE_CASES))
async def test_reference_defaults_match_embedded_specification(
    local: LocalPlannerClient, tools: dict[str, Any], use_case: str
) -> None:
    defaults = await tools["get_use_case_defaults"](use_case=use_case)
    assert "error" not in defaults, defaults
    spec = local._planner.generate_specification(DeploymentIntent(use_case=use_case, user_count=50))
    assert defaults["use_case"] == use_case
    assert defaults["description"]
    for metric in ("ttft", "itl", "e2e"):
        targets = defaults["slo_targets"][f"{metric}_ms"]
        assert targets["default"] == getattr(spec.slo_targets, f"{metric}_target_ms")
        target_range = getattr(spec.slo_targets, f"{metric}_range")
        assert (targets["min"], targets["max"]) == (target_range.min, target_range.max)
    assert defaults["workload"]["prompt_tokens"] == spec.workload_profile.prompt_tokens
    assert defaults["workload"]["output_tokens"] == spec.workload_profile.output_tokens


@pytest.mark.parametrize("users", [1, 50, 1000])
async def test_average_and_peak_traffic_match_embedded_capacity(
    local: LocalPlannerClient, tools: dict[str, Any], users: int
) -> None:
    result = await tools["get_expected_rps"](use_case="chatbot_conversational", user_count=users)
    assert "error" not in result, result
    spec = local._planner.generate_specification(
        DeploymentIntent(use_case="chatbot_conversational", user_count=users)
    )
    # Embedded recommendations include peak headroom and a minimum capacity floor.
    assert result["peak_rps"] == spec.workload_profile.expected_qps
    assert result["expected_concurrent_users"] == int(users * 0.2)
    assert result["expected_rps"] == round(int(users * 0.2) * 0.4 / 60, 2)
    assert 0 <= result["expected_rps"] <= result["peak_rps"]
    assert result["user_count"] == users


async def test_local_use_cases_cover_all_registered_values(tools: dict[str, Any]) -> None:
    result = await tools["list_use_cases"]()
    assert "error" not in result, result
    assert {case["id"] for case in result["use_cases"]} == VALID_USE_CASES
    assert result["count"] == len(VALID_USE_CASES)
    assert all(case["description"] for case in result["use_cases"])


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("get_slo_defaults", {"use_case": "unknown"}),
        ("get_workload_profile", {"use_case": "unknown"}),
        ("get_expected_rps", {"use_case": "unknown", "user_count": 50}),
        ("get_expected_rps", {"use_case": "chatbot_conversational", "user_count": 0}),
        ("get_expected_rps", {"use_case": "chatbot_conversational", "user_count": -1}),
    ],
)
async def test_invalid_reference_inputs_return_client_errors(
    local: LocalPlannerClient, method: str, arguments: dict[str, Any]
) -> None:
    with pytest.raises(PlannerAPIError) as raised:
        await getattr(local, method)(**arguments)
    assert raised.value.status_code == 400


async def test_local_plan_preserves_selected_model_and_gpu_metadata(
    tools: dict[str, Any],
) -> None:
    result = await tools["recommend_model"](
        text="support chatbot",
        use_case="chatbot_conversational",
        user_count=50,
        preferred_gpu_types=["H100", "A100-80"],
    )
    assert "error" not in result, result
    chosen = result["recommendations"]["top_balanced"]
    spec = result["specification"]
    plan = await tools["plan_deployment"](
        category="balanced",
        namespace="production-models",
        use_case=spec["use_case"],
        user_count=spec["user_count"],
        preferred_gpu_types=["H100", "A100-80"],
        **spec["slo_targets"],
        **spec["traffic_profile"],
    )
    assert "error" not in plan, plan
    assert plan["ready"] is True, plan
    assert plan["issues"] is None
    assert plan["model_id"] == chosen["model_id"]
    assert plan["namespace"] == "production-models"
    assert plan["runtime"] == "vllm-runtime"
    assert plan["gpu_type"] in {"H100", "A100-80"}
    assert chosen["gpu"] == f"{plan['gpu_count']}x {plan['gpu_type']}"
    assert plan["tensor_parallel"] >= 1
    assert plan["replicas"] >= 1
    assert plan["configs"]
    # Bundled models may lack a URI; planning should request it without losing metadata.
    if plan["storage_uri"] is None:
        assert plan["suggested_deploy_params"] is None
        assert any("storage_uri" in warning for warning in plan["warnings"])
    else:
        assert plan["suggested_deploy_params"]["model_id"] == chosen["model_id"]
