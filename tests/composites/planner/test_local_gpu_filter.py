"""Local recommendations must honor the same GPU boundary as the HTTP client."""

from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest

from rhoai_mcp.composites.planner.local_client import LocalPlannerClient
from tests.composites.planner.test_local_client import _make_mock_planner


@pytest.mark.parametrize(
    "allowed,returned,expected",
    [
        (["A100-80"], ["H100", "NVIDIA-A100-80"], "NVIDIA-A100-80"),
        (["A100-80"], ["H100"], None),
        (["A100-80"], [None, "L4"], None),
        (["A100-80", "H100"], ["H100", "A100-80"], "H100"),
        ([], ["H100", "A100-80"], "H100"),
        (["A100-80"], ["L4", "A100-80"], "A100-80"),
    ],
)
async def test_all_local_slots_filter_before_selecting_top(
    allowed: list[str], returned: list[str | None], expected: str | None
) -> None:
    planner = _make_mock_planner()
    template = planner.generate_recommendations.return_value.model_dump.return_value["balanced"][0]
    candidates: list[dict[str, Any]] = []
    for gpu in returned:
        candidate = deepcopy(template)
        candidate["gpu_config"] = {**template["gpu_config"], "gpu_type": gpu} if gpu else None
        candidate["configuration"]["gpu_config"] = candidate["gpu_config"]
        candidates.append(candidate)
    ranked = {
        key: deepcopy(candidates)
        for key in ("balanced", "lowest_cost", "lowest_latency", "best_quality")
    }
    planner.generate_recommendations.return_value.model_dump.return_value = ranked
    before = deepcopy(ranked)
    with patch("rhoai_mcp.composites.planner.local_client.Planner", return_value=planner):
        result = await LocalPlannerClient().recommend(
            text="Support chatbot",
            use_case_override="chatbot_conversational",
            user_count_override=50,
            gpu_types_override=allowed,
        )
    for key in ("top_balanced", "top_cost", "top_performance", "top_quality"):
        selected = getattr(result, key)
        if expected is None:
            assert selected is None, key
        else:
            assert selected is not None and selected.gpu_config.gpu_type == expected, key
    assert ranked == before
    assert result.total_configs_evaluated == 2847
    assert result.configs_after_filters == 542  # Upstream diagnostics retain their meaning.


async def test_real_local_gpu_miss_requires_an_explicit_unconstrained_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PLANNER_DETECT_CLUSTER_GPUS", "false")
    client = LocalPlannerClient()
    params = {
        "text": "Support chatbot",
        "use_case_override": "chatbot_conversational",
        "user_count_override": 50,
        "ttft_override_ms": 44,
    }
    constrained = await client.recommend(**params, gpu_types_override=["A100-80"])
    assert all(
        getattr(constrained, slot) is None
        for slot in ("top_balanced", "top_cost", "top_performance", "top_quality")
    )
    fallback = await client.recommend(**params, gpu_types_override=[])
    assert fallback.top_performance is not None
    assert fallback.top_performance.gpu_config.gpu_type.removeprefix("NVIDIA-") == "H100"
    assert constrained.specification["traffic_profile"] == fallback.specification["traffic_profile"]
