"""Regression checks for the navigator deployment skill's hardware fallback."""

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude/skills/navigator-deploy/SKILL.md"


def test_gpu_discovery_fallback_handles_unavailable_tool_and_unknown_inventory() -> None:
    """Hardware discovery must ask before running an unconstrained recommendation."""
    phase = (
        SKILL.read_text()
        .split("## Phase 3 — Get ranked configurations", 1)[1]
        .split("## Phase 4 — Deployment plan", 1)[0]
    )

    assert "If `get_cluster_resources` is available in this session, call it." in phase
    assert (
        "If `get_cluster_resources` is unavailable, fails, or returns missing, empty, "
        "or unmappable GPU inventory" in phase
    )
    assert "Do not substitute other cluster tools or infer hardware from benchmark data" in phase
    assert "Wait for the customer's answer before calling `recommend_model`" in phase
    assert (
        "Tool unavailability alone is not permission to recommend without a hardware filter"
        in phase
    )
    assert "If they provide GPU types" in phase
    assert "If they say they don't know or explicitly request unfiltered recommendations" in phase
    assert "Skip Step 2 and go straight to Step 3" not in phase
