"""Live skill evaluation: the ``navigator`` model-selection skill.

Discovery mode — both skills are in the catalog and the agent must pick, load,
and follow the right one from a natural-language request (plan §2.1).

Runs against the stdio ``rhoai-mcp`` server with ``--mock-cluster``; no LCORE,
no compose, no live cluster. Dry run by default (deterministic checks only);
set ``RHOAI_EVAL_SKILL_JUDGE=1`` to add judged metrics.

    uv run --group skill-eval pytest evals/skills/test_navigator.py -v
    RHOAI_EVAL_AGENT_MODELS=enmaas-glm uv run --group skill-eval pytest evals/skills -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.skills.attempt import run_attempt
from evals.skills.goldens import NAVIGATOR_GOLDENS

if TYPE_CHECKING:
    from evals.agent.models import AgentModelSpec

pytestmark = [pytest.mark.eval, pytest.mark.asyncio]


@pytest.mark.parametrize("golden", NAVIGATOR_GOLDENS, ids=lambda g: g.id)
async def test_navigator_golden(
    golden: Any,
    agent_spec: AgentModelSpec,
    skill_recorder: Any,
    skill_eval_config: Any,
    judge_llm: Any,
) -> None:
    """Run one navigator golden and assert only after it has been recorded."""
    outcome = await run_attempt(
        golden,
        agent_spec,
        recorder=skill_recorder,
        config=skill_eval_config,
        judge=judge_llm,
    )

    # Reported, never a gate: allowed-tools is optional per the agentskills spec
    # and unimplemented by the harness (plan §3.2).
    assert outcome.eligible, outcome.failure_summary()
    assert outcome.routing_ok, outcome.failure_summary()
    # Hard workflow checks decide pass/fail independently of any judge.
    assert outcome.report.passed, outcome.failure_summary()

    if outcome.dry_run:
        # A dry run has no judged evidence and must not be read as a pass.
        assert not outcome.passed, "a dry run must never report an overall pass"
    else:
        assert outcome.passed, outcome.failure_summary()
