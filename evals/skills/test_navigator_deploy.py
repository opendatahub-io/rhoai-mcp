"""Live skill evaluation: the ``navigator-deploy`` deployment skill.

The decisive property here is consent. ``navigator-deploy`` writes to the
cluster, so the checks assert that the plan is shown before approval is
requested, that no deployment write precedes the scripted affirmative, and —
for the withheld-consent golden — that there are **zero** writes even after the
final user message. Honouring a refusal is successful completion of that
scenario, not a failure to finish.

    uv run --group skill-eval pytest evals/skills/test_navigator_deploy.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.skills.attempt import run_attempt
from evals.skills.goldens import DEPLOY_GOLDENS

if TYPE_CHECKING:
    from evals.agent.models import AgentModelSpec

pytestmark = [pytest.mark.eval, pytest.mark.asyncio, pytest.mark.skill_eval]


@pytest.mark.parametrize("golden", DEPLOY_GOLDENS, ids=lambda g: g.id)
async def test_navigator_deploy_golden(
    golden: Any,
    agent_spec: AgentModelSpec,
    skill_recorder: Any,
    skill_eval_config: Any,
    judge_llm: Any,
) -> None:
    """Run one navigator-deploy golden and assert only after recording."""
    outcome = await run_attempt(
        golden,
        agent_spec,
        recorder=skill_recorder,
        config=skill_eval_config,
        judge=judge_llm,
    )

    assert outcome.eligible, outcome.failure_summary()
    assert outcome.routing_ok, outcome.failure_summary()
    assert outcome.report.passed, outcome.failure_summary()

    if outcome.dry_run:
        assert not outcome.passed, "a dry run must never report an overall pass"
    else:
        assert outcome.passed, outcome.failure_summary()
