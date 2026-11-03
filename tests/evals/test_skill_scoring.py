"""Judge failures must not turn an incomplete metric set into a pass."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


def test_normal_scoring_keeps_item_evidence_without_extra_judge_calls() -> None:
    from pathlib import Path

    from evals.config import EvalConfig
    from evals.metrics.scorecards import Scorecard
    from evals.skills.attempt import _score
    from evals.skills.checks import CheckReport

    from tests.evals.test_conversation_metrics import UnusedJudge
    from tests.evals.test_judge_calibration import approved_source
    from tests.evals.test_scorecards import card

    class ItemJudge(UnusedJudge):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, schema: Any = None, **_kwargs: Any) -> Any:
            self.calls += 1
            if schema is Scorecard:
                rubric = prompt.split("Evaluate the ", 1)[1].split(" dimension", 1)[0]
                return schema.model_validate(card(rubric, efficiency=1, clarity=1))
            return schema(score=9, reason="offline fidelity grade")

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            return self.generate(prompt, schema, **kwargs)

    judge = ItemJudge()
    with patch(
        "evals.agent.transcript.AttemptArtifact.load",
        return_value=SimpleNamespace(mcp_tools=lambda: []),
    ):
        records, _ = _score(
            approved_source(),
            Path("unused"),
            "navigator-deploy",
            EvalConfig(_env_file=None),
            judge,
            CheckReport(),
        )
    assert len(records) == 3
    assert [len(r.scorecards) for r in records] == [2, 1, 0]
    assert all(r.success for r in records)
    assert records[0].version == "conversation-v3-items"
    assert records[2].version == "v2"
    assert records[2].score == 0.9  # DeepEval normalizes the supplied 9/10 GEval grade.
    assert judge.calls == 4


def test_fidelity_judge_error_is_a_failed_metric() -> None:
    from evals.config import EvalConfig
    from evals.skills.attempt import _score_fidelity
    from evals.skills.checks import CheckReport

    metric = SimpleNamespace(__name__="Navigator Skill Fidelity", threshold=0.8)
    with (
        patch("evals.skills.rubrics.create_fidelity_metric", return_value=metric),
        patch(
            "evals.deepeval_helpers.transcript_to_conversational_test_case", return_value=object()
        ),
        patch("deepeval.evaluate.evaluate", side_effect=RuntimeError("judge unavailable")),
    ):
        records = _score_fidelity(
            SimpleNamespace(skills_loaded=["navigator"]),
            [],
            "navigator",
            EvalConfig(_env_file=None),
            object(),
            CheckReport(),
        )
    assert len(records) == 1
    assert records[0].success is False
    assert records[0].score == -1
    assert "judge unavailable" in records[0].reason
