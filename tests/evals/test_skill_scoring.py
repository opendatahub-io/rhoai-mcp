"""Judge failures must not turn an incomplete metric set into a pass."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


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
