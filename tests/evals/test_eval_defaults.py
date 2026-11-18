"""Verify DeepEval timeout defaults in fresh evaluation processes."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "startup",
    [
        "import evals.pytest_defaults; ",
        "import evals.skills.rescore as r; r.main(['--attempt', 'no-such-attempt']); ",
        "import evals.skills.rescore as r; r.rescore([]); ",
    ],
)
@pytest.mark.parametrize(
    ("overrides", "expected_attempt", "expected_task"),
    [
        ({}, 900.0, None),
        ({"DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE": "42"}, 42.0, None),
        ({"DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS": "42"}, 42.0, None),
        ({"DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE": "200"}, None, 200.0),
    ],
)
def test_deepeval_timeout_defaults(
    startup: str,
    overrides: dict[str, str],
    expected_attempt: float | None,
    expected_task: float | None,
) -> None:
    pytest.importorskip("deepeval")
    env = os.environ.copy()
    for name in (
        "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
        "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS",
        "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
        "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS",
    ):
        env.pop(name, None)
    env.update(overrides)
    code = (
        startup + "from deepeval.config.settings import get_settings; "
        "s = get_settings(); "
        "print(s.DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS, s.DEEPEVAL_PER_TASK_TIMEOUT_SECONDS)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )
    attempt, task = map(float, result.stdout.split())
    if expected_attempt is not None:
        assert attempt == expected_attempt
    if expected_task is not None:
        assert task == expected_task
        assert attempt < task


def test_importing_evals_and_rescore_does_not_change_timeout() -> None:
    env = os.environ.copy()
    env.pop("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", None)
    code = (
        "import os, evals, evals.skills.rescore; "
        "print(os.environ.get('DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE', 'unset'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "unset"
