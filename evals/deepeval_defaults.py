"""Defaults applied explicitly by evaluation entry points."""

import os

_TIMEOUT_SETTINGS = (
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS",
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS",
)


def configure_deepeval() -> None:
    """Set the judge timeout unless a per-attempt or outer budget is supplied."""
    if not any(name in os.environ for name in _TIMEOUT_SETTINGS):
        os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "900"
