"""Eval result recording and persistence.

Provides EvalRecorder for session-scoped recording and evaluate_and_record()
as a drop-in wrapper around deepeval.evaluate() that persists results to JSONL.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.reporting.models import (
    EnvironmentRecord,
    EvalRecord,
    GitRecord,
    MetricRecord,
)

if TYPE_CHECKING:
    from evals.agent import AgentResult
    from evals.config import EvalConfig

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "eval_history.jsonl"


def _get_git_info() -> GitRecord:
    """Get current git commit and branch."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
        branch = "unknown"
    return GitRecord(commit=commit, branch=branch)


class EvalRecorder:
    """Session-scoped eval result recorder.

    Holds a stable run_id and config reference for the duration of a pytest
    session. Each call to write() appends one JSONL line.
    """

    def __init__(self, config: EvalConfig, path: Path | None = None) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.config = config
        self.path = path or DEFAULT_RESULTS_PATH
        self._git = _get_git_info()

    def write(self, record: EvalRecord) -> None:
        """Append a single eval record as one JSONL line."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        logger.info(f"Recorded eval result for scenario={record.scenario} to {self.path}")

    @property
    def git(self) -> GitRecord:
        return self._git


def evaluate_and_record(
    recorder: EvalRecorder,
    scenario: str,
    agent_result: AgentResult,
    test_cases: list[Any],
    metrics: list[Any],
) -> Any:
    """Wrap deepeval.evaluate() with result recording.

    Calls evaluate(), extracts metric data, builds an EvalRecord, appends
    it to the JSONL file, and returns the EvaluationResult unchanged.
    """
    from datetime import datetime, timezone

    from deepeval import evaluate

    start = time.monotonic()
    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        run_async=True,
        print_results=True,
    )
    duration = time.monotonic() - start

    config = recorder.config
    environment = EnvironmentRecord(
        llm_provider=config.llm_provider.value,
        llm_model=config.llm_model,
        eval_provider=config.eval_provider.value,
        eval_model=config.eval_model,
        cluster_mode=config.cluster_mode.value,
        mcp_use_threshold=config.mcp_use_threshold,
        task_completion_threshold=config.task_completion_threshold,
        max_agent_turns=config.max_agent_turns,
    )

    metric_records = []
    all_passed = True
    for test_result in eval_result.test_results:
        for md in test_result.metrics_data:
            success = bool(md.success)
            if not success:
                all_passed = False
            metric_records.append(
                MetricRecord(
                    name=md.metric_name,
                    score=float(md.score),
                    success=success,
                    threshold=float(md.threshold),
                    reason=md.reason or "",
                )
            )

    record = EvalRecord(
        run_id=recorder.run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        scenario=scenario,
        git=recorder.git,
        environment=environment,
        metrics=metric_records,
        turns=agent_result.turns,
        tool_names_used=agent_result.tool_names_used,
        passed=all_passed,
        duration_seconds=round(duration, 2),
    )

    recorder.write(record)
    return eval_result
