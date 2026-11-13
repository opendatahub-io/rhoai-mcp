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
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.reporting.models import (
    EnvironmentRecord,
    EvalRecord,
    GitRecord,
    MetricRecord,
)

if TYPE_CHECKING:
    from evals.config import EvalConfig
    from evals.lcs_client import LCSResult

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


def metric_records_from(eval_result: Any) -> tuple[list[MetricRecord], bool]:
    """Extract metric records and whether every one of them succeeded.

    A metric that errored has no usable score. It is recorded with
    ``success=False`` and its error in ``reason`` rather than being silently
    scored ``0.0``, because a judge error is an error, not a zero (plan §3.3).
    """
    records: list[MetricRecord] = []
    all_passed = True
    for test_result in eval_result.test_results:
        for md in test_result.metrics_data:
            errored = getattr(md, "error", None)
            details: dict[str, Any] = {}
            if not errored and md.reason:
                try:
                    parsed = json.loads(md.reason)
                    if isinstance(parsed, dict) and isinstance(parsed.get("scorecards"), list):
                        details = parsed
                except (ValueError, TypeError):
                    pass  # Historical and fidelity reasons are ordinary prose.
            success = bool(md.success) and not errored
            if not success:
                all_passed = False
            records.append(
                MetricRecord(
                    name=md.name,
                    # score is None on an errored metric; -1.0 marks "no score"
                    # distinguishably from a genuine 0.0.
                    score=float(md.score) if md.score is not None else -1.0,
                    success=success,
                    threshold=float(md.threshold) if md.threshold is not None else 0.0,
                    reason=str(errored) if errored else (details.get("summary") or md.reason or ""),
                    version=str(details.get("metric_version") or ""),
                    scorecards=details.get("scorecards", []),
                )
            )
    return records, all_passed


def overall_passed(
    *, checks_passed: bool, routing_ok: bool, metrics: list[MetricRecord], dry_run: bool,
    required_metrics: set[str] | None = None,
) -> bool:
    """Decide the attempt's verdict (plan §3.5).

    An overall pass requires successful routing, all hard checks, and all
    required metrics passing. Two cases that must **not** pass:

    - a judged attempt with no metric records at all — an empty metric list is
      missing evidence, not success;
    - a dry run, which by definition has no judged evidence.
    """
    if not routing_ok or not checks_passed:
        return False
    if dry_run:
        return False
    if not metrics:
        return False
    if required_metrics is not None and not required_metrics.issubset({m.name for m in metrics}):
        return False
    return all(m.success for m in metrics)


def evaluate_and_record(
    recorder: EvalRecorder,
    scenario: str,
    lcs_result: LCSResult,
    test_cases: list[Any],
    metrics: list[Any],
) -> Any:
    """Wrap deepeval.evaluate() with result recording.

    Calls evaluate(), extracts metric data, builds an EvalRecord, appends
    it to the JSONL file, and returns the EvaluationResult unchanged.
    """
    from datetime import datetime

    from deepeval.evaluate import AsyncConfig, evaluate

    start = time.monotonic()
    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
    )
    duration = time.monotonic() - start

    config = recorder.config
    environment = EnvironmentRecord(
        agent_backend="lcs",
        agent_model="",  # Model is configured in Llama Stack, not eval framework
        lcs_url=config.lcs_url,
        eval_provider=config.eval_provider.value,
        eval_model=config.eval_model,
        cluster_mode=config.cluster_mode.value,
        mcp_use_threshold=config.mcp_use_threshold,
        task_completion_threshold=config.task_completion_threshold,
    )

    metric_records, all_passed = metric_records_from(eval_result)

    record = EvalRecord(
        run_id=recorder.run_id,
        timestamp=datetime.now(UTC).isoformat(),
        scenario=scenario,
        git=recorder.git,
        environment=environment,
        metrics=metric_records,
        turns=lcs_result.turns,
        tool_names_used=lcs_result.tool_names_used,
        passed=all_passed,
        duration_seconds=round(duration, 2),
    )

    recorder.write(record)
    return eval_result
