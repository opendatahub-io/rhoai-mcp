"""Tests for the eval record write/read path (plan §3.5).

Getting these right is what makes the data trustworthy later: a correct record
can always be re-read, while a lossy one cannot be recovered. Two properties
matter most and both fail silently — new metadata surviving the round-trip, and
an attempt without judged evidence never reading as a pass.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evals.reporting.models import (
    OUTCOME_HARNESS_ERROR,
    OUTCOME_ROUTING_FAILURE,
    CheckRecord,
    EnvironmentRecord,
    EvalRecord,
    GitRecord,
    MetricRecord,
)
from evals.reporting.reader import load_records
from evals.reporting.recorder import metric_records_from, overall_passed


def environment() -> EnvironmentRecord:
    return EnvironmentRecord(
        agent_backend="pydantic-ai",
        agent_model="openai-sol",
        lcs_url="",
        eval_provider="google-vertex",
        eval_model="gemini-2.5-flash",
        cluster_mode="mock",
        mcp_use_threshold=0.5,
        task_completion_threshold=0.6,
    )


def write(tmp_path: Path, *records: EvalRecord) -> Path:
    path = tmp_path / "eval_history.jsonl"
    with path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")
    return path


class TestRecordRoundTrip:
    def test_skill_eval_metadata_survives(self, tmp_path: Path) -> None:
        """reader selects known fields; new ones must be added explicitly."""
        record = EvalRecord(
            run_id="r1",
            timestamp="2026-10-31T12:00:00+00:00",
            scenario="navigator-happy-path",
            git=GitRecord(commit="abc1234", branch="main"),
            environment=environment(),
            attempt_id="att-1",
            scoring_id="sc-1",
            skill="navigator",
            golden_id="navigator-happy-path",
            skills_loaded=["navigator"],
            outcome=OUTCOME_ROUTING_FAILURE,
            error_stage="checkpoint:confirm-workload",
            error_detail="no recommendation table",
            checks=[CheckRecord(name="routing_loaded_expected_skill", status="fail", detail="x")],
            artifact_path="evals/results/r1/att-1",
            trace_id="trace-9",
            dry_run=True,
            agent_seconds=42.5,
            skill_sha="deadbeef",
            scenario_sha="cafe",
            rubric_version="v1",
        )

        (loaded,) = load_records(str(write(tmp_path, record)))

        assert loaded.attempt_id == "att-1"
        assert loaded.scoring_id == "sc-1"
        assert loaded.skill == "navigator"
        assert loaded.golden_id == "navigator-happy-path"
        assert loaded.skills_loaded == ["navigator"]
        assert loaded.outcome == OUTCOME_ROUTING_FAILURE
        assert loaded.error_stage == "checkpoint:confirm-workload"
        assert loaded.artifact_path == "evals/results/r1/att-1"
        assert loaded.trace_id == "trace-9"
        assert loaded.dry_run is True
        assert loaded.agent_seconds == 42.5
        assert loaded.skill_sha == "deadbeef"
        assert loaded.rubric_version == "v1"

    def test_check_results_survive_with_their_status(self, tmp_path: Path) -> None:
        """``unexercised`` must not be flattened into pass/fail on reload."""
        record = EvalRecord(
            run_id="r1",
            timestamp="t",
            scenario="s",
            git=GitRecord(commit="c", branch="b"),
            environment=environment(),
            checks=[
                CheckRecord(name="a", status="pass"),
                CheckRecord(name="b", status="unexercised", detail="branch unreachable"),
                CheckRecord(name="c", status="n/a", detail="precondition absent"),
            ],
        )

        (loaded,) = load_records(str(write(tmp_path, record)))

        assert [(c.name, c.status) for c in loaded.checks] == [
            ("a", "pass"),
            ("b", "unexercised"),
            ("c", "n/a"),
        ]

    def test_historical_lcs_record_without_new_fields_still_loads(self, tmp_path: Path) -> None:
        """Backward compatibility: old lines predate every field above."""
        legacy = {
            "run_id": "old",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "scenario": "cluster_exploration",
            "git": {"commit": "aaa", "branch": "main"},
            "environment": {
                "llm_provider": "lcs",
                "llm_model": "granite",
                "lcs_url": "http://localhost:8443",
                "eval_provider": "google-vertex",
                "eval_model": "gemini-2.5-flash",
                "cluster_mode": "mock",
                "mcp_use_threshold": 0.5,
                "task_completion_threshold": 0.6,
            },
            "metrics": [],
            "turns": 3,
            "passed": True,
        }
        path = tmp_path / "eval_history.jsonl"
        path.write_text(json.dumps(legacy) + "\n")

        (loaded,) = load_records(str(path))

        assert loaded.scenario == "cluster_exploration"
        assert loaded.environment.agent_backend == "lcs"
        assert loaded.attempt_id == ""
        assert loaded.dry_run is False
        assert loaded.checks == []

    def test_malformed_line_is_skipped_without_losing_the_rest(self, tmp_path: Path) -> None:
        good = EvalRecord(
            run_id="r1",
            timestamp="t",
            scenario="s",
            git=GitRecord(commit="c", branch="b"),
            environment=environment(),
        )
        path = write(tmp_path, good)
        with path.open("a") as handle:
            handle.write("{not json\n")

        assert len(load_records(str(path))) == 1


class TestOverallPassed:
    """An attempt must never pass on absent evidence."""

    def _metric(self, success: bool) -> MetricRecord:
        return MetricRecord(
            name="MultiTurnMCPUse", score=0.9, success=success, threshold=0.5, reason=""
        )

    def test_all_green_passes(self) -> None:
        assert overall_passed(
            checks_passed=True,
            routing_ok=True,
            metrics=[self._metric(True)],
            dry_run=False,
        )

    def test_empty_metric_list_does_not_pass(self) -> None:
        """An empty metric list is missing evidence, not success."""
        assert not overall_passed(checks_passed=True, routing_ok=True, metrics=[], dry_run=False)

    def test_dry_run_never_passes(self) -> None:
        assert not overall_passed(checks_passed=True, routing_ok=True, metrics=[], dry_run=True)

    def test_routing_failure_never_passes(self) -> None:
        assert not overall_passed(
            checks_passed=True,
            routing_ok=False,
            metrics=[self._metric(True)],
            dry_run=False,
        )

    def test_failed_hard_check_never_passes(self) -> None:
        assert not overall_passed(
            checks_passed=False,
            routing_ok=True,
            metrics=[self._metric(True)],
            dry_run=False,
        )

    def test_failed_metric_does_not_pass(self) -> None:
        assert not overall_passed(
            checks_passed=True,
            routing_ok=True,
            metrics=[self._metric(False)],
            dry_run=False,
        )


class TestMetricExtraction:
    """A judge error is an error, not a zero score (plan §3.3)."""

    class _MetricData:
        def __init__(self, **kw: Any) -> None:
            self.name = kw.get("name", "m")
            self.score = kw.get("score", 1.0)
            self.success = kw.get("success", True)
            self.threshold = kw.get("threshold", 0.5)
            self.reason = kw.get("reason", "")
            self.error = kw.get("error")

    class _TestResult:
        def __init__(self, metrics_data: list[Any]) -> None:
            self.metrics_data = metrics_data

    class _EvalResult:
        def __init__(self, test_results: list[Any]) -> None:
            self.test_results = test_results

    def test_successful_metric_is_recorded(self) -> None:
        result = self._EvalResult([self._TestResult([self._MetricData(score=0.8)])])

        records, all_passed = metric_records_from(result)

        assert all_passed
        assert records[0].score == 0.8

    def test_errored_metric_is_not_scored_zero(self) -> None:
        """-1.0 marks 'no score', distinguishable from a genuine 0.0."""
        result = self._EvalResult(
            [self._TestResult([self._MetricData(score=None, success=True, error="judge 503")])]
        )

        records, all_passed = metric_records_from(result)

        assert not all_passed
        assert records[0].score == -1.0
        assert "judge 503" in records[0].reason

    def test_errored_metric_does_not_count_as_success(self) -> None:
        result = self._EvalResult(
            [self._TestResult([self._MetricData(success=True, error="boom")])]
        )

        records, _ = metric_records_from(result)

        assert records[0].success is False


class TestOutcomeRecording:
    def test_harness_error_attempt_is_still_written(self, tmp_path: Path) -> None:
        """A failed setup must produce a record, not silence (plan §3.5)."""
        record = EvalRecord(
            run_id="r1",
            timestamp="t",
            scenario="navigator-happy-path",
            git=GitRecord(commit="c", branch="b"),
            environment=environment(),
            outcome=OUTCOME_HARNESS_ERROR,
            error_stage="tool_schema_fetch",
            error_detail="connection refused",
            passed=False,
        )

        (loaded,) = load_records(str(write(tmp_path, record)))

        assert loaded.outcome == OUTCOME_HARNESS_ERROR
        assert loaded.error_stage == "tool_schema_fetch"
        assert loaded.passed is False
