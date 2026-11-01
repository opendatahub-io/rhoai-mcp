"""Run one golden against one model, score it, and record it.

The order here is the point (plan §3.5, §3.6):

1. run the agent;
2. **write the artifact** — including for a partial or failed attempt;
3. **load it back** and adapt — so live scoring and re-scoring share one path
   and a lossy artifact fails the live run rather than being discovered later;
4. run the deterministic checks;
5. optionally run judged metrics;
6. **write the record**, then let the caller assert.

Nothing raises for a model or check failure: those are outcomes, recorded and
returned. Only the caller's assertion fails the test, and it runs last so a
failure can never erase the attempt from the results.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.agent.runner import (
    OUTCOME_EVALUATED,
    OUTCOME_ROUTING_FAILURE,
    REPO_ROOT,
    run_conversation,
)
from evals.reporting.models import (
    CheckRecord,
    EnvironmentRecord,
    EvalRecord,
    MetricRecord,
)
from evals.reporting.recorder import metric_records_from, overall_passed
from evals.skills.checks import CheckReport, tools_outside_allowlist
from evals.skills.goldens import RUBRIC_VERSION, Golden

if TYPE_CHECKING:
    from evals.agent.models import AgentModelSpec
    from evals.agent.transcript import AgentTranscript
    from evals.config import EvalConfig
    from evals.reporting.recorder import EvalRecorder

RESULTS_ROOT = REPO_ROOT / "evals" / "results"


@dataclass
class AttemptOutcome:
    """Everything a test needs to assert on, after recording has happened."""

    golden: Golden
    spec_id: str
    attempt_id: str
    outcome: str
    passed: bool
    report: CheckReport
    transcript: AgentTranscript
    artifact_path: Path
    metrics: list[MetricRecord] = field(default_factory=list)
    routing_ok: bool = False
    dry_run: bool = True
    error: str | None = None
    error_stage: str | None = None
    tools_outside_allowlist: list[str] = field(default_factory=list)

    def failure_summary(self) -> str:
        """A assertion message that names what actually went wrong."""
        lines = [
            f"{self.golden.id} on {self.spec_id}: outcome={self.outcome}",
            f"artifact: {self.artifact_path}",
        ]
        if self.error:
            lines.append(f"error[{self.error_stage}]: {self.error}")
        for check in self.report.failures:
            lines.append(f"  FAIL {check.name}: {check.detail}")
        for metric in self.metrics:
            if not metric.success:
                lines.append(f"  METRIC {metric.name}={metric.score}: {metric.reason[:200]}")
        if self.report.unexercised:
            names = ", ".join(c.name for c in self.report.unexercised)
            lines.append(f"  (unexercised: {names})")
        return "\n".join(lines)


async def run_attempt(
    golden: Golden,
    spec: AgentModelSpec,
    *,
    recorder: EvalRecorder,
    config: EvalConfig,
    judge: Any = None,
    results_root: Path | None = None,
    tracing: Any = None,
) -> AttemptOutcome:
    """Execute, persist, score and record one golden/model attempt.

    When ``tracing`` is an enabled session, the whole conversation runs inside a
    single parent span so its turns form one trace, and the resulting trace id
    is stored on the record and artifact for correlation (plan §3.6).
    """
    from evals.agent.tracing import TracingSession, instrumentation

    attempt_id = uuid.uuid4().hex[:12]
    root = results_root or RESULTS_ROOT
    session = tracing if tracing is not None else TracingSession()

    agent_started = time.monotonic()
    # Each scripted turn becomes its own trace, grouped into one MLflow session
    # by attempt id; the runner owns those spans since it owns the turn loop.
    result = await run_conversation(
        spec,
        golden.script,
        run_id=recorder.run_id,
        attempt_id=attempt_id,
        extra_capabilities=instrumentation(session),
        tracing=session,
        golden_id=golden.id,
    )
    agent_seconds = time.monotonic() - agent_started

    # -- persist first, so nothing below can lose the attempt ---------------- #
    artifact = result.artifact(
        metadata={
            "golden_id": golden.id,
            "skill": golden.skill,
            "scenario_sha": golden.scenario_sha(),
            "skill_sha": golden.skill_sha(),
            "rubric_version": RUBRIC_VERSION,
            "trace_session_id": result.trace_session_id,
            **spec.describe(),
        }
    )
    artifact_path = artifact.save(root)

    # -- re-read it: the artifact is the scoring input, live too ------------- #
    return score_artifact(
        artifact_path,
        golden,
        config=config,
        recorder=recorder,
        judge=judge,
        spec_id=spec.id,
        run_outcome=result.outcome,
        run_error=result.error,
        run_error_stage=result.error_stage,
        turns=result.turns_completed,
        agent_seconds=agent_seconds,
        trace_id=result.trace_session_id,
    )


def score_artifact(
    artifact_path: Path,
    golden: Golden,
    *,
    config: EvalConfig,
    recorder: EvalRecorder,
    judge: Any = None,
    spec_id: str | None = None,
    run_outcome: str | None = None,
    run_error: str | None = None,
    run_error_stage: str | None = None,
    turns: int | None = None,
    agent_seconds: float = 0.0,
    trace_id: str | None = None,
) -> AttemptOutcome:
    """Grade and record a saved attempt.

    **This is the one scoring path.** A live run calls it after writing its
    artifact; ``eval-rescore`` calls it on an artifact written minutes or weeks
    ago. Keeping both on this function is what stops live scoring and re-scoring
    from drifting apart (plan §3.6).

    A re-score reuses the artifact's ``attempt_id`` and mints a fresh
    ``scoring_id``, so repeated scorings of one agent run stay distinguishable
    while remaining linked to it (plan §3.5).

    Args:
        artifact_path: Directory written by :meth:`AttemptArtifact.save`.
        golden: The scenario definition supplying checks and metadata.
        config: Eval configuration (thresholds, judge settings).
        recorder: Destination for the JSONL record.
        judge: Judge LLM, or ``None`` for a dry run.
        spec_id: Agent model id. Read from artifact metadata when omitted.
        run_outcome: Live-run outcome. Read from metadata when omitted, since a
            re-score did not run an agent and cannot observe one.
        run_error: Live-run error text, likewise.
        run_error_stage: Live-run error stage, likewise.
        turns: Turns completed, likewise.
        agent_seconds: Agent execution time; zero for a re-score, which ran none.
    """
    from evals.agent.transcript import AttemptArtifact

    artifact = AttemptArtifact.load(artifact_path)
    meta = artifact.metadata
    transcript = artifact.transcript()
    dry_run = judge is None

    report = golden.grade(transcript)
    routing_ok = golden.skill in transcript.skills_loaded

    outcome = run_outcome if run_outcome is not None else meta.get("outcome", OUTCOME_EVALUATED)
    if outcome == OUTCOME_EVALUATED and not routing_ok:
        outcome = OUTCOME_ROUTING_FAILURE

    metrics: list[MetricRecord] = []
    scoring_seconds = 0.0
    if judge is not None:
        metrics, scoring_seconds = _score(
            transcript, artifact_path, golden.skill, config, judge, report
        )

    passed = overall_passed(
        checks_passed=report.passed,
        routing_ok=routing_ok,
        metrics=metrics,
        dry_run=dry_run,
    )

    outside = tools_outside_allowlist(transcript, golden.allowed_tools())
    resolved_spec = spec_id or str(meta.get("id") or meta.get("spec_id") or "unknown")
    error = run_error if run_error is not None else (meta.get("error") or None)
    error_stage = (
        run_error_stage if run_error_stage is not None else (meta.get("error_stage") or None)
    )

    record = EvalRecord(
        run_id=recorder.run_id,
        timestamp=datetime.now(UTC).isoformat(),
        scenario=golden.id,
        git=recorder.git,
        environment=EnvironmentRecord(
            agent_backend="pydantic-ai",
            agent_model=resolved_spec,
            lcs_url="",
            eval_provider=config.eval_provider.value,
            eval_model=config.eval_model if judge is not None else "",
            cluster_mode=config.cluster_mode.value,
            mcp_use_threshold=config.mcp_use_threshold,
            task_completion_threshold=config.task_completion_threshold,
        ),
        metrics=metrics,
        turns=turns if turns is not None else int(meta.get("turns_completed", 0) or 0),
        tool_names_used=transcript.tool_names_used,
        passed=passed,
        duration_seconds=round(scoring_seconds, 2),
        # Reused, not regenerated: this identifies the agent run being scored.
        attempt_id=artifact.attempt_id,
        scoring_id=uuid.uuid4().hex[:12],
        skill=golden.skill,
        golden_id=golden.id,
        skills_loaded=transcript.skills_loaded,
        outcome=outcome,
        error_stage=error_stage or "",
        error_detail=error or "",
        checks=[CheckRecord(**c) for c in report.to_list()],
        artifact_path=str(artifact_path),
        dry_run=dry_run,
        agent_seconds=round(agent_seconds, 2),
        skill_sha=golden.skill_sha(),
        scenario_sha=golden.scenario_sha(),
        rubric_version=RUBRIC_VERSION,
        # A re-score runs no agent and so has no trace of its own; carrying the
        # original attempt's id lets it point back at that trace (plan §3.6).
        trace_id=trace_id if trace_id is not None else str(meta.get("trace_session_id", "")),
    )
    recorder.write(record)

    return AttemptOutcome(
        golden=golden,
        spec_id=resolved_spec,
        attempt_id=artifact.attempt_id,
        outcome=outcome,
        passed=passed,
        report=report,
        transcript=transcript,
        artifact_path=artifact_path,
        metrics=metrics,
        routing_ok=routing_ok,
        dry_run=dry_run,
        error=error,
        error_stage=error_stage,
        tools_outside_allowlist=outside,
    )


def _score(
    transcript: AgentTranscript,
    artifact_path: Path,
    skill: str,
    config: EvalConfig,
    judge: Any,
    report: CheckReport,
) -> tuple[list[MetricRecord], float]:
    """Run the judged MCP metrics against the rebuilt transcript.

    A judge error is recorded as an error, never as a zero score, and a metric
    whose preconditions are absent is recorded ``n/a`` with the specific missing
    precondition rather than silently skipped (plan §3.3).
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig

    from evals.agent.transcript import AttemptArtifact
    from evals.deepeval_helpers import (
        NothingToScoreError,
        build_mcp_server_from_schemas,
        transcript_to_conversational_test_case,
    )
    from evals.metrics.config import (
        create_multi_turn_mcp_use_metric,
        create_task_completion_metric,
    )

    schemas = AttemptArtifact.load(artifact_path).mcp_tools()
    try:
        case = transcript_to_conversational_test_case(
            transcript, build_mcp_server_from_schemas(schemas)
        )
    except NothingToScoreError as exc:
        report.add("judged_metrics", "n/a", str(exc))
        return [], 0.0

    if not transcript.mcp_tool_calls:
        report.add(
            "judged_metrics",
            "n/a",
            "no MCP tool calls in the trajectory; MCP metrics have no input",
        )
        return [], 0.0

    metrics = [
        create_multi_turn_mcp_use_metric(config),
        create_task_completion_metric(config),
    ]
    # Hold the judge fixed across agent models.
    for metric in metrics:
        metric.model = judge

    started = time.monotonic()
    try:
        eval_result = evaluate(
            test_cases=[case], metrics=metrics, async_config=AsyncConfig(run_async=False)
        )
    except Exception as exc:  # noqa: BLE001 - a judge failure is an error, not a 0
        report.add("judged_metrics", "n/a", f"judge failed: {type(exc).__name__}: {exc}")
        return [], time.monotonic() - started

    records, _ = metric_records_from(eval_result)
    records += _score_fidelity(transcript, schemas, skill, config, judge, report)
    return records, time.monotonic() - started


def _score_fidelity(
    transcript: AgentTranscript,
    schemas: list[Any],
    skill: str,
    config: EvalConfig,
    judge: Any,
    report: CheckReport,
) -> list[MetricRecord]:
    """Score skill adherence with the per-skill GEval rubric (plan §3.3).

    Scored on a *separate* test case built with ``include_skill_body=True``: the
    judge cannot assess adherence to a document it has not been shown. The MCP
    metrics deliberately do not get that case, since the skill body is not part
    of the trajectory they score.

    Conditional on routing — with no skill loaded there is nothing to adhere to,
    so the result is ``n/a`` with a reason, never 0.0.
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig

    from evals.deepeval_helpers import (
        build_mcp_server_from_schemas,
        transcript_to_conversational_test_case,
    )
    from evals.skills.rubrics import create_fidelity_metric

    if skill not in transcript.skills_loaded:
        report.add(
            "skill_fidelity",
            "n/a",
            f"{skill!r} was never loaded; adherence to it is unscoreable",
        )
        return []

    metric = create_fidelity_metric(skill, config, model=judge)
    if metric is None:
        report.add("skill_fidelity", "n/a", f"no fidelity rubric defined for {skill!r}")
        return []

    case = transcript_to_conversational_test_case(
        transcript,
        build_mcp_server_from_schemas(schemas),
        include_skill_body=True,
        skill=skill,
        annotate_tool_calls=True,
    )
    try:
        result = evaluate(
            test_cases=[case], metrics=[metric], async_config=AsyncConfig(run_async=False)
        )
    except Exception as exc:  # noqa: BLE001
        report.add("skill_fidelity", "n/a", f"rubric failed: {type(exc).__name__}: {exc}")
        return []

    records, _ = metric_records_from(result)
    return records
