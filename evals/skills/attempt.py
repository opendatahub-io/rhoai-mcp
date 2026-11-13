"""Run one golden against one model, score it, and record it.

The order here is the point:

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
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.agent.runner import (
    OUTCOME_EVALUATED,
    OUTCOME_ROUTING_FAILURE,
    REPO_ROOT,
    run_conversation,
)
from evals.config import ClusterMode
from evals.reporting.models import (
    CheckRecord,
    EnvironmentRecord,
    EvalRecord,
    MetricRecord,
)
from evals.reporting.recorder import metric_records_from, overall_passed
from evals.skills.checks import CheckReport
from evals.skills.goldens import RUBRIC_VERSION, Golden

if TYPE_CHECKING:
    from evals.agent.models import AgentModelSpec
    from evals.agent.transcript import AgentTranscript
    from evals.config import EvalConfig
    from evals.reporting.recorder import EvalRecorder

RESULTS_ROOT = REPO_ROOT / "evals" / "results"
SKILL_RESULTS_PATH = RESULTS_ROOT / "skill_history.jsonl"
CHECKS_VERSION = "v2-gpu-discovery"


@dataclass
class AttemptOutcome:
    """Everything a test needs to assert on, after recording has happened."""

    golden: Golden
    spec_id: str
    attempt_id: str
    scoring_id: str
    outcome: str
    passed: bool
    report: CheckReport
    transcript: AgentTranscript
    artifact_path: Path
    metrics: list[MetricRecord] = field(default_factory=list)
    routing_ok: bool = False
    eligible: bool = False
    dry_run: bool = True
    error: str | None = None
    error_stage: str | None = None

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
) -> AttemptOutcome:
    """Execute, persist, score and record one golden/model attempt."""

    attempt_id = uuid.uuid4().hex[:12]
    root = results_root or RESULTS_ROOT

    agent_started = time.monotonic()
    result = await run_conversation(
        spec,
        golden.script,
        run_id=recorder.run_id,
        attempt_id=attempt_id,
        mock_cluster=config.cluster_mode == ClusterMode.MOCK,
    )
    agent_seconds = time.monotonic() - agent_started

    # -- persist first, so nothing below can lose the attempt ---------------- #
    artifact = result.artifact(
        metadata={
            "golden_id": golden.id,
            "skill": golden.skill,
            "scenario_sha": golden.scenario_sha(),
            "scenario_snapshot": golden.snapshot(),
            "generation_git": asdict(recorder.git),
            "selection_expectation": asdict(golden.expectation)
            if golden.expectation is not None
            else None,
            "skill_sha": golden.skill_sha(),
            "rubric_version": RUBRIC_VERSION,
            **spec.describe(),
            "cluster_mode": config.cluster_mode.value,
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
) -> AttemptOutcome:
    """Grade and record a saved attempt.

    **This is the one scoring path.** A live run calls it after writing its
    artifact; ``eval-rescore`` calls it on an artifact written minutes or weeks
    ago. Keeping both on this function is what stops live scoring and re-scoring
    from drifting apart.

    A re-score reuses the artifact's ``attempt_id`` and mints a fresh
    ``scoring_id``, so repeated scorings of one agent run stay distinguishable
    while remaining linked to it.

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
    snapshot = meta.get("scenario_snapshot")
    if snapshot is not None:
        saved = Golden.from_snapshot(snapshot)
        if (
            saved.id != golden.id
            or saved.skill != golden.skill
            or saved.scenario_sha() != meta.get("scenario_sha")
        ):
            raise ValueError("saved scenario snapshot does not match artifact identity or hash")
        golden = saved
    elif not meta.get("scenario_sha") or meta["scenario_sha"] != golden.scenario_sha():
        raise ValueError(
            "incompatible artifact: missing scenario snapshot and scenario hash does not match"
        )
    transcript = artifact.transcript()
    dry_run = judge is None

    report = golden.grade(transcript)
    routing_ok = golden.skill in transcript.skills_loaded

    outcome = run_outcome if run_outcome is not None else meta.get("outcome", "unknown")
    if outcome == OUTCOME_EVALUATED and not routing_ok:
        outcome = OUTCOME_ROUTING_FAILURE

    metrics: list[MetricRecord] = []
    scoring_seconds = 0.0
    if judge is not None:
        metrics, scoring_seconds = _score(
            transcript,
            artifact_path,
            golden.skill,
            config,
            judge,
            report,
            scenario=golden.description,
        )

    error = run_error if run_error is not None else (meta.get("error") or None)
    from evals.skills.rubrics import NAVIGATOR_DEPLOY_FIDELITY_NAME, NAVIGATOR_FIDELITY_NAME

    required_metrics = {
        "Conversation MCP Use",
        "Conversation MCP Task Completion",
        (NAVIGATOR_FIDELITY_NAME if golden.skill == "navigator" else NAVIGATOR_DEPLOY_FIDELITY_NAME)
        + " [Conversational GEval]",
    }
    eligible = (
        outcome == OUTCOME_EVALUATED
        and not error
        and int(meta.get("turns_completed", 0) or 0) >= len(golden.script)
    )
    passed = overall_passed(
        checks_passed=report.passed,
        routing_ok=routing_ok and eligible,
        metrics=metrics,
        dry_run=dry_run,
        required_metrics=required_metrics,
    )

    resolved_spec = spec_id or str(meta.get("id") or meta.get("spec_id") or "unknown")
    error_stage = (
        run_error_stage if run_error_stage is not None else (meta.get("error_stage") or None)
    )

    scoring_id = uuid.uuid4().hex[:12]
    record = EvalRecord(
        run_id=artifact.run_id,
        timestamp=datetime.now(UTC).isoformat(),
        scenario=golden.id,
        git=recorder.git,
        environment=EnvironmentRecord(
            agent_backend="pydantic-ai",
            agent_model=resolved_spec,
            lcs_url="",
            eval_provider=config.eval_provider.value,
            eval_model=config.eval_model if judge is not None else "",
            cluster_mode=meta.get("cluster_mode", "unknown"),
            mcp_use_threshold=config.mcp_use_threshold,
            task_completion_threshold=config.task_completion_threshold,
            eval_model_base_url=config.judge_base_url or "",
        ),
        metrics=metrics,
        turns=turns if turns is not None else int(meta.get("turns_completed", 0) or 0),
        tool_names_used=transcript.tool_names_used,
        passed=passed,
        duration_seconds=round(scoring_seconds, 2),
        # Reused, not regenerated: this identifies the agent run being scored.
        attempt_id=artifact.attempt_id,
        scoring_id=scoring_id,
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
        skill_sha=str(meta.get("skill_sha", "")),
        scenario_sha=str(meta.get("scenario_sha", "")),
        rubric_version=RUBRIC_VERSION,
        checks_version=CHECKS_VERSION,
        generation_git=meta.get("generation_git", {}),
    )
    recorder.write(record)

    result = AttemptOutcome(
        golden=golden,
        spec_id=resolved_spec,
        attempt_id=artifact.attempt_id,
        scoring_id=scoring_id,
        outcome=outcome,
        passed=passed,
        report=report,
        transcript=transcript,
        artifact_path=artifact_path,
        metrics=metrics,
        routing_ok=routing_ok,
        eligible=eligible,
        dry_run=dry_run,
        error=error,
        error_stage=error_stage,
    )
    print(render_outcome(result))
    return result


def render_outcome(outcome: AttemptOutcome) -> str:
    """Render the same concise result for live runs and replay."""
    status = "CHECKS ONLY" if outcome.dry_run else ("PASS" if outcome.passed else "FAIL")
    scores = {m.name: f"{m.score:.4f}" if m.score >= 0 else "ERROR" for m in outcome.metrics}
    use = scores.get("Conversation MCP Use", "-")
    complete = scores.get("Conversation MCP Task Completion", "-")
    fidelity = next((value for key, value in scores.items() if "Skill Fidelity" in key), "-")
    checks = "pass" if outcome.report.passed else "fail"
    lines = [
        "SCENARIO  AGENT  STATUS  CHECKS  USE  COMPLETE  FIDELITY",
        f"{outcome.golden.id}  {outcome.spec_id}  {status}  {checks}  {use}  {complete}  {fidelity}",
    ]
    lines.append(
        f"attempt={outcome.attempt_id} scoring={outcome.scoring_id} artifact={outcome.artifact_path}"
    )
    if outcome.error:
        lines.append(f"ERROR [{outcome.error_stage}]: {outcome.error}")
    for check in (
        outcome.report.failures + outcome.report.unexercised + outcome.report.not_applicable
    ):
        lines.append(f"{check.status.upper()} {check.name}: {check.detail}")
    for metric in outcome.metrics:
        if not metric.success:
            lines.append(f"METRIC {metric.name}: {metric.reason}")
    return "\n".join(lines)


def _score(
    transcript: AgentTranscript,
    artifact_path: Path,
    skill: str,
    config: EvalConfig,
    judge: Any,
    report: CheckReport,
    *,
    scenario: str = "",
) -> tuple[list[MetricRecord], float]:
    """Run the judged MCP metrics against the rebuilt transcript.

    A judge error is recorded as an error, never as a zero score, and a metric
    whose preconditions are absent is recorded ``n/a`` with the specific missing
    precondition rather than silently skipped.
    """
    from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, evaluate

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
    from evals.metrics.conversation import CONVERSATION_METRIC_VERSION

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

    context = f"{scenario}\n\n{transcript.skill_body(skill) or ''}"
    metrics = [
        create_multi_turn_mcp_use_metric(config, model=judge, context=context),
        create_task_completion_metric(config, model=judge, context=context),
    ]

    started = time.monotonic()
    try:
        eval_result = evaluate(
            test_cases=[case],
            metrics=metrics,
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(show_indicator=False, print_results=False),
            # Metric names are stable across versions; do not reuse old scores.
            cache_config=CacheConfig(use_cache=False, write_cache=False),
        )
    except Exception as exc:  # noqa: BLE001 - a judge failure is an error, not a 0
        report.add("judged_metrics", "n/a", f"judge failed: {type(exc).__name__}: {exc}")
        return [
            MetricRecord(
                name=metric.__name__,
                score=-1.0,
                success=False,
                threshold=float(metric.threshold or 0.0),
                reason=f"{type(exc).__name__}: {exc}",
                version=CONVERSATION_METRIC_VERSION,
            )
            for metric in metrics
        ], time.monotonic() - started

    records, _ = metric_records_from(eval_result)
    for record in records:
        record.version = CONVERSATION_METRIC_VERSION
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
    """Score skill adherence with the per-skill GEval rubric.

    Scored on a *separate* test case built with ``include_skill_body=True``: the
    judge cannot assess adherence to a document it has not been shown. The MCP
    metrics receive the same document as workflow context when building their
    conversation-wide task, rather than as an extra assistant message.

    Conditional on routing — with no skill loaded there is nothing to adhere to,
    so the result is ``n/a`` with a reason, never 0.0.
    """
    from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, evaluate

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
            test_cases=[case],
            metrics=[metric],
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(show_indicator=False, print_results=False),
            cache_config=CacheConfig(use_cache=False, write_cache=False),
        )
    except Exception as exc:  # noqa: BLE001
        report.add("skill_fidelity", "n/a", f"rubric failed: {type(exc).__name__}: {exc}")
        return [
            MetricRecord(
                name=metric.__name__,
                score=-1.0,
                success=False,
                threshold=metric.threshold,
                reason=f"{type(exc).__name__}: {exc}",
                version=RUBRIC_VERSION,
            )
        ]

    records, _ = metric_records_from(result)
    for record in records:
        record.version = RUBRIC_VERSION
    return records
