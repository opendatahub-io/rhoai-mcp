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
) -> AttemptOutcome:
    """Execute, persist, score and record one golden/model attempt."""
    attempt_id = uuid.uuid4().hex[:12]
    root = results_root or RESULTS_ROOT
    dry_run = judge is None

    agent_started = time.monotonic()
    result = await run_conversation(
        spec, golden.script, run_id=recorder.run_id, attempt_id=attempt_id
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
            **spec.describe(),
        }
    )
    artifact_path = artifact.save(root)

    # -- re-read it: the artifact is the scoring input, live too ------------- #
    from evals.agent.transcript import AttemptArtifact

    transcript = AttemptArtifact.load(artifact_path).transcript()

    report = golden.grade(transcript)
    routing_ok = golden.skill in transcript.skills_loaded

    outcome = result.outcome
    if outcome == OUTCOME_EVALUATED and not routing_ok:
        outcome = OUTCOME_ROUTING_FAILURE

    metrics: list[MetricRecord] = []
    scoring_seconds = 0.0
    if judge is not None:
        metrics, scoring_seconds = _score(transcript, artifact_path, golden, config, judge, report)

    passed = overall_passed(
        checks_passed=report.passed,
        routing_ok=routing_ok,
        metrics=metrics,
        dry_run=dry_run,
    )

    outside = tools_outside_allowlist(transcript, golden.allowed_tools())

    record = EvalRecord(
        run_id=recorder.run_id,
        timestamp=datetime.now(UTC).isoformat(),
        scenario=golden.id,
        git=recorder.git,
        environment=EnvironmentRecord(
            agent_backend="pydantic-ai",
            agent_model=spec.id,
            lcs_url="",
            eval_provider=config.eval_provider.value,
            eval_model=config.eval_model if judge is not None else "",
            cluster_mode=config.cluster_mode.value,
            mcp_use_threshold=config.mcp_use_threshold,
            task_completion_threshold=config.task_completion_threshold,
        ),
        metrics=metrics,
        turns=result.turns_completed,
        tool_names_used=transcript.tool_names_used,
        passed=passed,
        duration_seconds=round(scoring_seconds, 2),
        attempt_id=attempt_id,
        scoring_id=uuid.uuid4().hex[:12],
        skill=golden.skill,
        golden_id=golden.id,
        skills_loaded=transcript.skills_loaded,
        outcome=outcome,
        error_stage=result.error_stage or "",
        error_detail=result.error or "",
        checks=[CheckRecord(**c) for c in report.to_list()],
        artifact_path=str(artifact_path),
        dry_run=dry_run,
        agent_seconds=round(agent_seconds, 2),
        skill_sha=golden.skill_sha(),
        scenario_sha=golden.scenario_sha(),
        rubric_version=RUBRIC_VERSION,
    )
    recorder.write(record)

    return AttemptOutcome(
        golden=golden,
        spec_id=spec.id,
        attempt_id=attempt_id,
        outcome=outcome,
        passed=passed,
        report=report,
        transcript=transcript,
        artifact_path=artifact_path,
        metrics=metrics,
        routing_ok=routing_ok,
        dry_run=dry_run,
        error=result.error,
        error_stage=result.error_stage,
        tools_outside_allowlist=outside,
    )


def _score(
    transcript: AgentTranscript,
    artifact_path: Path,
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
    return records, time.monotonic() - started
