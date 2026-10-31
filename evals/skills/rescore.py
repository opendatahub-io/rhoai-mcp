"""Re-score saved attempt artifacts without running an agent.

Re-scoring is the same path as a live run minus the agent (plan §3.6), which is
what makes rubric calibration and checker work a **controlled** comparison: the
transcript is frozen, so a score that moves did so because the rubric or the
check changed, not because a non-deterministic model behaved differently.

That matters more here than usual. Nothing in this matrix is deterministic —
the gpt-5.6 family rejects ``temperature`` outright — so re-running the agent to
test a rubric change compares two different conversations and tells you nothing.

Each re-score reuses the artifact's ``attempt_id`` and mints a new
``scoring_id``, so repeated scorings stay linked to the one agent run.

Usage::

    # Everything from a run, deterministic checks only (free)
    uv run --group skill-eval python -m evals.skills.rescore --run <run_id>

    # One attempt, with the judge
    uv run --group skill-eval python -m evals.skills.rescore --attempt <id> --judge

    # Every saved attempt of one golden, judged (rubric calibration)
    uv run --group skill-eval python -m evals.skills.rescore \\
        --golden navigator-happy-path --judge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.agent.transcript import AttemptArtifact
from evals.skills.attempt import RESULTS_ROOT, score_artifact
from evals.skills.goldens import GOLDENS_BY_ID

if TYPE_CHECKING:
    from evals.skills.attempt import AttemptOutcome


def find_artifacts(
    root: Path | None = None,
    *,
    run_id: str | None = None,
    attempt_id: str | None = None,
    golden_id: str | None = None,
    skill: str | None = None,
) -> list[Path]:
    """Locate saved artifact directories matching the given filters.

    ``run_id`` and ``attempt_id`` match by prefix, so a short id is enough.

    Returns:
        Matching directories, sorted by creation time (oldest first).
    """
    base = root or RESULTS_ROOT
    if not base.exists():
        return []

    found: list[tuple[str, Path]] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_id and not run_dir.name.startswith(run_id):
            continue
        for attempt_dir in sorted(run_dir.iterdir()):
            if not (attempt_dir / "artifact.json").exists():
                continue
            if attempt_id and not attempt_dir.name.startswith(attempt_id):
                continue
            try:
                artifact = AttemptArtifact.load(attempt_dir)
            except (ValueError, KeyError, OSError):
                # A stale-schema or corrupt artifact is skipped, not migrated.
                continue
            meta = artifact.metadata
            if golden_id and meta.get("golden_id") != golden_id:
                continue
            if skill and meta.get("skill") != skill:
                continue
            found.append((artifact.created_at, attempt_dir))

    return [path for _, path in sorted(found)]


def render_listing(paths: list[Path]) -> str:
    """Tabulate matching artifacts so ids can be copied into other commands.

    Shared by ``rescore --list`` and ``show --list`` so the two cannot drift.
    """
    header = f"{'RUN':<14} {'ATTEMPT':<14} {'GOLDEN':<34}{'MODEL':<17}CREATED"
    lines = [header, "-" * len(header)]
    for path in paths:
        artifact = AttemptArtifact.load(path)
        meta = artifact.metadata
        lines.append(
            f"{path.parent.name:<14} {path.name:<14} "
            f"{str(meta.get('golden_id') or '-'):<34}"
            f"{str(meta.get('id') or meta.get('spec_id') or '-'):<17}"
            f"{artifact.created_at[:19]}"
        )
    lines.append("")
    lines.append(
        f"{len(paths)} attempt(s). Pass an id with --attempt (prefix match is enough), "
        'e.g. make eval-show ARGS="--attempt ' + (paths[-1].name[:8] if paths else "<id>") + '"'
    )
    return "\n".join(lines)


def rescore(
    paths: list[Path], *, judge: Any = None, recorder: Any = None, config: Any = None
) -> list[AttemptOutcome]:
    """Re-score each artifact through the live scoring path.

    Artifacts whose golden is no longer defined are skipped with a warning
    rather than guessed at: a scenario that no longer exists cannot be graded
    against checks that have since changed shape.
    """
    from evals.config import EvalConfig
    from evals.reporting.recorder import EvalRecorder

    cfg = config or EvalConfig()
    rec = recorder or EvalRecorder(cfg)

    outcomes: list[AttemptOutcome] = []
    for path in paths:
        artifact = AttemptArtifact.load(path)
        golden_id = artifact.metadata.get("golden_id", "")
        golden = GOLDENS_BY_ID.get(golden_id)
        if golden is None:
            print(f"  skip {path.name}: unknown golden {golden_id!r}", file=sys.stderr)
            continue
        outcomes.append(score_artifact(path, golden, config=cfg, recorder=rec, judge=judge))
    return outcomes


def render(outcomes: list[AttemptOutcome]) -> str:
    """One line per attempt, plus failures and metric scores."""
    if not outcomes:
        return "no artifacts matched"

    lines: list[str] = []
    for outcome in outcomes:
        verdict = "PASS" if outcome.passed else ("DRY" if outcome.dry_run else "FAIL")
        lines.append(
            f"[{verdict:<4}] {outcome.golden.id:<34} {outcome.spec_id:<16} "
            f"attempt={outcome.attempt_id}  checks: {outcome.report.summary()}"
        )
        for check in outcome.report.failures:
            lines.append(f"         FAIL {check.name}: {check.detail[:120]}")
        for metric in outcome.metrics:
            flag = "ok " if metric.success else "FAIL"
            score = "n/a" if metric.score < 0 else f"{metric.score:.3f}"
            lines.append(f"         {flag} {metric.name}: {score} (thr {metric.threshold})")

    judged = [o for o in outcomes if not o.dry_run]
    lines.append("")
    lines.append(
        f"{len(outcomes)} attempt(s) re-scored; {len(judged)} judged, "
        f"{len(outcomes) - len(judged)} dry"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", dest="run_id", help="run id (prefix match)")
    parser.add_argument("--attempt", dest="attempt_id", help="attempt id (prefix match)")
    parser.add_argument("--golden", dest="golden_id", help="only this golden")
    parser.add_argument("--skill", help="only this skill")
    parser.add_argument("--results-root", default=None)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="run judged metrics (costs money); omit for free deterministic checks",
    )
    parser.add_argument("--list", action="store_true", help="list matches and exit")
    args = parser.parse_args(argv)

    root = Path(args.results_root) if args.results_root else None
    paths = find_artifacts(
        root,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        golden_id=args.golden_id,
        skill=args.skill,
    )

    if not paths:
        print("no artifacts matched", file=sys.stderr)
        return 1

    if args.list:
        print(render_listing(paths))
        return 0

    judge = None
    if args.judge:
        from evals.config import EvalConfig
        from evals.providers import create_judge_llm

        judge = create_judge_llm(EvalConfig())

    outcomes = rescore(paths, judge=judge)
    print(render(outcomes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
