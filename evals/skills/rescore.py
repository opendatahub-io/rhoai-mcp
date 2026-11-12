"""Re-evaluate saved executions without starting an agent or MCP server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from evals.agent.transcript import AttemptArtifact
from evals.reporting.recorder import EvalRecorder
from evals.skills.attempt import RESULTS_ROOT, SKILL_RESULTS_PATH, AttemptOutcome, score_artifact
from evals.skills.config import SkillEvalConfig
from evals.skills.goldens import GOLDENS_BY_ID


def find_artifacts(root: Path, *, run_id: str | None, attempt_id: str | None) -> list[Path]:
    """Select artifacts by explicit run or attempt prefix."""
    return sorted(
        p.parent
        for p in root.glob("*/*/artifact.json")
        if (run_id is None or p.parent.parent.name.startswith(run_id))
        and (attempt_id is None or p.parent.name.startswith(attempt_id))
    )


def rescore(
    paths: list[Path], *, judge: Any = None, recorder: Any = None, config: Any = None
) -> list[AttemptOutcome]:
    cfg = config or SkillEvalConfig()
    rec = recorder or EvalRecorder(cfg, path=SKILL_RESULTS_PATH)
    outcomes = []
    for path in paths:
        artifact = AttemptArtifact.load(path)
        golden_id = artifact.metadata.get("golden_id", "")
        golden = GOLDENS_BY_ID.get(golden_id)
        if golden is None:
            raise ValueError(f"{path}: unknown scenario {golden_id!r}")
        outcomes.append(score_artifact(path, golden, config=cfg, recorder=rec, judge=judge))
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--run", dest="run_id", help="run id (prefix)")
    selector.add_argument("--attempt", dest="attempt_id", help="attempt id (unique prefix)")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--judge", action="store_true", help="run the configured judge")
    args = parser.parse_args(argv)
    paths = find_artifacts(args.results_root, run_id=args.run_id, attempt_id=args.attempt_id)
    if not paths or (args.attempt_id and len(paths) != 1):
        print(
            "No unique matching artifact found." if args.attempt_id else "No artifacts matched.",
            file=sys.stderr,
        )
        return 1
    cfg = SkillEvalConfig()
    judge = None
    if args.judge:
        from evals.providers import create_judge_llm

        judge = create_judge_llm(cfg)
    try:
        outcomes = rescore(paths, judge=judge, config=cfg)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Cannot re-evaluate artifact: {exc}", file=sys.stderr)
        return 1
    return int(
        any(
            not o.passed if judge else not (o.eligible and o.routing_ok and o.report.passed)
            for o in outcomes
        )
    )


if __name__ == "__main__":
    sys.exit(main())
