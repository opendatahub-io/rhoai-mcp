"""Provider comparison report for eval results."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from evals.reporting.formatting import format_table, provider_label, truncate
from evals.reporting.models import EvalRecord


def provider_comparison_report(
    records: list[EvalRecord],
    scenario: str | None = None,
    last_n: int = 10,
    fmt: str = "terminal",
    *,
    cohort_path: Path | None = None,
    run_ids: list[str] | None = None,
    artifacts_root: Path | None = None,
) -> str:
    """Compare eval scores across agent backends/models.

    Groups records by agent backend label, computes average score
    per metric, and renders a comparison table.

    Args:
        records: All eval records.
        scenario: Filter to a specific scenario (None = all).
        last_n: Use only the last N records per provider group.
        fmt: 'terminal' or 'markdown'.
    """
    if cohort_path or run_ids:
        from evals.reporting.skill_comparison import (
            build_skill_comparison,
            load_cohort,
            render_skill_comparison,
        )

        return render_skill_comparison(
            build_skill_comparison(
                records,
                cohort=load_cohort(cohort_path) if cohort_path else None,
                run_ids=run_ids,
                scenario=scenario,
                last_n=last_n,
                artifacts_root=artifacts_root,
            ),
            fmt=fmt,
        )
    if any(r.skill for r in records):
        raise ValueError("Skill comparisons require an explicit --cohort or --run-id selection")
    if fmt == "json":
        raise ValueError("JSON comparison output requires a skill cohort or run IDs")
    if last_n < 1:
        raise ValueError("--last must be positive")
    if not records:
        return "No eval records found."

    filtered = records
    if scenario:
        filtered = [r for r in filtered if r.scenario == scenario]
    if not filtered:
        return f"No records found for scenario={scenario}"

    # Group by agent backend label
    groups: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in filtered:
        key = provider_label(r)
        groups[key].append(r)

    # Collect all metric names
    all_metric_names: list[str] = []
    for r in filtered:
        for m in r.metrics:
            if m.name not in all_metric_names:
                all_metric_names.append(m.name)

    headers = [
        "Agent Backend",
        *[truncate(n, 20) for n in all_metric_names],
        "Avg Turns",
        "Pass Rate",
    ]
    alignments = ["l", *["r"] * len(all_metric_names), "r", "r"]

    # Build rows sorted by average score descending
    row_data: list[tuple[float, list[str]]] = []
    for key, group in groups.items():
        recent = sorted(group, key=lambda r: r.timestamp)[-last_n:]
        metric_avgs: dict[str, float | None] = {}
        for mn in all_metric_names:
            scores = [
                m.score
                for r in recent
                for m in r.metrics
                if m.name == mn and math.isfinite(m.score) and 0 <= m.score <= 1
            ]
            metric_avgs[mn] = sum(scores) / len(scores) if scores else None

        avg_turns = sum(r.turns for r in recent) / len(recent)
        pass_count = sum(1 for r in recent if r.passed)
        pass_rate = f"{pass_count}/{len(recent)}"

        row = [truncate(key, 30)]
        total_score = 0.0
        for mn in all_metric_names:
            avg = metric_avgs[mn]
            if avg is not None:
                total_score += avg
            row.append(f"{avg:.2f}" if avg is not None else "n/a")
        row.append(f"{avg_turns:.1f}")
        row.append(pass_rate)

        row_data.append((total_score, row))

    # Sort by total score descending
    row_data.sort(key=lambda x: x[0], reverse=True)
    rows = [rd[1] for rd in row_data]

    title = "Agent Backend Comparison"
    if scenario:
        title += f" - {scenario}"

    table = format_table(headers, rows, alignments, fmt=fmt)

    if fmt == "markdown":
        return f"## {title}\n\n{table}"
    return f"{title}\n\n{table}"
