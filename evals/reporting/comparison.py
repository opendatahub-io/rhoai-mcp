"""Provider comparison report for eval results."""

from __future__ import annotations

from collections import defaultdict

from evals.reporting.formatting import format_table, truncate
from evals.reporting.models import EvalRecord


def provider_comparison_report(
    records: list[EvalRecord],
    scenario: str | None = None,
    last_n: int = 10,
    fmt: str = "terminal",
) -> str:
    """Compare eval scores across providers/models.

    Groups records by (llm_provider, llm_model), computes average score
    per metric, and renders a comparison table.

    Args:
        records: All eval records.
        scenario: Filter to a specific scenario (None = all).
        last_n: Use only the last N records per provider group.
        fmt: 'terminal' or 'markdown'.
    """
    if not records:
        return "No eval records found."

    filtered = records
    if scenario:
        filtered = [r for r in filtered if r.scenario == scenario]
    if not filtered:
        return f"No records found for scenario={scenario}"

    # Group by provider/model
    groups: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in filtered:
        key = f"{r.environment.llm_provider}/{r.environment.llm_model}"
        groups[key].append(r)

    # Collect all metric names
    all_metric_names: list[str] = []
    for r in filtered:
        for m in r.metrics:
            if m.name not in all_metric_names:
                all_metric_names.append(m.name)

    headers = ["Provider/Model", *[truncate(n, 20) for n in all_metric_names],
               "Avg Turns", "Pass Rate"]
    alignments = ["l", *["r"] * len(all_metric_names), "r", "r"]

    # Build rows sorted by average score descending
    row_data: list[tuple[float, list[str]]] = []
    for key, group in groups.items():
        recent = sorted(group, key=lambda r: r.timestamp)[-last_n:]
        metric_avgs: dict[str, float] = {}
        for mn in all_metric_names:
            scores = [m.score for r in recent for m in r.metrics if m.name == mn]
            metric_avgs[mn] = sum(scores) / len(scores) if scores else 0.0

        avg_turns = sum(r.turns for r in recent) / len(recent)
        pass_count = sum(1 for r in recent if r.passed)
        pass_rate = f"{pass_count}/{len(recent)}"

        row = [truncate(key, 30)]
        total_score = 0.0
        for mn in all_metric_names:
            avg = metric_avgs[mn]
            total_score += avg
            row.append(f"{avg:.2f}")
        row.append(f"{avg_turns:.1f}")
        row.append(pass_rate)

        row_data.append((total_score, row))

    # Sort by total score descending
    row_data.sort(key=lambda x: x[0], reverse=True)
    rows = [rd[1] for rd in row_data]

    title = "Provider Comparison"
    if scenario:
        title += f" - {scenario}"

    table = format_table(headers, rows, alignments, fmt=fmt)

    if fmt == "markdown":
        return f"## {title}\n\n{table}"
    return f"{title}\n\n{table}"
