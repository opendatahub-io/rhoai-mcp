"""Score trending report for eval results."""

from __future__ import annotations

from evals.reporting.formatting import format_table, truncate
from evals.reporting.models import EvalRecord


def score_trend_report(
    records: list[EvalRecord],
    scenario: str | None = None,
    provider: str | None = None,
    last_n: int = 20,
    fmt: str = "terminal",
) -> str:
    """Show chronological score history for a scenario/provider.

    Args:
        records: All eval records.
        scenario: Filter to a specific scenario (None = all).
        provider: Filter to a specific provider (e.g. 'openai/gpt-4o').
        last_n: Show only the last N records.
        fmt: 'terminal' or 'markdown'.
    """
    if not records:
        return "No eval records found."

    filtered = list(records)
    if scenario:
        filtered = [r for r in filtered if r.scenario == scenario]
    if provider:
        filtered = [
            r for r in filtered
            if f"{r.environment.llm_provider}/{r.environment.llm_model}" == provider
        ]
    if not filtered:
        parts = []
        if scenario:
            parts.append(f"scenario={scenario}")
        if provider:
            parts.append(f"provider={provider}")
        return f"No records found for {', '.join(parts)}"

    # Sort by timestamp and take last N
    filtered.sort(key=lambda r: r.timestamp)
    filtered = filtered[-last_n:]

    # Collect all metric names
    all_metric_names: list[str] = []
    for r in filtered:
        for m in r.metrics:
            if m.name not in all_metric_names:
                all_metric_names.append(m.name)

    headers = ["Date", "Commit", "Scenario", *[truncate(n, 18) for n in all_metric_names],
               "Turns", "Pass", "Duration"]
    alignments = ["l", "l", "l", *["r"] * len(all_metric_names), "r", "c", "r"]

    rows = []
    for r in filtered:
        date = r.timestamp[:10] if len(r.timestamp) >= 10 else r.timestamp
        metric_scores = {m.name: m for m in r.metrics}
        row = [date, r.git.commit, truncate(r.scenario, 20)]
        for mn in all_metric_names:
            m = metric_scores.get(mn)
            row.append(f"{m.score:.2f}" if m else "-")
        row.append(str(r.turns))
        row.append("Y" if r.passed else "N")
        row.append(f"{r.duration_seconds:.1f}s")
        rows.append(row)

    # Footer with summary stats
    if len(filtered) >= 2:
        footer_parts = []
        for mn in all_metric_names:
            scores = [m.score for r in filtered for m in r.metrics if m.name == mn]
            if scores:
                avg = sum(scores) / len(scores)
                delta = scores[-1] - scores[0]
                sign = "+" if delta >= 0 else ""
                footer_parts.append(f"{mn}: avg={avg:.2f} trend={sign}{delta:.2f}")
        footer = " | ".join(footer_parts)
    else:
        footer = ""

    title = "Score Trend"
    if scenario:
        title += f" - {scenario}"
    if provider:
        title += f" ({provider})"

    table = format_table(headers, rows, alignments, fmt=fmt)

    if fmt == "markdown":
        result = f"## {title}\n\n{table}"
        if footer:
            result += f"\n\n{footer}"
        return result

    result = f"{title}\n\n{table}"
    if footer:
        result += f"\n\n{footer}"
    return result
