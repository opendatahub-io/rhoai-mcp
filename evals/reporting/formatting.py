"""Terminal and markdown table formatting for eval reports."""

from __future__ import annotations

from evals.reporting.models import EvalRecord


def truncate(text: str, width: int) -> str:
    """Truncate text to width, adding ellipsis if needed."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


def provider_label(record: EvalRecord) -> str:
    """Format provider/model as a compact label."""
    return f"{record.environment.llm_provider}/{record.environment.llm_model}"


def format_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
    fmt: str = "terminal",
) -> str:
    """Render a fixed-width table.

    Args:
        headers: Column header strings.
        rows: List of row data (each row is a list of strings).
        alignments: Per-column alignment ('l', 'r', 'c'). Defaults to left.
        fmt: 'terminal' for ASCII borders, 'markdown' for GFM table.
    """
    if not headers:
        return ""

    num_cols = len(headers)
    if alignments is None:
        alignments = ["l"] * num_cols

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(cell))

    def _pad(text: str, width: int, align: str) -> str:
        if align == "r":
            return text.rjust(width)
        if align == "c":
            return text.center(width)
        return text.ljust(width)

    if fmt == "markdown":
        header_line = "| " + " | ".join(
            _pad(h, col_widths[i], alignments[i]) for i, h in enumerate(headers)
        ) + " |"
        sep_parts = []
        for i in range(num_cols):
            dash = "-" * col_widths[i]
            if alignments[i] == "r":
                sep_parts.append("-" * (col_widths[i] - 1) + ":")
            elif alignments[i] == "c":
                sep_parts.append(":" + "-" * (col_widths[i] - 2) + ":")
            else:
                sep_parts.append(dash)
        sep_line = "| " + " | ".join(sep_parts) + " |"
        data_lines = []
        for row in rows:
            padded = [_pad(row[i] if i < len(row) else "", col_widths[i], alignments[i])
                      for i in range(num_cols)]
            data_lines.append("| " + " | ".join(padded) + " |")
        return "\n".join([header_line, sep_line, *data_lines])

    # Terminal format with +---+ borders
    border = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(
        _pad(h, col_widths[i], alignments[i]) for i, h in enumerate(headers)
    ) + " |"
    data_lines = []
    for row in rows:
        padded = [_pad(row[i] if i < len(row) else "", col_widths[i], alignments[i])
                  for i in range(num_cols)]
        data_lines.append("| " + " | ".join(padded) + " |")

    return "\n".join([border, header_line, border, *data_lines, border])


def format_summary(records: list[EvalRecord], run_id: str | None = None,
                   fmt: str = "terminal") -> str:
    """Format a summary table for a single eval run.

    If run_id is None, uses the latest run_id found in records.
    """
    if not records:
        return "No eval records found."

    if run_id is None:
        run_id = records[-1].run_id

    run_records = [r for r in records if r.run_id == run_id]
    if not run_records:
        return f"No records found for run_id={run_id}"

    # Collect all unique metric names across scenarios
    all_metric_names: list[str] = []
    for r in run_records:
        for m in r.metrics:
            if m.name not in all_metric_names:
                all_metric_names.append(m.name)

    first = run_records[0]
    title = f"Eval Run: {run_id} | {first.git.branch}@{first.git.commit}"
    provider_info = f"Agent: {first.environment.llm_provider}/{first.environment.llm_model}"
    judge_info = f"Judge: {first.environment.eval_provider}/{first.environment.eval_model}"

    headers = ["Scenario", *[truncate(n, 25) for n in all_metric_names],
               "Turns", "Pass", "Duration"]
    alignments = ["l", *["r"] * len(all_metric_names), "r", "c", "r"]

    rows = []
    for r in run_records:
        metric_scores = {m.name: m for m in r.metrics}
        row = [truncate(r.scenario, 25)]
        for mn in all_metric_names:
            m = metric_scores.get(mn)
            row.append(f"{m.score:.2f}" if m else "-")
        row.append(str(r.turns))
        row.append("Y" if r.passed else "N")
        row.append(f"{r.duration_seconds:.1f}s")
        rows.append(row)

    table = format_table(headers, rows, alignments, fmt=fmt)

    if fmt == "markdown":
        return f"## {title}\n\n{provider_info} | {judge_info}\n\n{table}"
    return f"{title}\n{provider_info} | {judge_info}\n\n{table}"
