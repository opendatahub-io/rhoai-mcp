"""CLI for eval result reporting.

Provides summary, compare, and trend subcommands for viewing eval results.
"""

from __future__ import annotations

import argparse

from evals.reporting.comparison import provider_comparison_report
from evals.reporting.formatting import format_summary
from evals.reporting.reader import load_records
from evals.reporting.trending import score_trend_report


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--format", choices=["terminal", "markdown"], default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--file", default=None,
        help="Path to eval_history.jsonl (default: evals/results/eval_history.jsonl)",
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for eval reporting CLI."""
    parser = argparse.ArgumentParser(
        prog="evals.reporting",
        description="RHOAI MCP eval result reporting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # summary subcommand
    summary_parser = subparsers.add_parser(
        "summary", help="Show latest eval run summary",
    )
    summary_parser.add_argument("--run-id", default=None, help="Specific run ID to show")
    _add_common_args(summary_parser)

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare", help="Compare eval scores across providers/models",
    )
    compare_parser.add_argument("--scenario", default=None, help="Filter by scenario name")
    compare_parser.add_argument(
        "--last", type=int, default=10,
        help="Use last N records per provider group (default: 10)",
    )
    _add_common_args(compare_parser)

    # trend subcommand
    trend_parser = subparsers.add_parser(
        "trend", help="Show eval score trends over time",
    )
    trend_parser.add_argument("--scenario", default=None, help="Filter by scenario name")
    trend_parser.add_argument(
        "--provider", default=None,
        help="Filter by provider/model (e.g. 'openai/gpt-4o')",
    )
    trend_parser.add_argument(
        "--last", type=int, default=20,
        help="Show last N records (default: 20)",
    )
    _add_common_args(trend_parser)

    args = parser.parse_args(argv)
    records = load_records(args.file)

    if args.command == "summary":
        output = format_summary(records, run_id=args.run_id, fmt=args.format)
    elif args.command == "compare":
        output = provider_comparison_report(
            records, scenario=args.scenario, last_n=args.last, fmt=args.format,
        )
    else:  # trend
        output = score_trend_report(
            records, scenario=args.scenario, provider=args.provider,
            last_n=args.last, fmt=args.format,
        )

    print(output)


if __name__ == "__main__":
    main()
