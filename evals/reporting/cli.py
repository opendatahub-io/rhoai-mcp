"""CLI for eval result reporting.

Provides summary, compare, and trend subcommands for viewing eval results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.reporting.comparison import provider_comparison_report
from evals.reporting.formatting import format_summary
from evals.reporting.reader import load_records
from evals.reporting.trending import score_trend_report


def _add_common_args(parser: argparse.ArgumentParser, *, json_output: bool = False) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--format",
        choices=["terminal", "markdown", "json"] if json_output else ["terminal", "markdown"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--file",
        default=None,
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
        "summary",
        help="Show latest eval run summary",
    )
    summary_parser.add_argument("--run-id", default=None, help="Specific run ID to show")
    _add_common_args(summary_parser)

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare eval scores across providers/models",
    )
    compare_parser.add_argument("--scenario", default=None, help="Filter by scenario name")
    compare_parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Use last N attempts per model/golden for skill comparisons (default: 10)",
    )
    selection = compare_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cohort", type=Path, help="Frozen sweep JSON containing records and frozen_configuration"
    )
    selection.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Original run ID; repeat for all models in the selected sweep",
    )
    compare_parser.add_argument(
        "--artifacts-root",
        type=Path,
        help="Portable artifact root containing original run/attempt directories",
    )
    compare_parser.add_argument(
        "--scoring",
        choices=["original", "latest"],
        default="original",
        help="Choose the original or latest scoring for each attempt (default: original)",
    )
    compare_parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Record that attempts ran concurrently, so timing is descriptive",
    )
    compare_parser.add_argument("--output", type=Path, help="Write the comparison to this path")
    _add_common_args(compare_parser, json_output=True)

    # trend subcommand
    trend_parser = subparsers.add_parser(
        "trend",
        help="Show eval score trends over time",
    )
    trend_parser.add_argument("--scenario", default=None, help="Filter by scenario name")
    trend_parser.add_argument(
        "--provider",
        default=None,
        help="Filter by provider/model (e.g. 'openai/gpt-4o')",
    )
    trend_parser.add_argument(
        "--last",
        type=int,
        default=20,
        help="Show last N records (default: 20)",
    )
    _add_common_args(trend_parser)

    args = parser.parse_args(argv)
    if args.command == "compare" and args.cohort and args.file:
        parser.error("Choose --cohort or --file, not both")
    records = [] if args.command == "compare" and args.cohort else load_records(args.file)

    if args.command == "summary":
        output = format_summary(records, run_id=args.run_id, fmt=args.format)
    elif args.command == "compare":
        try:
            output = provider_comparison_report(
                records,
                scenario=args.scenario,
                last_n=args.last,
                fmt=args.format,
                cohort_path=args.cohort,
                run_ids=args.run_ids,
                artifacts_root=args.artifacts_root,
                scoring=args.scoring,
                concurrent=args.concurrent,
            )
            if args.output:
                inputs = [args.cohort, Path(args.file)] if args.file else [args.cohort]
                if any(p and p.resolve() == args.output.resolve() for p in inputs):
                    parser.error("Output must not overwrite comparison inputs")
                from evals.reporting.reader import DEFAULT_RESULTS_PATH

                if args.output.resolve() == DEFAULT_RESULTS_PATH.resolve():
                    parser.error("Output must not overwrite eval history")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(output + "\n")
                return
        except (ValueError, OSError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    else:  # trend
        output = score_trend_report(
            records,
            scenario=args.scenario,
            provider=args.provider,
            last_n=args.last,
            fmt=args.format,
        )

    print(output)


if __name__ == "__main__":
    main()
