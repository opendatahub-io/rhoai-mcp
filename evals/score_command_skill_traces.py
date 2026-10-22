"""Score saved command/skill workflows with DeepEval, without running Claude or LCS."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.command_skill_check import SCENARIOS, compare_artifacts
from evals.config import EvalConfig, LLMProvider
from evals.deepeval_helpers import (
    build_mcp_server_from_schemas,
    fetch_stdio_tool_schemas,
    result_to_conversational_test_case,
)
from evals.mcp_config import REPO_ROOT
from evals.metrics.config import create_multi_turn_mcp_use_metric, create_task_completion_metric
from evals.reporting.formatting import format_table
from evals.reporting.models import EvalRecord
from evals.reporting.reader import load_records
from evals.reporting.recorder import EvalRecorder, evaluate_and_record
from evals.result_types import ClaudeCodeResult, ClaudeToolCall


def scenario_result(scenario: dict[str, Any]) -> ClaudeCodeResult:
    """Concatenate each observed exchange once, preserving user gates and tool IDs."""
    exchanges = scenario["exchanges"]
    return ClaudeCodeResult(
        task=exchanges[0]["task"],
        final_output=exchanges[-1]["final_output"],
        messages=[message for exchange in exchanges for message in exchange["messages"]],
        tool_calls=[
            ClaudeToolCall(**call) for exchange in exchanges for call in exchange["tool_calls"]
        ],
        turns=sum(exchange["turns"] for exchange in exchanges),
        conversation_id=exchanges[0]["conversation_id"],
    )


def paired_report(records: list[EvalRecord], run_id: str) -> str:
    """Pair scenario/metric rows within this scoring run, without pooling older runs."""
    values = {
        (record.phase, record.scenario, metric.name): metric
        for record in records
        if record.run_id == run_id
        for metric in record.metrics
    }
    names = sorted({(scenario, name) for _phase, scenario, name in values})
    rows = []
    for scenario, name in names:
        before = values.get(("commands", scenario, name))
        after = values.get(("skills", scenario, name))
        rows.append(
            [
                scenario,
                name,
                f"{before.score:.3f}" if before else "missing",
                f"{after.score:.3f}" if after else "missing",
                f"{after.score - before.score:+.3f}" if before and after else "—",
                "PASS" if before and before.success else "FAIL",
                "PASS" if after and after.success else "FAIL",
            ]
        )
    return format_table(
        ["Scenario", "Metric", "Commands", "Skills", "Delta", "Commands pass", "Skills pass"],
        rows,
        fmt="markdown",
    )


def score_traces(
    *,
    before: Path,
    after: Path,
    output_dir: Path,
    config: EvalConfig,
    schema_fetcher: Callable[..., Any] = fetch_stdio_tool_schemas,
    evaluator: Callable[..., Any] = evaluate_and_record,
    metric_factory: Callable[[EvalConfig], list[Any]] | None = None,
) -> bool:
    """Validate the primary gate first, then score both complete workflows in each phase."""
    artifacts = {
        "commands": json.loads(before.read_text(encoding="utf-8")),
        "skills": json.loads(after.read_text(encoding="utf-8")),
    }
    failures = [
        f"{name}: {detail}"
        for name, passed, detail in compare_artifacts(artifacts["commands"], artifacts["skills"])
        if not passed
    ]
    if failures:
        raise ValueError("Primary gate artifacts are invalid: " + "; ".join(failures))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite or append to an earlier scoring run in this directory.
    for name in (
        "commands.jsonl",
        "skills.jsonl",
        "run.json",
        "comparison.md",
        "tool-schemas.json",
    ):
        if (output_dir / name).exists():
            raise ValueError(
                f"Scoring output already exists: {output_dir / name}; use a new output directory"
            )
    run_id = uuid.uuid4().hex[:12]
    manifest = {
        "run_id": run_id,
        "status": "running",
        "failure": None,
        "judge_provider": config.eval_provider.value,
        "judge_model": config.eval_model,
        "sources": {
            phase: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for phase, path in (("commands", before), ("skills", after))
        },
    }
    try:
        schemas = asyncio.run(schema_fetcher(REPO_ROOT))
        (output_dir / "tool-schemas.json").write_text(
            json.dumps([tool.model_dump(mode="json") for tool in schemas], indent=2) + "\n",
            encoding="utf-8",
        )
        server = build_mcp_server_from_schemas(schemas)
        prepared = {}
        for phase, artifact in artifacts.items():
            for name in SCENARIOS:
                result = scenario_result(artifact["scenarios"][name])
                unknown = set(result.tool_names_used) - {tool.name for tool in schemas}
                if unknown:
                    raise ValueError(
                        f"{phase}/{name}: recorded tools missing from current schemas: {sorted(unknown)}"
                    )
                test_case = result_to_conversational_test_case(result, server)
                test_case.name = f"{phase}/{name}"
                prepared[phase, name] = (result, test_case)
        for phase, artifact in artifacts.items():
            model_names = {
                scenario["initialization"].get("model", "")
                for scenario in artifact["scenarios"].values()
            }
            recorder = EvalRecorder(
                config,
                path=output_dir / f"{phase}.jsonl",
                phase=phase,
                agent_backend="claude-code",
                agent_model=", ".join(sorted(model_names)),
                run_id=run_id,
            )
            for name in SCENARIOS:
                result, test_case = prepared[phase, name]
                metrics = (
                    metric_factory(config)
                    if metric_factory
                    else [
                        create_multi_turn_mcp_use_metric(config),
                        create_task_completion_metric(config),
                    ]
                )
                print(
                    f"Scoring {phase}/{name} with {config.eval_provider.value}/{config.eval_model}",
                    flush=True,
                )
                evaluator(
                    recorder=recorder,
                    scenario=name,
                    lcs_result=result,
                    test_cases=[test_case],
                    metrics=metrics,
                )
        records = load_records(str(output_dir / "commands.jsonl")) + load_records(
            str(output_dir / "skills.jsonl")
        )
        expected = {(phase, name) for phase in artifacts for name in SCENARIOS}
        if {(record.phase, record.scenario) for record in records} != expected or len(records) != 4:
            raise ValueError("Missing or duplicate phase/scenario scoring results")
        report = paired_report(records, run_id)
        (output_dir / "comparison.md").write_text(report + "\n", encoding="utf-8")
        metrics_by_phase = {
            (record.phase, record.scenario): {metric.name for metric in record.metrics}
            for record in records
        }
        paired = all(
            metrics_by_phase["commands", name] == metrics_by_phase["skills", name]
            for name in SCENARIOS
        )
        passed = paired and all(record.passed for record in records)
        manifest["status"] = "passed" if passed else "failed"
        print(report)
        return passed
    except Exception as exc:
        manifest.update(status="error", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output_dir / "run.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-provider", choices=[provider.value for provider in LLMProvider])
    parser.add_argument("--judge-model")
    args = parser.parse_args(argv)
    overrides = {}
    if args.judge_provider:
        overrides["eval_provider"] = args.judge_provider
    if args.judge_model:
        overrides["eval_model"] = args.judge_model
    try:
        return (
            0
            if score_traces(
                before=args.before,
                after=args.after,
                output_dir=args.output_dir,
                config=EvalConfig(**overrides),
            )
            else 1
        )
    except Exception as exc:
        print(f"Scoring failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
