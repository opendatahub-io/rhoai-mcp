"""Pretty-print a saved attempt artifact: the trajectory, offline.

Everything here comes out of ``messages.json``, so it needs no collector, no
server, and no network. It is the first thing to reach for when a check or a
rubric disagrees with your reading of a run.

This complements tracing rather than replacing it (plan §4): a trace viewer adds
timing, cross-run comparison and a UI, but it cannot show what was never
exported, and it is not available offline or after the fact.

Usage::

    uv run --group skill-eval python -m evals.skills.show --attempt <id>
    uv run --group skill-eval python -m evals.skills.show --attempt <id> --full
    uv run --group skill-eval python -m evals.skills.show --run <id> --list
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from evals.agent.transcript import AgentTranscript, AttemptArtifact
from evals.skills.goldens import GOLDENS_BY_ID
from evals.skills.rescore import find_artifacts, render_listing

RULE = "─" * 78


def _wrap(text: str, indent: str = "    ", limit: int | None = 600) -> str:
    body = text if limit is None else text[:limit] + ("…" if len(text) > limit else "")
    return "\n".join(
        textwrap.fill(line, width=78, initial_indent=indent, subsequent_indent=indent) or indent
        for line in body.splitlines()
    )


def render(artifact: AttemptArtifact, *, full: bool = False) -> str:
    """Render one attempt as a readable trajectory."""
    transcript: AgentTranscript = artifact.transcript()
    meta = artifact.metadata
    limit = None if full else 600
    out: list[str] = []

    out.append(RULE)
    out.append(f"attempt {artifact.attempt_id}   run {artifact.run_id}")
    out.append(f"golden  {meta.get('golden_id', '?')}   skill {meta.get('skill', '?')}")
    out.append(
        f"model   {meta.get('id', '?')} ({meta.get('model_name', '?')} via {meta.get('api', '?')})"
    )
    out.append(f"created {artifact.created_at[:19]}   outcome {meta.get('outcome', '?')}")
    if meta.get("error"):
        out.append(f"error   [{meta.get('error_stage')}] {meta['error']}")
    out.append(RULE)

    out.append(f"routing      : loaded {transcript.skills_loaded or 'nothing'}")
    body = transcript.skill_body()
    out.append(f"skill body   : {len(body) if body else 0} chars delivered")
    out.append(f"catalog      : {len(transcript.skill_catalog)} chars offered")
    out.append(f"MCP tools    : {', '.join(transcript.tool_names_used) or 'none'}")
    out.append(f"schemas      : {len(artifact.tool_schemas)} tools offered to the model")
    out.append(RULE)

    for index, turn in enumerate(transcript.turns):
        if turn.role == "user":
            out.append(f"\n[{index}] USER")
            out.append(_wrap(turn.content, limit=limit))
            continue

        out.append(f"\n[{index}] ASSISTANT")
        if turn.thinking and full:
            out.append(_wrap(f"(thinking) {turn.thinking}", limit=limit))
        for call in turn.tool_calls:
            kind = "harness" if call.kind == "harness" else "mcp"
            flag = "  ERROR" if call.is_error else ""
            out.append(f"    → [{kind}] {call.name}({_args(call.arguments)}){flag}")
            if call.error:
                out.append(_wrap(f"error: {call.error}", indent="        ", limit=limit))
            elif call.result_text is not None:
                out.append(
                    _wrap(
                        f"result: {call.result_text}",
                        indent="        ",
                        limit=None if full else 300,
                    )
                )
        if turn.content:
            out.append(_wrap(turn.content, limit=limit))

    golden = GOLDENS_BY_ID.get(str(meta.get("golden_id", "")))
    if golden is not None:
        out.append("\n" + RULE)
        out.append("deterministic checks")
        report = golden.grade(transcript)
        for check in report.results:
            marker = {"pass": "ok  ", "fail": "FAIL", "n/a": "n/a ", "unexercised": "----"}[
                check.status
            ]
            detail = f"  {check.detail}" if check.detail and check.status != "pass" else ""
            out.append(f"  [{marker}] {check.name}{detail[:100]}")
        out.append(f"  summary: {report.summary()}  passed={report.passed}")

    out.append(RULE)
    return "\n".join(out)


def _args(arguments: dict) -> str:
    rendered = json.dumps(arguments, ensure_ascii=False)
    return rendered if len(rendered) <= 110 else rendered[:107] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", dest="attempt_id", help="attempt id (prefix match)")
    parser.add_argument("--run", dest="run_id", help="run id (prefix match)")
    parser.add_argument("--golden", dest="golden_id", help="only this golden")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--list", action="store_true", help="list matches and exit")
    parser.add_argument(
        "--full", action="store_true", help="do not truncate text, and show thinking"
    )
    args = parser.parse_args(argv)

    paths = find_artifacts(
        Path(args.results_root) if args.results_root else None,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        golden_id=args.golden_id,
    )
    if not paths:
        print("no artifacts matched", file=sys.stderr)
        return 1

    if args.list:
        print(render_listing(paths))
        return 0

    for path in paths:
        print(render(AttemptArtifact.load(path), full=args.full))
    return 0


if __name__ == "__main__":
    sys.exit(main())
