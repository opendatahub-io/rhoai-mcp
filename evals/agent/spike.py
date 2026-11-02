"""Phase 0 connectivity and adapter spike.

Acceptance (plan §4, Phase 0):

1. One agent/model connects to ``rhoai-mcp`` over stdio, loads a skill, makes a
   ``recommend_model`` call, and continues a second turn in the same agent
   context.
2. Typed DeepEval objects are built from that live run, with the tool call
   attributed to the turn that issued it.
3. The attempt artifact is saved and reloaded into a readable transcript.

A full phase-gated skill flow is explicitly *not* required here; the scripted
replies below only need to get the agent as far as a model recommendation.

Run::

    uv run --group skill-eval python -m evals.agent.spike
    uv run --group skill-eval python -m evals.agent.spike --model enmaas-glm
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from evals.agent.models import DEFAULT_SPEC_IDS, get_spec
from evals.agent.runner import REPO_ROOT, ScriptedTurn, run_conversation
from evals.agent.transcript import AttemptArtifact

RESULTS_ROOT = REPO_ROOT / "evals" / "results"

#: Enough replies to drive the navigator past its workload-confirmation gate and
#: into a recommendation, without asserting the full phase script.
SCRIPT: list[ScriptedTurn] = [
    ScriptedTurn(
        name="opening",
        prompt=(
            "I need an LLM for our customer support bot. About 50 support agents "
            "will use it at peak and a typical request is around 800 tokens."
        ),
    ),
    ScriptedTurn(
        name="priorities",
        prompt=(
            "Response speed matters most, then cost. Please go ahead and use the "
            "defaults for anything else, then show me the model recommendations."
        ),
    ),
    # The navigator gates Phase 3 on an explicit confirmation of the workload
    # profile. A reply that does not confirm leaves the agent waiting, which is
    # correct skill behaviour rather than a model failure - the script has to
    # match the skill's gates.
    ScriptedTurn(
        name="confirm-workload",
        prompt="Confirmed. Go ahead and show me the model recommendations.",
    ),
    # `get_cluster_resources` is not in the default plugin set, so Phase 3 Step 1
    # takes the documented ask-the-customer branch and the agent asks which GPUs
    # the cluster has. Supplying them here is what unblocks `recommend_model`.
    #
    # This means the automated GPU-discovery branch, and the skills' product ->
    # planner-type mapping table, are *unexercised* by this run. That is visible
    # rather than silent, and is the state the goldens must record until the
    # tool is exposed by default (plan §3.1, §3.4).
    ScriptedTurn(
        name="supply-gpu-inventory",
        prompt="The cluster has NVIDIA A100 80GB GPUs.",
    ),
    ScriptedTurn(
        name="follow-up",
        prompt="Of those, which one would you pick and why?",
    ),
]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_SPEC_IDS[0], help="agent model spec id")
    parser.add_argument("--run-id", default="phase0-spike")
    parser.add_argument(
        "--results-root", default=str(RESULTS_ROOT), help="where to write the artifact"
    )
    args = parser.parse_args()

    spec = get_spec(args.model)
    print(f"=== Phase 0 spike: {spec.id} ({spec.model_name} via {spec.api}) ===\n")

    result = await run_conversation(spec, SCRIPT, run_id=args.run_id)

    print(
        f"outcome={result.outcome} turns={result.turns_completed}/{len(SCRIPT)} "
        f"elapsed={result.elapsed_s:.1f}s"
    )
    if result.error:
        print(f"error[{result.error_stage}]: {result.error}")
    print()

    transcript = result.transcript()
    ok = True

    # -- 1. connectivity, skill loading, tool use, multi-turn ---------------- #
    print("1. Live agent run")
    ok &= _check(
        "connected over stdio", bool(result.tool_schemas), f"{len(result.tool_schemas)} tools"
    )
    ok &= _check("loaded a skill", bool(transcript.skills_loaded), str(transcript.skills_loaded))
    ok &= _check(
        "received the skill body",
        bool(transcript.skill_body()),
        f"{len(transcript.skill_body() or '')} chars",
    )
    ok &= _check(
        "called recommend_model",
        "recommend_model" in transcript.tool_names_used,
        ", ".join(transcript.tool_names_used) or "no MCP calls",
    )
    ok &= _check(
        "continued a second turn in the same context",
        result.turns_completed >= 2,
        f"{result.turns_completed} turns",
    )

    # Not a pass/fail condition: which branch runs depends on the server's
    # plugin set, and reporting it keeps the unexercised one visible instead of
    # letting a conditional assertion read as coverage (plan §3.1).
    discovered = "get_cluster_resources" in transcript.tool_names_used
    print(
        f"  [INFO] GPU inventory branch: "
        f"{'automated discovery' if discovered else 'asked the customer'}"
        f" ({'tool available' if discovered else 'tool absent from default plugin set'})"
    )
    if not discovered:
        print("  [INFO] unexercised: GPU product -> planner-type mapping table")

    # -- 2. typed DeepEval objects with correct attribution ------------------ #
    print("\n2. DeepEval objects")
    from mcp.types import CallToolResult

    from evals.deepeval_helpers import (
        NothingToScoreError,
        build_mcp_server_from_schemas,
        transcript_to_conversational_test_case,
    )

    server = build_mcp_server_from_schemas(result.tool_schemas)
    try:
        case = transcript_to_conversational_test_case(transcript, server)
    except NothingToScoreError as exc:
        # A harness error can end an attempt before any message exists. Report
        # the missing precondition rather than crashing; the artifact below is
        # still written so the failure stays debuggable.
        _check("built a ConversationalTestCase", False, str(exc))
        artifact = result.artifact(metadata={"phase": "0-spike", **spec.describe()})
        directory = artifact.save(Path(args.results_root))
        print(f"\n  partial artifact saved: {directory.relative_to(REPO_ROOT)}")
        print("\n=== Phase 0 spike: FAIL ===")
        return 1

    ok &= _check("built a ConversationalTestCase", bool(case.turns), f"{len(case.turns)} turns")
    interaction_turns = [t for t in case.turns if t.mcp_tools_called]
    # mcp_tools_called is Optional on Turn; narrow once rather than at each use.
    calls_by_turn = [(t, list(t.mcp_tools_called or ())) for t in interaction_turns]
    every_call = [c for _, calls in calls_by_turn for c in calls]
    ok &= _check("has MCP interactions", bool(interaction_turns), f"{len(interaction_turns)} turns")
    ok &= _check(
        "results are typed CallToolResult",
        all(isinstance(c.result, CallToolResult) for c in every_call),
    )
    ok &= _check(
        "_mcp_interaction derives True (no manual assignment)",
        all(t._mcp_interaction for t in interaction_turns),
    )
    ok &= _check(
        "skill loading is not counted as an MCP call",
        not any(c.name == "load_capability" for c in every_call),
    )
    recommend_turns = [
        t for t, calls in calls_by_turn if any(c.name == "recommend_model" for c in calls)
    ]
    ok &= _check(
        "recommend_model attributed to a single turn",
        len(recommend_turns) >= 1,
        f"{len(recommend_turns)} turn(s)",
    )

    fidelity = transcript_to_conversational_test_case(transcript, server, include_skill_body=True)
    ok &= _check(
        "fidelity case carries the delivered skill body",
        any("# Skill:" in (t.content or "") for t in fidelity.turns),
    )

    # -- 3. artifact save/load round-trip ------------------------------------ #
    print("\n3. Artifact round-trip")
    artifact = result.artifact(metadata={"phase": "0-spike", **spec.describe()})
    directory = artifact.save(Path(args.results_root))
    reloaded = AttemptArtifact.load(directory)
    rebuilt = reloaded.transcript()

    ok &= _check("saved", directory.exists(), str(directory.relative_to(REPO_ROOT)))
    ok &= _check("reloaded", bool(reloaded.messages()), f"{len(reloaded.messages())} messages")
    ok &= _check("transcript is identical after reload", rebuilt.to_dict() == transcript.to_dict())
    ok &= _check(
        "tool schemas survive the round-trip",
        len(reloaded.mcp_tools()) == len(result.tool_schemas),
    )
    ok &= _check(
        "skill body survives the round-trip", rebuilt.skill_body() == transcript.skill_body()
    )
    ok &= _check(
        "re-scoring the artifact gives the same DeepEval case",
        transcript_to_conversational_test_case(
            rebuilt, build_mcp_server_from_schemas(reloaded.mcp_tools())
        ).turns
        == case.turns,
    )

    # -- summary -------------------------------------------------------------- #
    print("\n--- MCP calls in order ---")
    for callrec in transcript.mcp_tool_calls:
        flag = " ERROR" if callrec.is_error else ""
        print(f"  turn {callrec.turn_index:>2}  {callrec.name}{flag}")

    print(f"\n=== Phase 0 spike: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
