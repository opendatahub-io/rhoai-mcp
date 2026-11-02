# Anthropic sweep: harness investigation

**Date:** 2026-11-02 · **Evaluated source:** `f766dbf`

**Follow-up:** the [harness correction](./anthropic-correction.md) implements the
parser fix, explicit adaptive thinking, and shared discovery instructions in
separate controlled steps. Thinking alone failed to resolve discovery; the
shared host policy restored it in both full flows. Both retain a real QPS
handoff failure. The investigation below records the original evidence and
proposals; the correction report is the current implementation status.

The user reports successful automatic Navigator discovery in Claude Code using
the same `claude-opus-4-6` model. That rules out slash invocation and model ID as
explanations for the difference. Our **0/2** describes the frozen Pydantic harness
configuration; it does not establish that Opus cannot use Navigator.

## 1. Reasoning configuration is the first controlled experiment

`enmaas-opus` has `extra_settings={}`. The runner supplies no thinking override.
An offline native Messages transport audit using the installed Pydantic stack,
the same model/provider and 53 saved MCP schemas confirms that both initial and
post-load requests omit `thinking` and `output_config`. Neither saved Anthropic
attempt contains a thinking part.

For Opus 4.6, the native API leaves thinking off unless enabled explicitly.
[Anthropic thinking documentation](https://platform.claude.com/docs/en/build-with-claude/thinking).
Claude Code supports adaptive reasoning for this model and a separate toggle to
disable thinking. Its effective user settings were not captured here.
[Claude Code model configuration](https://code.claude.com/docs/en/model-config#adaptive-reasoning-and-fixed-thinking-budgets).

**Inference:** the absence of an explicit thinking setting is a strong
configuration suspect. Provider defaults are not equivalent reasoning settings.
The audit does not establish EnMaaS's upstream effective settings or prove this
caused the failures. Absence of returned thinking blocks alone would also be
insufficient proof.

The smallest proposed change is to add
`extra_settings={"anthropic_thinking": {"type": "adaptive"}}` to `enmaas-opus`.
First verify its native request offline, then qualify that option through EnMaaS
with a bounded diagnostic. Leave effort at the provider default for this first
experiment; record it explicitly as unspecified. Keep host instructions, skill,
goldens, judge, and cutoffs fixed to isolate the setting. A future live repeat
must be separate from the completed sweep, with no automatic retries.

## 2. Skill discovery is underspecified by the host

`build_agent()` installs `LocalWorkspace`, `Skills`, and the MCP toolset, with no
additional host instructions. Its initial catalog says capabilities **can** be
loaded using `load_capability`; it never tells the model it must load a matching
skill before asking clarifying questions or calling MCP tools.

All 53 MCP tools are visible immediately. Loading Navigator adds instructions;
it does not reveal these tools. The generic catalog's claim that a capability's
tools stay hidden applies to capability-owned tools, not this separate MCP
toolset. Navigator is an instruction-only capability here.

The vague golden nevertheless requires Navigator loaded before sending the
second user message. In attempt `0472b7917e09`, Claude used `list_use_cases` and
asked sensible clarification questions. The checkpoint stopped the script before
the customer supplied the requested information. Navigator's body was never
loaded, so downstream failures and missing fidelity are consequences of that
stop, not observations of a completed workflow.

Pydantic's skill mechanism supplies the catalog and deferred loading; automatic
loading remains a model decision.
[Pydantic skills documentation](https://pydantic.dev/docs/ai/harness/skills/).
Claude Code also discovers skills from their descriptions and loads their bodies
when used, but this does not make the two hosts' full prompts identical.
[Claude Code skills documentation](https://code.claude.com/docs/en/skills).

If reasoning configuration alone does not resolve discovery, the next proposed
host change is one shared instruction: load an applicable catalog skill before
responding or using domain tools, then follow its workflow. Apply it uniformly
to all models, retain both skill choices, and test discovery offline. Do not
hardwire Navigator for Claude or weaken the checkpoint to improve its score.

## 3. Skill delivery works after loading

The happy-path native history contains the complete Navigator instruction body.
The offline transport audit verifies the full body also reaches the serialized
Messages request after `load_capability`: in a user message wrapped in `<system>`
tags, with a top-level explanation of that fallback. This is the installed
Pydantic adapter's handling for Opus 4.6; it is not evidence that the skill was
silently dropped. No causal claim about the message role is established.

The audit scripts the load response and final text, uses a dummy key and an
in-memory HTTP transport, and executes no MCP action. Its only local settings
override is `max_tokens=1024` to avoid SDK streaming constraints. It tests
serialization, not model behavior or gateway acceptance. Local evidence:
`evals/results/anthropic_wire_diagnostic.py` and
`evals/results/navigator-sweep-anthropic-wire-diagnostic.json`.

The harness warns that skill `allowed-tools` frontmatter is unsupported. This
does not by itself explain the Claude Code difference: Claude Code documents
that field as permission preapproval, not an exclusive tool restriction.
[Claude Code skill permissions](https://code.claude.com/docs/en/skills).

## 4. Formatting inflates the happy-path hard-failure count

Attempt `841518974181` has three failed hard checks. On a derived in-memory
transcript, removing bold markers only from table row labels resolves
`table_rows_exact` and `navigator_handoff_agrees_with_selection`. The independent
workload/QPS check still fails (`0.13` versus `0.06666666666666667`). Fidelity
remains the original failing `0.70`; no judge was rerun.

This demonstrates a parser sensitivity rather than a missing selection. A small
shared parser correction should accept equivalent bold/plain labels while still
rejecting wrong labels, order, columns, missing cells, and changed values.
Regression tests and free replay should verify both the recovered selection and
the retained QPS failure. Keep the original official records unchanged.

## Reviewable next steps

1. Correct the table-label parser with offline regression coverage and replay;
   preserve original scores and verdicts.
2. Add explicit adaptive thinking to the Opus spec, with offline wire coverage.
   A bounded EnMaaS qualification/repeat would then isolate this one behavioral
   change; it is not part of the completed eight-conversation sweep.
3. If discovery still fails, review the shared host-instruction contract as a
   separate change before considering any skill wording changes.
4. Resume Navigator comparison/reporting with configuration-aware groups and
   `n/a` for missing scores. Keep Navigator-deploy paid work deferred.

No benchmark source, checkpoint, parser, prompt, or model configuration was
changed during this investigation. No additional paid model or judge calls were
made. Original results remain **Anthropic 0/2**, with the concrete caveats above.
