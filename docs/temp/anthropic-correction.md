# Anthropic harness correction

**Date:** 2026-11-02 · **Base commit:** `f766dbf`, with separately frozen source diffs

**Later finding:** the [handoff contract investigation](./navigator-handoff-qps-investigation.md)
shows the current average-demand template conflicts with downstream planning's
capacity input. The failures below remain valid against the evaluated template;
the next correction should align the shared contract rather than simply make
Claude copy average demand. No historical result has been rewritten.

The subsequent [handoff contract correction](./navigator-handoff-capacity-correction.md)
implements that alignment with offline tests and diagnostic replay. No live
model run has yet used the changed skill/scenario hashes.

The original [four-model sweep](./navigator-sweep.md) remains **5/8**, including
Anthropic **0/2**. This follow-up addresses the harness issues identified in the
[investigation](./anthropic-harness-investigation.md). It uses only Navigator
goldens through EnMaaS; no deployment sweep was added.

## Changes

- **Shared discovery policy:** `build_agent()` now instructs every model to load
  a skill matching the user's request before responding or using domain tools,
  and to follow its workflow and confirmation gates. Both skills remain in the
  catalog; the model still chooses and loads one. The instruction names neither
  Navigator nor any golden and supplies no scenario answers.
- **Explicit Opus thinking:** `enmaas-opus` requests
  `anthropic_thinking={"type": "adaptive"}`. Effort remains unspecified. Native
  Messages, the EnMaaS endpoint/key, and the model ID remain the same. Opus 4.6
  requires thinking to be enabled explicitly at the API level.
  [Anthropic thinking documentation](https://platform.claude.com/docs/en/build-with-claude/thinking).
- **Table row labels:** the checker recognizes Markdown strong emphasis using
  `**...**` or `__...__` when validating labels and extracting the selected Model
  and GPU. Recommendation values and the required table shape stay intact.

Skill text, scripted prompts, checkpoints, judge, rubrics, thresholds, and limits
are unchanged. These are harness corrections; no Opus-specific workflow prompt
or automatic skill preload was introduced.

## Controlled experiments

Each experiment runs the two Navigator goldens once, vague requirements first,
with automatic pytest retries disabled and the source frozen within the batch.
Every attempt is recorded, including failed and incomplete conversations.

**Experiment A: explicit adaptive thinking, original host instructions.** Run
`f4deca98cb83`, **0/2**, with two routing failures after one completed turn each.
Vague attempt `6ab7ac67f1c8` asked for clarification without loading Navigator.
Happy attempt `200a9a71d4eb` tried domain tools directly without loading it. Both
stopped at the existing skill-load checkpoints. Native histories contain one
and five thinking blocks respectively, establishing that EnMaaS accepted and
returned thinking. This rules out thinking configuration alone as a sufficient
correction in these attempts.

Only the happy case had MCP interactions, producing use **0.65** and completion
**0.25**. Vague has no MCP metrics, and neither has fidelity because Navigator
was never delivered. Missing scores are `n/a`, not zero or judge failures.

**Experiment B: shared discovery policy, adaptive thinking retained.** Run
`176efb61812e` discovers Navigator in **2/2** cases, completes **2/2** five-turn
scripts, and passes **6/6 judge metrics**. Both overall verdicts remain failed
because each handoff uses peak QPS rather than expected QPS. This is the only
failed hard check in either case. Native histories confirm the shared policy
and six thinking blocks in each conversation.

| Golden | Attempt | Turns | MCP use | Completion | Fidelity | Failed hard check |
|---|---|---:|---:|---:|---:|---|
| Vague requirements | `cf884cc3a7cf` | 5/5 | 0.95 | 0.80 | 0.90 | Handoff workload/QPS |
| Happy path | `4ea760c43ecb` | 5/5 | 0.90 | 0.80 | 0.90 | Handoff workload/QPS |

Each handoff has `expected_qps: 0.13`; the source `get_expected_rps.expected_rps`
is `0.06666666666666667`. The check remains strict. The judge also notes this
error, along with omitted offers to relax empty slots and small constraint
rounding errors, while still scoring above the fixed cutoffs. Passing judge
thresholds do not establish that all structured fields are correct.

The runtime change between experiments A and B is only the shared host
instruction; the parser and adaptive thinking are identical. This supports the
host-policy correction for discovery, with one stochastic attempt per case.
Thinking alone was insufficient. Neither experiment establishes that the
handoff issue is exclusive to Anthropic.

## Offline validation

All **299 offline evaluation tests pass**, including 19 added parameterized
cases. Tests exercise bold labels, retained table-shape checks, wrong displayed
model/GPU, wrong QPS, native thinking serialization on both requests of a
tool-use cycle, and delivery of the shared policy to all four model families
without preloading either skill. Targeted lint, formatting, and source type
checks pass.

Free replay of all eight original sweep artifacts changes only Anthropic's
happy-path hard checks: its table-label and selected-slot failures disappear,
while `navigator_handoff_agrees_with_workload` still fails for QPS `0.13` instead
of `0.06666666666666667`. Original artifacts, scores, and official verdicts
remain unchanged; no judge ran for this replay.

## Provenance and limits

The two experiments have distinct run IDs and pre-run manifests recording source
diffs/hashes, model settings, scenarios, prior history/artifacts, and fixed judge
settings. They use `caffeinate -i`, pytest `-s --reruns 0`, mock cluster state,
and the existing limits of 300 seconds, 30 requests, and 40 tool calls per turn.
The fixed judge is EnMaaS `gpt-6.1-sol`, cutoffs 0.60/0.75/0.80, MCP metric
version `conversation-v2`, and fidelity `v2`. Live native artifacts and logs
are under `evals/results/`.

The two batches took 71.94 and 250.28 seconds respectively. Their combined
recorded agent time is 226.68 seconds and judge time 94.55 seconds. Power logs
show no sleep/wake events in either interval. There were no timeouts, harness
errors, judge errors, or deployment writes. Four new live conversations were
recorded, with no automatic reruns; currency cost was not returned. All prior
history bytes, native artifacts, and JSON reports were verified unchanged.

[Machine-readable evidence](./anthropic-correction-results.json) includes all
four result records, per-configuration settings/source diffs and hashes,
thinking-block counts, original-artifact replay, and provenance checks. Source
manifests cover 163 runtime/configuration/skill files; test changes are retained
in their accompanying diffs. The final configuration was frozen throughout
its batch, and only `evals/agent/runner.py` changed at runtime between batches.

This is a small controlled diagnostic, not a reliability estimate. The shared
host policy affects all agents, so future comparisons must distinguish this
configuration from the original GPT/GLM/DeepSeek sweep. No full matrix repeat
is required to inspect this correction, and no paid Navigator-deploy work is
included.

## Next macro step

Review the small harness change first. The remaining issue is the handoff's
traffic source: Navigator already names `get_expected_rps.expected_rps` in its
template, but Opus repeatedly substitutes the recommendation's peak traffic.
A narrow clarification distinguishing these source fields is the next skill
change to consider. No skill wording was changed and no further live run was
started in this correction step.
