# Navigator handoff QPS: contract investigation

**Date:** 2026-11-02 · **Method:** saved transcripts and offline adapter probes

**Follow-up:** the [capacity contract correction](./navigator-handoff-capacity-correction.md)
is implemented and verified offline with 330 passing tests. The evidence below
describes the earlier contract; original artifacts and verdicts are preserved.

The failure exposes a **contract inconsistency between Navigator's handoff and
the planner interface**. The current checker faithfully enforces Navigator's
template, which requests average demand. The consumer of that handoff expects
the capacity value used to generate the recommendation. Treating this only as
an Anthropic copying mistake was an incomplete diagnosis.

No skill, checker, runtime code, or official result was changed in this
investigation. There were no model/judge calls or network connections.

## Value trace

For the golden's support chatbot with 50 users:

| Stage and field | Meaning | Value |
|---|---|---:|
| `get_expected_rps.expected_rps` | Average demand | 0.06666666666666667 |
| `get_expected_rps.peak_rps` | Buffered capacity, rounded by the planner | 0.13 |
| `recommend_model.specification.traffic_profile.expected_qps` | Capacity used for recommendations | 0.13 |
| Navigator's current handoff template | Copies average demand into `expected_qps` | 0.06666666666666667 |
| Native `plan_deployment` input contract | Takes `expected_qps` from the recommendation specification | 0.13 |
| Opus's emitted handoff | Matches recommendation capacity | 0.13 |

The average is `50 × 0.20 × 0.4 / 60`. Installed `llm-d-planner` 0.3.3 applies
a peak multiplier of 2, a minimum capacity of 0.1 QPS, then rounds to two
decimals. Its field is named `expected_qps` although its generator explicitly
describes it as peak capacity. The local adapter exposes this as `peak_rps`
while calculating average demand separately.

## Where the mismatch occurs

Review covered **12 saved attempts**, of which **9 emitted handoffs**. All nine
received average `0.06666666666666667` and peak `0.13`; every successful
recommendation result also had capacity `0.13`.

The three completed Anthropic attempts emitted `0.13`. The two corrected-harness
attempts displayed both rates correctly before workload confirmation, then
emitted the capacity value in the final handoff. The first visible substitution
is therefore in handoff construction, after recommendation results arrive.
This establishes the conflicting source values, not the model's internal cause
for choosing one.

The six GPT/GLM/DeepSeek handoffs copied average demand and passed the existing
QPS check. They comply with the literal Navigator template. Under a corrected
capacity contract they would need different output too. This is a shared
contract issue; changing the checker only to admit Claude would be inappropriate.

## Downstream impact

Navigator-deploy's Opening says to parse the handoff and skip directly to
planning. Its standalone planning instructions and the `plan_deployment` tool
both specify traffic from the recommendation specification. The two paths
therefore currently supply different traffic targets for the same workload.

Offline probes exercised the actual local and remote adapter `generate_config`
methods, stopping immediately before recommendation generation:

| Adapter | Supplied handoff QPS | QPS sent into recommendation generation |
|---|---:|---:|
| Local | 0.06666666666666667 | 0.06666666666666667 |
| Local | 0.13 | 0.13 |
| Remote | 0.06666666666666667 | 0.06666666666666667 |
| Remote | 0.13 | 0.13 |

Both adapters generate a specification and then overwrite its QPS with the
caller-supplied value. Neither reapplies the capacity multiplier or floor.
The current average-demand handoff thus reduces the planning input by about
49% relative to the recommendation in this example. For a one-user chatbot,
average demand is about 0.00133 while the planner's capacity floor is 0.1, so
hardcoding a factor of two would also be incorrect.

This proves a changed planning input, not a particular GPU-count or deployment
change. No hardware sizing result or deployment was generated. The remote
probe used a locally generated specification fixture and stubbed API methods;
it does not establish the behavior of a separately deployed planner backend.

## Recommended correction

Preserve the selected recommendation's capacity through the handoff:

1. In Navigator's handoff template, source `expected_qps` from the selected
   successful `recommend_model` response's
   `specification.traffic_profile.expected_qps`. Explain that this includes the
   planner's buffer/floor. Keep displaying average and peak rates in Phase 2.
   Copy the returned value; do not recompute it or multiply it again.
2. Update the handoff agreement check to use that same selected recommendation
   as the QPS authority. Keep checking use case, users, token sizes, SLOs, and
   GPU inventory against their existing evidence. Missing, invalid, or
   non-finite capacity evidence must fail; do not fall back to average demand.
3. Add offline regression coverage for the different average/capacity values,
   the small-workload floor, wrong values, missing evidence, and selected-slot
   provenance. Verify the handoff-to-planning traffic path without a paid
   Navigator-deploy conversation.
4. Keep old results tied to their original instructions and contract. Free
   replay can show the new rule's effect, but must not relabel old model runs as
   compliant/noncompliant with instructions they were never given. Any live
   Navigator follow-up uses a new skill hash/configuration and separate records.

This requires no public API rename, extra handoff fields, or planner arithmetic
change. An alternative is to preserve average demand and explicitly reconstruct
capacity in the consumer, but that changes more behavior and duplicates planner
rounding/floor logic. The existing deployment API already accepts the selected
recommendation's capacity directly.

## Evidence

[Machine-readable investigation](./navigator-handoff-qps-investigation-results.json)
contains per-attempt source values and visible traffic lines, four adapter-input
captures, generated traffic for four workloads, source hashes, and preservation
checks for original JSON reports, histories, and native artifacts. The probe
blocks socket connections and stops before recommendation/deployment generation.

Relevant contracts and implementation:

- [Navigator handoff](../../.claude/skills/navigator/SKILL.md), Handoff section.
- [Navigator-deploy consumer](../../.claude/skills/navigator-deploy/SKILL.md), Opening and Phase 4.
- [Local planner adapter](../../src/rhoai_mcp/composites/planner/local_client.py), `get_expected_rps`, `recommend`, and `generate_config`.
- [Remote planner adapter](../../src/rhoai_mcp/composites/planner/client.py), `generate_config`.
- [Deployment tool contract](../../src/rhoai_mcp/composites/planner/tools.py), `plan_deployment`.
- [Current agreement checks](../../evals/skills/agreement.py), `confirmed_workload` and `plan_selection_errors`.

Original sweep and Anthropic follow-up verdicts remain unchanged. The next macro
step is the shared contract correction above, beginning with offline coverage.
