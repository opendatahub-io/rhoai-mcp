# Navigator handoff: preserve recommendation capacity

**Date:** 2026-11-02 · **Validation:** offline only

Navigator's handoff now preserves the capacity used to generate the customer's
chosen recommendation. This resolves the [contract inconsistency](./navigator-handoff-qps-investigation.md)
between the previous average-demand handoff and deployment planning's capacity
input. For the 50-user chatbot, the handoff carries **0.13 QPS**, retaining the
planner's buffer, while Phase 2 still displays average demand of about 0.0667.

## Changes

- The Navigator skill sources `expected_qps` from
  `specification.traffic_profile.expected_qps` in the successful response that
  supplied the selected slot. It says to preserve the returned value, including
  the planner's buffer and minimum, without recomputation.
- The handoff checker resolves the selected model/GPU offer before checking
  QPS, then uses that response's capacity. Missing or invalid capacity fails;
  average demand and Phase 2 peak estimates are not fallback sources.
- Recommendation responses containing a structured `error` cannot supply
  selected-offer evidence. Failed native tool outcomes remain excluded too.
- Confirmed average traffic is still required to be finite and positive.
  User count, use case, tokens, SLOs, category, model, and inventory retain their
  checks. Numeric tolerance remains unchanged.
- Both Navigator golden descriptions now state the capacity requirement, which
  changes their scenario hashes. Scripted prompts and checkpoints are unchanged.

No planner arithmetic, deployment skill, tool API, judge rubric, cutoff, or
agent setting changed for this step.

## Verification

**330 offline evaluation tests passed**, including 31 new parameterized cases.
Targeted lint, formatting, and source type checks passed. Coverage includes:

- distinct average, Phase 2 peak, and selected-capacity values;
- minimum capacity for small workloads and rejection of a recomputed average;
- absent, non-finite, non-positive, or boolean capacity sources;
- invalid confirmed average traffic despite an otherwise valid handoff;
- failed/structured-error returns, unrelated reruns, and post-selection results;
- parsing a capacity handoff and passing it into the deployment-plan contract.

The existing [offline adapter captures](./navigator-handoff-qps-investigation-results.json)
establish that local and remote adapters preserve the supplied QPS unchanged.
Their source hashes still match; this correction needs no adapter change.

Free replay of **12 saved artifacts** changes only
`navigator_handoff_agrees_with_workload`: three capacity handoffs now match the
new contract, six average-demand handoffs do not, and three partial attempts
remain failed. The two completed corrected-harness Opus artifacts have no
failed hard checks under the new rule. This is a **counterfactual contract
diagnostic**, not a new live pass or a reassessment of adherence to the old
instructions those models received. Judge scores were not rerun or relabelled.

## Configuration and provenance

| Identity | Before | After |
|---|---|---|
| Navigator skill SHA | `5094f4ddc18b` | `25392e419e13` |
| Happy-path scenario SHA | `33f00a099618` | `eaa225a67e99` |
| Vague-requirements scenario SHA | `2b3824a27abf` | `f1f282812e97` |

[Machine-readable validation](./navigator-handoff-capacity-correction-results.json)
contains before/after hard checks, source/configuration hashes, and preservation
evidence. All **113 previous report/history/native evidence files** were verified
unchanged. Normal history still contains 46 records. No agent, judge, or paid
deployment conversation was run for this step.

The original sweep remains its historical **5/8** under its evaluated contract.
No live model has yet been tested against the new skill/scenario hashes. The
next reviewable live step, if continued, is a separate two-golden Anthropic
Navigator batch with the existing shared discovery policy, adaptive thinking,
judge, and cutoffs. Keep it separate from the original sweep; a full matrix or
Navigator-deploy repeat is outside this correction.
