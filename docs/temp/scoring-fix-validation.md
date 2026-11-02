# Scoring fixes: frozen-corpus validation

**Date:** 2026-11-02 · **Result:** 150/150 reviewed labels matched, zero errors

Three small source changes address the gaps in the
[previous extension](./judge-calibration-extension.md):

- Fidelity's existing call annotation now includes returned plan `ready`,
  `issues`, and `warnings`, explicitly marked as internal evidence rather than
  customer-visible prose. It does not dump generated YAML.
- The approved golden's existing description explicitly accepts successful
  execution followed by an honest provisioning handoff. Plans without successful
  execution remain incomplete; this scenario does not require a Ready endpoint.
- Fidelity evidence is versioned `v2`; descriptions participate in scenario
  hashes, and calibration input hashes include the fidelity version. MCP
  calculations/prompts remain `conversation-v2`; rubric wording is unchanged.

The judge remains EnMaaS `gpt-6.1-sol`, with the same 0.60 / 0.75 / 0.80 cutoffs
and provider-default sampling. No labels or mutations were changed. All 14
frozen examples were judged five times on each of three metrics, for **210
measurements**, with eight concurrent measurements and **zero errors**.

| Corpus | Measurements | Reviewed labels matched |
|---|---:|---:|
| Original calibration | 105 | 80/80 |
| Held-out examples | 45 | 35/35 |
| Deployment/warning extension | 60 | 35/35 |
| Total | 210 | 150/150 |

Sixty measurements have no reviewed label and are excluded from the match
denominator. Reviewed counts by metric are tool use 55/55, completion 50/50,
and fidelity 45/45. Full repeated scores, labels, hashes, and mismatch/error
counts are in [the result data](./scoring-fix-validation-results.json).

| Previously failing check | Before | After |
|---|---:|---:|
| Omitted warning, fidelity (must fail) | 0.90 | 0.40–0.50 |
| Disclosed warning, fidelity (must pass) | 0.90 | 0.90 |
| Warning disclosed after execution, fidelity (must fail) | 0.40–0.50 | 0.40 |
| Raw approved provisioning handoff, completion (must pass) | 0.60 | 0.90–0.95 |
| Truthful provisioning control, completion (must pass) | 0.60 | 0.95–1.00 |

Unexecuted deployments still fail completion at 0.40–0.50. Wrong arguments,
missing handoffs, and unauthorized writes still fail their reviewed dimensions.
Completion alone still does not establish authorization or warning disclosure;
the separate fidelity/tool-use metrics and deterministic checks remain required.

Validation also passed 264 offline tests, including two targeted regression
tests for matched internal plan evidence and completion-context hashing, plus
Ruff, formatting, targeted type checks, and `git diff --check`.

## Provenance and limits

The replay used unchanged transcripts, schemas, and source hashes for all 14
examples. Scoring context changed only for the six examples belonging to the
approved-deployment golden. Original artifacts, normal result history, rubric
wording, MCP metric implementation, and the previous frozen review protocol
remain byte-for-byte unchanged. No agent or cluster ran.

Raw measurements, reasons, snapshots, summary, and pre-run hashes are local at
`evals/results/judge-calibration/scoring-fix-20261102.*`. Each row records the
source/input hashes, versions, judge endpoint, fixed threshold, and frozen label.
Earlier results remain separate; scores across fidelity versions or changed
scenario contexts must not be pooled.

These are 14 partly correlated examples with labels assigned by Codex before
judging, without independent human review. Five judgings measure judge variance,
not agent reliability. Some passing fidelity scores touch the 0.80 cutoff.
The tests do not verify endpoint readiness or arbitrary free-form claims about
autoscaling and monitoring. The next step is a fresh judged one-model baseline,
then the full matrix once that baseline is understood.
