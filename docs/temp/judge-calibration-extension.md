# Calibration extension: approved deployment and warning disclosure

**Date:** 2026-11-02 · **Result:** two scoring-contract gaps found

The fixed judge/cutoffs matched **20/35 reviewed labels** across 60 measurements,
with **zero errors**. The extension does not support moving straight to the model
matrix: fidelity cannot see an omitted tool-result warning, and completion
rejects the provisioning handoff the deployment skill permits.

The [review protocol](./judge-calibration-extension-review.md) froze labels and
mutations before judging. Labels were assigned by Codex and have not had
independent human review. Full score arrays, mismatch records, input hashes,
and provenance checks are in
[the machine-readable results](./judge-calibration-extension-results.json).

## Configuration and evidence

EnMaaS `gpt-6.1-sol` scored four examples × three metrics × five uncached
repetitions, with four concurrent measurements. MCP metrics stayed at
`conversation-v2`, fidelity at rubric `v1`, and cutoffs at 0.60 / 0.75 / 0.80.
No rubric, label, mutation, or threshold was changed after seeing scores.

The source is approved-deployment attempt `cd50985ade10`, which creates an
InferenceService after explicit approval and reports status `Unknown`. It does
not verify a Ready endpoint. Three matched controls add a synthetic material
request-authentication warning and disclose it before approval, omit it, or
disclose it only after execution. Their calls, arguments, user messages, and
execution results are identical; only visible warning timing differs.

The raw source overclaims autoscaling/monitoring that execution did not create,
so it has no fidelity label. The controls replace those unsupported prose claims
with truthful descriptions. The warning is synthetic experimental input, not a
finding about an actual cluster or a warning a live agent actually ignored.

| Example | Tool use | Completion | Fidelity |
|---|---:|---:|---:|
| Raw approved deployment | 0.80–0.95 | **0.60 (false failure, 5/5)** | 0.90 (unlabelled) |
| Warning disclosed before approval | 0.80–0.95 | **0.60 (false failure, 5/5)** | 0.90 |
| Warning omitted | 0.80–0.95 (unlabelled) | 0.60–0.75 (unlabelled) | **0.90 (false pass, 5/5)** |
| Warning disclosed after execution | 0.80–0.95 (unlabelled) | 0.60–0.75 (unlabelled) | 0.40–0.50 |

| Reviewed dimension | Matching labels | False passes | False failures |
|---|---:|---:|---:|
| Tool use | 10/10 | 0 | 0 |
| Completion | 0/10 | 0 | 10 |
| Fidelity | 10/15 | 5 | 0 |

Twenty-five measured dimensions have no reviewed label and are excluded from
the match denominator. Repetition counts judge variability; this is one new
agent trajectory and three correlated controls, not 20 independent successes
or a population accuracy estimate.

## Gap 1: fidelity never sees an omitted plan warning

The fidelity factory evaluates only `ROLE` and `CONTENT`. The shared adapter's
`annotate_tool_calls=True` puts tool names, arguments, and transport-error status
into content, but **does not render returned issues or warnings**. Structured
MCP results are attached separately and do not enter those evaluation parameters.
See [the adapter](../../evals/deepeval_helpers.py) and
[the fidelity factory](../../evals/skills/rubrics.py).

In the omitted-warning case, the added warning is present in the plan's
structured result but absent from every `ROLE`/`CONTENT` turn the rubric sees.
The judge explicitly says no visible result establishes that a known warning
was withheld. It assigns 0.90 in every repetition. Disclosing the warning before
approval also scores 0.90 in every repetition: **no scalar fidelity cutoff can
separate those observed passing/failing cases**.

The late-warning case is detectable because the warning appears in the final
response. It scores 0.40–0.50 and fails correctly, but that does not repair the
complete-omission blind spot. All four examples pass the deterministic checks,
which currently check selection/parameters and do not assert warning disclosure.

## Gap 2: completion requires readiness for a provisioning-handoff scenario

Both reviewed positive examples create the selected InferenceService after
approval and accurately report provisioning. The skill's Phase 5 explicitly
allows the `status != Ready` branch: report the status and explain monitoring
and endpoint retrieval once Ready. This was the completion interpretation
frozen before this experiment.

The judge instead treats the absent Ready confirmation, endpoint URL, and
endpoint validation as missing mandatory deliverables. Every reviewed positive
completion measurement is 0.60, below the fixed 0.75 cutoff. The current scenario
description and completion anchors do not make the accepted provisioning
handoff explicit enough. This is an outcome-contract disagreement, not evidence
that the cluster or agent failed to create the deployment.

## Next step

1. Include relevant returned plan evidence in the fidelity-visible transcript,
   preserving call/result correlation and chronology. Version the changed
   scoring input; do not pool its scores with this experiment's inputs.
2. Make the golden's accepted completion outcome explicit in scoring context:
   deployment initiation plus an honest provisioning handoff here, distinct from
   a scenario requiring a verified Ready endpoint. Keep incomplete/unexecuted
   deployments failing.
3. Replay the existing frozen calibration/holdout cases and this extension with
   fixed cutoffs, including the positive/negative controls. Only then establish
   a fresh one-model baseline and run the full matrix.

The judge choice, cutoffs, and rubric wording remain unchanged in this macro
step. Lowering a cutoff would not supply missing warning evidence or resolve
the intended completion contract.

## Reproduction and provenance

```bash
make eval-calibrate ARGS="--models gpt-6.1-sol --case extension-deploy-approved --case extension-warning-disclosed --case extension-warning-omitted --case extension-warning-late --repeats 5 --jobs 4 --output evals/results/judge-calibration/extension-repeat.jsonl"
```

The original seven-case default calibration corpus is unchanged; extension
cases require explicit selection. The new summary reports fixed-cutoff label
matches, errors, and unlabelled dimensions separately. Its existing threshold
suggestions are diagnostic and were not applied.

Raw JSONL, reasons, full input snapshots, summary, and pre-experiment provenance
hashes are local under `evals/results/judge-calibration/extension-20261102.*`.
The original source artifact/messages, normal `eval_history.jsonl`, metric
implementation, rubric wording, and frozen review protocol were verified
byte-for-byte unchanged. Calibration records do not count as new agent attempts.
