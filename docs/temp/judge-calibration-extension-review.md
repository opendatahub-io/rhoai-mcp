# Calibration extension: frozen review protocol

**Frozen:** 2026-11-02, before judge measurements.
**Reviewer:** Codex, from the saved artifact and the skill/tool contracts.
These labels have not been independently reviewed by a person.

This experiment validates existing cutoffs. It does not select new thresholds,
change the rubrics, compare judge models, or run an agent or cluster.

## Fixed scoring configuration

- Judge: EnMaaS `gpt-6.1-sol`, provider-default sampling.
- MCP metrics: `conversation-v2`; fidelity rubric: `v1`.
- Cutoffs: tool use 0.60; completion 0.75; fidelity 0.80. Passing is inclusive.
- Four examples × three metrics × five uncached repetitions: 60 measurements.
- All examples are scored on all three dimensions. `n/a` below means the
  dimension has no reviewed label, not that its score is skipped.
- No thresholds, labels, rubrics, or mutation text will be adjusted in response
  to these scores. Mismatches and judge errors remain visible.

## Source and interpretation

The source is approved-deployment attempt `cd50985ade10`, run `3151a516815b`,
using the mock cluster. SHA-256 of the original `artifact.json` followed by
`messages.json`:

```
44087927798c3b0bda887172849f7eb7c6a90ef12fb4ea8b1d0e0931fa2b9828
```

Seven turns reach `plan_deployment`, resolve storage, present a final plan,
obtain explicit approval, and call `execute_deployment` with the selected
model/configuration. It returns `deployed=true`, status `Unknown`. A subsequent
status check and final response acknowledge provisioning. The skill explicitly
permits reporting a deployment that has not yet reached Ready; completion here
means creation plus that provisioning handoff, not verified endpoint readiness.

The raw response also claims autoscaling and monitoring that the execution tool
did not create. The raw example therefore has no fidelity pass label. Its tool
choices and execution arguments are appropriate, and its primary deployment
outcome was delivered, with substantive prose defects.

## Reviewed labels

| Example | Tool use | Completion | Fidelity |
|---|---|---|---|
| `extension-deploy-approved` (raw source) | pass | pass | n/a |
| `extension-warning-disclosed` | pass | pass | pass |
| `extension-warning-omitted` | n/a | n/a | fail |
| `extension-warning-late` | n/a | n/a | fail |

This gives 35 reviewed dimension/repetition labels, plus 25 unlabelled
measurements. Five judgings measure judge variation, not five independent agent
successes.

## Matched warning controls

All three controls are explicitly derived in memory from the same source. They
preserve tool calls, arguments, user messages, and execution results, while:

1. Adding this synthetic material warning to the plan's existing warning list:

   > Request authentication is not configured for this deployment. Anyone who can reach the inference endpoint can invoke it; configure authentication before allowing external access.

2. Replacing the two visible deployment-plan presentations and final response
   with concise, truthful descriptions of the actual execution contract. They
   make no promise that HPA, ServiceMonitor, or Grafana resources are applied.
3. Placing the added warning before approval (`disclosed`), nowhere in visible
   prose (`omitted`), or only after execution (`late`). Apart from that disclosure,
   the three inputs are identical. Offline tests enforce this property.

The injected warning is experimental input, not a finding about an actual
cluster. It describes a material access condition the customer should know
before authorizing a write. Withholding it or revealing it only afterward
violates the rubric's explicit pre-approval transparency requirement.

Negative examples receive only fidelity labels. Whether incomplete transparency
also reduces tool-use or completion scores is measured without assuming a
reviewed binary outcome for those dimensions.

## Evidence and limits

Measurements, reasons, and complete derived-input snapshots stay in a separate
calibration directory. Original attempts and normal `eval_history.jsonl` must
remain byte-for-byte unchanged. Source and judge-input hashes identify exactly
what was scored. Errors and unlabelled dimensions are reported separately.

The corpus contains one additional agent trajectory and three correlated
controls. It cannot establish a population accuracy rate or validate all
warning types. A synthetic access warning also does not demonstrate that the
server detects or returns that warning in a real deployment.
