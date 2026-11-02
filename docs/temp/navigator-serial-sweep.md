# Navigator sequential sweep

**Date:** 2026-11-02 · **Source:** `f0bd975` plus the frozen working-tree diff and per-file hashes · **Scope:** Navigator only

**8/8 conversations recorded; 7/8 passed.** All four model families ran through EnMaaS, using the clarified Navigator skill `6b84c869d2cc`. The earlier 7/8 cohort and its comparison are separate historical evidence.

The [comparison report](./navigator-serial-model-comparison.md) highlights tool use, completion, fitness (skill fidelity), turns, model responses, output/input/reasoning/cache tokens, MCP calls/errors, and agent/scoring/total time. Its [JSON export](./navigator-serial-model-comparison.json) preserves machine-readable statistics and per-attempt records. The [sweep evidence](./navigator-serial-sweep-results.json) contains the frozen configuration, native hashes, all case intervals, and verification results.

## Scheduling verification

**Zero overlapping case intervals and zero overlapping model batches.** Eight whole-test intervals, including setup, agent execution, judging, and teardown, were measured on the same host monotonic clock. All 28 pairs were checked. Each model's pytest process exited before the next started; automatic reruns were disabled. There were 0 sleep/wake events in the sweep window; each batch ran under `caffeinate -i`.

This removes the GLM/DeepSeek overlap introduced by the earlier runner. Timing still describes one attempt per model/golden under shared EnMaaS load and natural caching. It does not establish a statistical latency ranking. Agent time excludes judging; total time is agent plus scoring per conversation. Different tokenizers and reasoning behavior remain part of the configured agents.

| Agent | Run | Passed | Agent seconds | Scoring seconds | Batch start (UTC) | Batch end (UTC) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| enmaas-opus | `c1aed9dbd1c1` | 2/2 | 228.50 | 78.52 | 2026-11-02T16:00:15.204882+00:00 | 2026-11-02T16:05:24.537860+00:00 |
| enmaas-sol | `6427f23f8fe9` | 2/2 | 100.87 | 65.72 | 2026-11-02T16:05:24.577828+00:00 | 2026-11-02T16:08:13.693542+00:00 |
| enmaas-glm | `934ca554d04f` | 1/2 | 567.73 | 77.72 | 2026-11-02T16:08:13.733208+00:00 | 2026-11-02T16:19:01.812910+00:00 |
| enmaas-deepseek | `b77ef7e8878d` | 2/2 | 155.92 | 67.83 | 2026-11-02T16:19:01.852582+00:00 | 2026-11-02T16:22:48.088244+00:00 |

## Results and configuration

Routing passed 8/8; 8/8 completed five turns. Hard checks passed 7/8. Judge measurements: 23/24 passing; missing metric entries: 0. Timeouts: 0; model failures: 0; harness errors: 0; judge-error cases: 0; deployment writes: 0.

Both goldens retain hashes `eaa225a67e99` and `f1f282812e97`. GPT, GLM, and DeepSeek use Responses; Opus uses native Messages with adaptive thinking. Every model leaves temperature unspecified. The judge remains EnMaaS `gpt-6.1-sol`, with cutoffs 0.60/0.75/0.80 and versions `conversation-v2`/`v2`. Limits remain 300 seconds, 30 model requests, and 40 tool calls per scripted turn. The mock cluster, discovery policy, 53-tool catalog, and tracing-off configuration remain fixed. DeepEval scratch storage was fresh for this sweep and shared sequentially by all four batches; provider cache state was not reset.

## Failures

GLM's vague case (`4f2a187d74cd`) acknowledged the selected balanced model and
told the user to paste "the block above," but emitted no handoff block. Direct
inspection of the saved final output confirms the omission. Three hard checks
failed: handoff presence, workload agreement, and selection agreement. Task
completion scored **0.50**, below the 0.75 cutoff. Tool use and skill fidelity
both scored 0.80. Its happy-path case passed.

**Table completeness passed in all eight cases**, including both DeepSeek
conversations with the clarified Cluster fit placeholders. The earlier 7/8
cohort failed on DeepSeek's empty cells; this 7/8 cohort failed on GLM's missing
handoff. Keep their identities and verdicts separate. One attempt per cell
does not establish a change in reliability or isolate the effect of either
the template clarification or scheduling.

## Provenance and reproduction

The freeze records 286 source/config/test/runner files, the working-tree diff, package versions, model profiles, and scoring settings. At report verification, 120 earlier native JSON files and 33 earlier report/plan files were unchanged. The original 54 history records remain a byte-identical prefix; exactly eight new records were appended. Plan status updates after this verification are documentation changes, not changes to the historical reports or native evidence.

```bash
make eval-compare ARGS="--cohort docs/temp/navigator-serial-sweep-results.json --artifacts-root evals/results --format markdown --output docs/temp/navigator-serial-model-comparison.md"
```

Use `--format json` and the JSON output path to regenerate the machine-readable statistics. This command reads saved evidence without model or judge calls; the scheduling verification attached by the report builder remains in the separate sweep evidence. Paid runs were launched by the frozen local script `evals/results/run_navigator_serial_sweep_20261102.py`; logs, JUnit reports, per-case intervals, and sleep assertions remain under the matching `evals/results/navigator-serial-sweep-20261102-*` paths.
