# Navigator sweep, batch 1: EnMaaS GPT

**Date:** 2026-11-02 · **Source:** `f766dbf` · **Run:** `e2a176adaafe`

Both Navigator goldens passed on **EnMaaS `gpt-5.6-sol`**. Each completed all five
scripted turns, passed routing and every hard check, and passed all three judge
metrics. This completes **2 of the planned 8 conversations**; Anthropic, GLM,
and DeepSeek remain for separate two-case batches. Historical GLM attempts are
not counted as fresh records in this sweep.

| Golden | Attempt | Agent seconds | MCP use | Completion | Navigator fidelity | Result |
|---|---|---:|---:|---:|---:|---|
| `navigator-happy-path` | `c3bc27d9ea42` | 53.99 | 1.00 | 0.95 | 0.90 | Pass |
| `navigator-vague-requirements` | `7caa2026faac` | 50.52 | 1.00 | 0.95 | 0.90 | Pass |

Pytest reported **2 passed**, 3 known skill-frontmatter warnings, in **168.22
seconds**. Recorded agent time totals 104.51 seconds; judge time totals 63.32
seconds. All six metric measurements completed without errors. No model or
harness failure, timeout, automatic rerun, or deployment write occurred.

## Fixed configuration

- Agent spec: `enmaas-sol`, `gpt-5.6-sol`, Responses API at
  `https://api.enmaas.devshift.net/v1`, provider-default sampling.
- Judge: EnMaaS `gpt-6.1-sol` at the same endpoint.
- Thresholds: use 0.60, completion 0.75, fidelity 0.80.
- Metric versions: MCP `conversation-v2`, fidelity `v2`.
- Limits: 300 seconds, 30 requests, 40 tool calls per scripted turn.
- Target: `evals/skills/test_navigator.py`; mock MCP cluster state, tracing off,
  `-s`, `--reruns 0`, and `caffeinate -i` for idle-sleep prevention.

No source, skill, golden, judge, threshold, or limit was changed for the batch.
Scenario and skill hashes match the frozen configuration. Native histories,
MCP schemas, judge reasons, and result records were saved before assertions.
The active sleep-prevention assertion was captured; the post-run power log
contains no sleep/wake events during the recorded interval.

## Findings and limits

Both handoffs use the correct expected QPS, `0.06666666666666667`, preserve the
confirmed workload and balanced selection, and identify
`meta-llama/llama-3.1-8b-instruct`. Each trajectory has five MCP calls, with no
tool errors or calls outside the skill's declared allowlist.

The judge noted the same minor omission in both cases: the assistant explained
empty Cost and Quality slots but did not explicitly offer to relax constraints.
Completion and fidelity still passed at 0.95 and 0.90. This finding is recorded
for review; no prompt tuning or retry was used to improve the scores.

The MCP server used mock state. The bundled planner nevertheless made optional
external node probes, received HTTP 401, and skipped GPU auto-detection; these
were not failed MCP tool returns. GPU inventory came from the scripted customer
reply. Automated GPU discovery remains unexercised.

There is one attempt per golden, so this batch is not a reliability estimate or
a cross-model ranking. Currency cost was not returned by the evaluation output
and is not estimated here.

[Machine-readable results](./navigator-sweep-gpt-results.json) include full result
records, handoffs, timing and sleep evidence, settings, source/artifact hashes,
and the provenance capture note. Earlier history bytes, all 74 existing native
artifact files, and previous JSON reports were verified unchanged.

The next reviewable batch is the same two Navigator goldens on `enmaas-opus`
(`claude-opus-4-6`, native Messages), keeping the current judge, scenarios,
thresholds, limits, and no-retry policy fixed. That batch has not started.
