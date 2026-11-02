# Navigator updated sweep

**Date:** 2026-11-02 · **Source:** `e59d9e8` · **Scope:** Navigator only

**8/8 conversations recorded; 7/8 recorded attempts passed.** The approved eight-conversation sweep is complete.

The approved order is Anthropic first, then GPT, GLM, and DeepSeek after reviewing that first pair. GLM and DeepSeek overlap at the user's request. Anthropic counts toward the eight-case total. Every model receives the same shared discovery instructions and corrected Navigator handoff contract; model-specific API/settings are frozen below. Historical sweep and diagnostic records are excluded from these totals.

| Agent | Golden | Attempt | Turns | Hard checks | QPS | Use | Completion | Fidelity | Overall |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| Anthropic | Vague | `6293513c587d` | 5/5 | Pass | 0.13 | 0.90 | 0.95 | 0.90 | Pass |
| Anthropic | Happy | `9e84339148e4` | 5/5 | Pass | 0.13 | 0.80 | 0.90 | 0.90 | Pass |
| GPT | Vague | `2b4f92693ff5` | 5/5 | Pass | 0.13 | 1.00 | 0.90 | 0.90 | Pass |
| GPT | Happy | `e6939d8fabba` | 5/5 | Pass | 0.13 | 0.95 | 0.95 | 0.90 | Pass |
| GLM | Vague | `1e271f5c8836` | 5/5 | Pass | 0.13 | 0.95 | 0.90 | 0.80 | Pass |
| GLM | Happy | `02539ef5343e` | 5/5 | Pass | 0.13 | 1.00 | 0.90 | 0.80 | Pass |
| DeepSeek | Vague | `1dd60008ed20` | 5/5 | Fail | 0.13 | 1.00 | 0.90 | 0.90 | Fail |
| DeepSeek | Happy | `fae31aebd41b` | 5/5 | Pass | 0.13 | 0.80 | 0.90 | 0.90 | Pass |

Routing passed in 8/8 attempts; 8 completed all five turns. There are 24 judge measurements, 24 passing. Timeouts: 0; harness errors: 0; deployment writes: 0.

## Batches

| Agent | Run | Passed | Agent seconds | Judge seconds | Pytest seconds |
|---|---|---:|---:|---:|---:|
| Anthropic | `376e1d3e3631` | 2/2 | 188.55 | 73.28 | 262.19 |
| GPT | `355040b39529` | 2/2 | 108.91 | 74.77 | 183.99 |
| GLM | `e81dcbcfbb48` | 2/2 | 564.73 | 81.59 | 646.65 |
| DeepSeek | `f7e0a4db9eb2` | 1/2 | 89.97 | 59.63 | 150.09 |

## Configuration

| Spec | EnMaaS model | API | Extra agent settings |
|---|---|---|---|
| `enmaas-sol` | `gpt-5.6-sol` | responses | `{}` |
| `enmaas-opus` | `claude-opus-4-6` | messages | `{"anthropic_thinking":{"type":"adaptive"}}` |
| `enmaas-glm` | `rits/zai-org/glm-5-3` | responses | `{}` |
| `enmaas-deepseek` | `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017` | responses | `{}` |

Responses uses `https://api.enmaas.devshift.net/v1`. AnthropicProvider uses `https://api.enmaas.devshift.net`; its SDK adds `/v1/messages`. Sampling is provider-default with no requested temperature. Opus explicitly uses adaptive thinking; effort is unspecified. These settings do not establish equal reasoning or sampling behavior across providers.

The judge remains EnMaaS `gpt-6.1-sol`, cutoffs use **0.60**, completion **0.75**, fidelity **0.80**. MCP metric version is `conversation-v2`, fidelity `v2`. Each scripted turn has limits of 300 seconds, 30 requests, and 40 tool calls. Runs use mock cluster state, both skills in the discovery catalog, tracing off, pytest `-s --reruns 0`, and `caffeinate -i`.

| Identity | Frozen hash |
|---|---|
| Navigator skill | `25392e419e13` |
| Happy scenario | `eaa225a67e99` |
| Vague scenario | `f1f282812e97` |

## Findings

Anthropic pair passed under the corrected contract.

Every case loaded Navigator, received the corrected capacity instruction, and handed off the selected 0.13 QPS. All cases used the same 53 MCP schemas. GLM fidelity landed on the 0.80 cutoff in both cases; passing is not a claim of flawless adherence or a reliability estimate.

**GPT / navigator-vague-requirements:** recorded 1 tool error return(s). The GPT vague case includes a Planner HTTP 502 return; the conversation still completed and passed. The original return remains in the evidence.

**DeepSeek / navigator-vague-requirements:** outcome `evaluated`. Failed hard checks: table_cells_complete ('Cluster fit': empty cell(s) at column [2, 3, 4]). Full reasons and tool evidence are in the machine-readable report.

The empty Cluster fit cells reproduce the skill-template ambiguity already listed in [remaining work](./remaining-work.md): the template supplies one example value and leaves three cells empty. The strict table check correctly retains this failure. No template change or rerun was made during the frozen sweep.

## Evidence and limits

The pre-run manifest freezes 280 source/test/configuration files, all four agent profiles, scenario/skill identities, dependency versions, judge, thresholds, and limits. Completed batch intervals have sleep/wake logs and sleep-prevention assertions. No code, prompt, threshold, or limit was tuned between completed batches; every conversation has one attempt.

GLM and DeepSeek were scheduled concurrently at the user's request. DeepSeek uses a separate DeepEval scratch/results directory so judge processes cannot overwrite shared temporary files. Scoring settings are unchanged. Batch intervals record the overlap; timing includes concurrent load and is not a controlled latency comparison.

[Machine-readable results](./navigator-resweep-results.json) include full normal records, hard checks and judge reasons, delivered-instruction checks, tool errors and allowlist observations, handoffs, timing, source/artifact hashes, and pending cells. Earlier history bytes, native artifacts, and JSON reports were verified unchanged. Artifacts and full logs remain locally under `evals/results/`.

One attempt per cell is a smoke comparison, not a reliability estimate or ranking. The MCP state is mocked, and optional external planner node probes may be skipped. Currency cost is unavailable. Navigator-deploy goldens are excluded. Missing metric values remain `n/a`. The earlier [sweep](./navigator-sweep.md) and [handoff diagnostic replay](./navigator-handoff-capacity-correction.md) retain their original instructions/configurations and historical verdicts.
