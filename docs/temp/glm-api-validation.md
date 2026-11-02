# GLM API comparison and full Chat validation

**Date:** 2026-11-02 · **Source:** `29266d6` plus frozen diagnostic drivers · **Scope:** EnMaaS GLM, Navigator only

**Both APIs passed the matched final-turn replay. The subsequent Chat
Completions validation passed 1/2 full conversations.** These results do not
establish an API fix for the original handoff omission. The existing
[four-model report](./navigator-serial-model-comparison.md) remains **7/8**,
and its GLM Responses pair remains **1/2**.

The user authorized the API comparison and, if warranted, the two GLM Navigator
goldens. Exactly one replay per API and one fresh attempt per golden were run.
No skill wording, production model spec, judge, cutoff, or timeout was changed.

## 1. Matched final-turn replay

Parent: `4f2a187d74cd`, the failed Responses vague case. Both APIs received the
saved history immediately before its final answer: the same five user inputs,
visible assistant content, tool calls/results, full loaded skill, and prior
reasoning. The eight prior reasoning segments total 60,501 characters.
Responses retains their native reasoning items; Chat receives the identical
text as `reasoning_content`. An offline transport audit verified content
equivalence before either paid request. The failed original final answer was
not included in either request.

Both used the same EnMaaS model and endpoint, 53 saved MCP schemas, unspecified
effort/temperature/output cap/penalties, and non-streaming requests. No MCP
action or judge ran. SDK retries were disabled, and each transport allowed
only one request. Requests and original response bytes were retained locally;
authorization headers were not logged.

| API | Requests | HTTP | Handoff | Hard checks on derived transcript | Seconds |
| --- | ---: | ---: | --- | --- | ---: |
| Responses | 1 | 200 | Complete, correct `0.13` QPS | Pass | 19.814 |
| Chat Completions | 1 | 200 | Complete, correct `0.13` QPS | Pass | 16.455 |

The original omission did not reproduce on either route. This is one sample
per API, not a reliability estimate or latency ranking. It neither proves
nor excludes intermittent model/gateway behavior. These replays are diagnostic
artifacts, not new five-turn agent runs; they did not append ordinary history.
[Replay evidence](./glm-api-replay-results.json).

## 2. Fresh two-golden Chat Completions validation

Chat's successful final-turn replay justified checking its normal discovery,
tool-use, and handoff flow end to end. The temporary diagnostic profile
`enmaas-glm-chat` changes the roster GLM spec's API from `responses` to `chat`.
The production roster is unchanged. Both cases ran sequentially, including
judging, with automatic reruns disabled and sleep prevention enabled.

| Golden | Attempt | Turns | Hard checks | Tool use | Completion | Fidelity | Overall |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| Vague requirements | `e8d402ddd2ae` | 5 | Pass | 0.95 | 0.95 | 0.80 | Pass |
| Happy path | `2aaa97922492` | 5 | Pass | 0.30 | 0.90 | 0.60 | Fail |

Both loaded Navigator, emitted complete handoffs preserving `0.13` QPS, and
passed table completeness and all other hard checks. Four of six judge scores
passed. There were no timeouts, harness/model/judge errors, deployment writes,
or forbidden calls. Fidelity at the vague case's 0.80 cutoff is a pass, not a
claim of flawless adherence.

The happy case failed the tool-use and fidelity cutoffs. Direct inspection
confirms that it called `recommend_model(preferred_gpu_types=[])` before the
customer supplied GPU inventory, without asking for it or obtaining explicit
permission for an unfiltered search. It made three such calls, then four more
recommendation calls after receiving A100-80 inventory. Later correction did
not undo the initial gate violation. The judge also criticized an H100 result
in the A100-filtered phase without the expected explicit fallback sequence;
the original reasons are retained in the evidence.

**A coverage gap is visible:** the existing hard checks pass this case because
they cover workload confirmation, table structure, selected recommendation
agreement, and deployment boundaries, but do not enforce the GPU-inventory
prerequisite across all recommendation calls. The calibrated tool-use/fidelity
metrics caught it. This report does not add or change a checker or regrade the
historical cases.

## 3. Timing and usage

| Chat golden | Model responses | MCP calls | Output tokens | Reasoning tokens | Agent seconds | Scoring seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vague requirements | 9 | 6 | 11,822 | 9,346 | 189.41 | 40.11 | 229.52 |
| Happy path | 11 | 9 | 28,760 | 24,910 | 507.26 | 37.89 | 545.15 |

The happy case took **8 minutes 27 seconds of agent time**, completed all five
turns normally, then was judged. Its extra recommendation work and large
recorded reasoning output help explain the wait; no timeout occurred. Nine MCP
calls comprise two workload calls and seven recommendation calls. Reasoning
tokens are included in output tokens. Total time is agent plus scoring per
conversation, not a model-only latency or billed-cost measurement.

The [Chat comparison report](./glm-chat-model-comparison.md) and
[JSON export](./glm-chat-model-comparison.json) include quality means, measured
coverage, input/cache tokens, responses, turns, tool counts, errors, and timing.
Their means include both the passing and failing cases. One measured whole-case
interval pair confirms zero overlap, including judging and teardown.

For historical context, the original Responses pair had one missing-handoff
failure; its happy case passed. The Chat pair also passed 1/2, with its failure
instead in tool use and fidelity. These are separate cohorts and are not pooled
into a model ranking or a revised full-roster verdict.

## 4. Configuration and Pi.dev parity limits

The model remains `rits/zai-org/glm-5-3` through
`https://api.enmaas.devshift.net/v1`. Skill `6b84c869d2cc`, both scenario hashes,
the shared host instruction, 53 MCP schemas, mock cluster, and tracing-off
configuration remain fixed. All **142 compared runtime/configuration files**
match the original sequential roster's hashes. The frozen full-validation
manifest covers 287 source/configuration/test/driver files.

The judge remains EnMaaS `gpt-6.1-sol`, thresholds **0.60 / 0.75 / 0.80**,
MCP version `conversation-v2`, and fidelity `v2`. Per-turn limits remain
300 seconds, 30 model requests, and 40 tool calls.

This tests **the API route**, not full Pi.dev parity. It does not request Pi's
default medium thinking level, a 65,536-token ceiling, explicit zero penalties,
or streaming, nor reproduce Pi's complete host prompt/tool interface. Those
remain separate variables. The original Codex/Pi catalog files do not configure
the Pydantic AI sweep.

## 5. Conclusion and next review

Neither a deterministic Responses defect nor an overall quality improvement
from switching to Chat was established. Chat can complete both handoffs, but
the full validation exposes a broader workflow violation. Keep the production
roster and skill unchanged on this evidence alone; further configuration
experiments or paid validation require a separate decision.

The next useful offline step is a regression/coverage review for the explicit
GPU-inventory gate, using the saved happy case and successful controls. That
would turn the observed prerequisite violation into a named deterministic
finding without changing any recorded verdict or inventing a successful run.

## Evidence and reproduction

[Full validation records](./glm-chat-validation-results.json) include the two
ordinary records, judge reasons, source/native hashes, paired replay evidence,
and case intervals. All 62 earlier history records remain a byte-identical
prefix; exactly two full-run records were appended, bringing history to 64.
The 156 earlier native files and 39 earlier report/plan files were unchanged
at report verification. Subsequent changes are limited to living plan status
updates and new reports.

```bash
make eval-compare ARGS="--cohort docs/temp/glm-chat-validation-results.json --artifacts-root evals/results --format markdown --output docs/temp/glm-chat-model-comparison.md"
```

Use `--format json` and the JSON output path for machine-readable statistics.
This reads saved records without paid calls. Local drivers and wire payloads
remain under `evals/results/glm_api_pair_20261102.py`,
`evals/results/glm-api-pair-20261102/`, and
`evals/results/run_glm_chat_validation_20261102.py`; the diagnostic profile and
timing plugin are hashed in the frozen configuration. No new default API or
model-family setting was introduced.
