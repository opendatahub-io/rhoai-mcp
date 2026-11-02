# Navigator sweep: four EnMaaS models

**Date:** 2026-11-02 · **Source:** `f766dbf` · **Scope:** Navigator only

All **8 planned conversations were attempted and recorded; 5 passed**. GPT and
GLM passed both goldens, DeepSeek passed one, and Anthropic passed neither under
the frozen evaluation configuration. These are single attempts, not model rankings.
Anthropic's results require the [harness investigation](./anthropic-harness-investigation.md)
before drawing conclusions about its behavior in Claude Code.

| Agent | Golden | Attempt | Turns | Hard checks | MCP use | Completion | Fidelity | Overall |
|---|---|---|---:|---|---:|---:|---:|---|
| GPT | Happy | `c3bc27d9ea42` | 5/5 | Pass | 1.00 | 0.95 | 0.90 | Pass |
| GPT | Vague | `7caa2026faac` | 5/5 | Pass | 1.00 | 0.95 | 0.90 | Pass |
| Anthropic | Happy | `841518974181` | 5/5 | Fail | 0.60 | 0.80 | 0.70 | Fail |
| Anthropic | Vague | `0472b7917e09` | 1/5 | Fail | 1.00 | 0.50 | n/a | Fail: routing |
| GLM | Happy | `b649e936348a` | 5/5 | Pass | 1.00 | 0.90 | 0.90 | Pass |
| GLM | Vague | `5235bc3109f9` | 5/5 | Pass | 1.00 | 0.95 | 1.00 | Pass |
| DeepSeek | Happy | `593efbe1d2f7` | 5/5 | Pass | 0.30 | 0.80 | 0.50 | Fail |
| DeepSeek | Vague | `477165468821` | 5/5 | Pass | 0.80 | 0.80 | 0.90 | Pass |

Happy and Vague are `navigator-happy-path` and `navigator-vague-requirements`.
Thresholds remain use **0.60**, completion **0.75**, fidelity **0.80**. Overall
passing requires routing, hard checks, and every applicable judge metric.

Seven conversations completed all five turns; routing passed in seven and hard
checks in six. There were **23 metric measurements, 19 passing**, with no judge
errors. The missing fidelity score is `n/a`: Anthropic's vague case never loaded
Navigator, so it had no delivered skill body against which to assess fidelity.
There were no timeouts, harness errors, or deployment writes.

## Configuration and execution

| Spec | Provider model | API | Run | Agent seconds | Judge seconds |
|---|---|---|---|---:|---:|
| `enmaas-sol` | `gpt-5.6-sol` | Responses | `e2a176adaafe` | 104.51 | 63.32 |
| `enmaas-opus` | `claude-opus-4-6` | Messages | `bb81ab50582e` | 113.31 | 82.42 |
| `enmaas-glm` | `rits/zai-org/glm-5-3` | Responses | `0d494fc53b7e` | 422.47 | 88.26 |
| `enmaas-deepseek` | `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017` | Responses | `01ccf87c28d3` | 169.75 | 86.55 |

All agents and the fixed `gpt-6.1-sol` judge use EnMaaS. Responses uses
`https://api.enmaas.devshift.net/v1`; AnthropicProvider uses the root
`https://api.enmaas.devshift.net`, with the SDK adding `/v1/messages`.
Agent settings request provider-default sampling and no extra settings, including
no explicit Anthropic thinking configuration. Uniformly omitted settings do not
establish equivalent reasoning behavior across providers.

MCP metrics remain `conversation-v2`, fidelity evidence `v2`. Each turn has a
300-second timeout, 30-request limit, and 40-tool-call limit. Each model ran its
two Navigator goldens once against mock cluster state with both skills available
for discovery, tracing off, pytest `-s --reruns 0`, and `caffeinate -i`. Power logs
show no sleep/wake events during the batches. Agent time totals 810.04 seconds;
judge time totals 320.55 seconds. Currency cost was not returned.

No source, skill, scenario, judge, threshold, or limit changed between batches.
No failing attempt was retried. Native histories, all 53 MCP tool schemas,
structured results, hard checks, judge reasons, and provenance were saved.

## Findings

**Anthropic happy path:** the handoff uses peak QPS `0.13` instead of the
`get_expected_rps` value `0.06666666666666667`. Fidelity also fails: the judge
identified skipped workflow gates, unfiltered recommendations before hardware
permission, and retaining a recommendation after failed tightening. Two planner
calls returned structured errors; `explore_cluster` was outside the skill's
declared allowlist.

Two other hard failures are coupled to formatting: bold row labels such as
`**Model**` fail literal label comparison and prevent selected-slot extraction.
An offline diagnostic removing only row-label bold markers from a derived
transcript resolves those two checks. The QPS failure remains. This diagnostic
does not change official results or the failing fidelity score.

**Anthropic vague requirements:** Claude called `list_use_cases`, presented
available use cases, and asked for intent, users, and constraints. It did not load
Navigator. The checkpoint before the second scripted user message requires the
skill already loaded, so the script stopped after one completed turn. The other
missing downstream outputs follow that early stop. This is a routing-contract
failure with useful clarification behavior, not two complete failed workflows.

**DeepSeek happy path:** every hard check passed, but tool-use and fidelity
failed. Judge findings include recommendations before hardware permission, a
quality floor of 32 instead of 33, unnecessary registry fallback, and retaining
an earlier performance result after tightening failed. There were four structured
tool errors and an allowlist deviation to `list_registered_models`.

**Passing cases:** GPT had no failed MCP returns or allowlist deviations; both
handoffs used correct expected QPS. Its judge noted an omitted explicit offer to
relax constraints for empty slots. GLM passed both complete flows without failed
MCP returns. DeepSeek's vague case passed, with judge caveats about hardware
revalidation, quality tightening, and an omitted available performance candidate.
Passing thresholds do not mean flawless adherence.

## Evidence and next step

[Machine-readable results](./navigator-sweep-results.json) contain the eight full
records and detailed evidence. The [GPT batch report](./navigator-sweep-gpt.md)
is retained as the historical first-batch snapshot. Earlier history bytes, 78
pre-existing native artifact files, and previous JSON reports were verified
unchanged. Live artifacts are local under `evals/results/<run>/<attempt>/`.

The full sweep is complete. Before extending comparative reporting, review the
[Anthropic investigation](./anthropic-harness-investigation.md): missing adaptive
thinking configuration, the catalog/checkpoint contract, and table-label parsing
are separate concerns. Correct them in reviewable steps, retaining this frozen
sweep. Any new live diagnostic must have a distinct configuration and record;
no additional paid run or Navigator-deploy sweep is included in these results.

**Subsequent macro step:** the [Anthropic correction](./anthropic-correction.md)
records the implementation and four additional diagnostic conversations under
two separate configurations. They are outside this frozen eight-case sweep.
