# Evaluation plan completion and free EnMaaS judge assessment

**Date:** 2026-11-03 · **Branch:** `tarilabs-20261031-deepevalpydanticai` · **Reviewed commit:** `9eedeed`

**Subsequent cost decision:** the user requested switching to DeepSeek. The
working tree now defaults to DeepSeek and uses item-based MCP scoring
`conversation-v3-items`; fidelity remains on G-Eval's 0–10 scale. The screening
results below describe the earlier `conversation-v2` experiment. See the
[current scoring contract](../JUDGE-SCORING.md); old scores remain historical.

Most of the planned evaluation infrastructure is complete. The current
Navigator-only sweep is complete; the original broader ambition of evaluating
both skills across the entire model matrix is not. Judge replacement is a
separate experiment from agent-model selection.

## Plan versus implementation

| Work | Assessment | Evidence |
| --- | --- | --- |
| Phase 0: dependencies, native skill discovery, stdio, model qualification | Complete | `pyproject.toml`, `uv.lock`, `evals/agent/models.py`, `evals/agent/runner.py`, [qualification](./enmaas-matrix-qualification.md) |
| Phase 1: expose `get_cluster_resources` by default without duplicate registration | Deferred; not implemented | Tool remains registered only by training; defaults exclude training. Both skills ask for inventory instead. [Known gap](./remaining-work.md#3-phase-1--deferred-by-decision) |
| Phase 2: runner, checkpoints, both skills, four goldens, hard checks | Implemented, with historical live evidence | `evals/skills/goldens.py`, `checks.py`, `agreement.py`; [deployment evidence](./remaining-work.md#42-navigator-deploy-approved--deployment-initiation-verified) |
| Durable native transcripts, call-ID adapter, offline viewer and rescoring | Implemented | `evals/agent/transcript.py`, `evals/deepeval_helpers.py`, `evals/skills/show.py`, `rescore.py` |
| Both required MCP metrics and skill fidelity | Implemented | `evals/metrics/conversation.py` subclasses both stock MCP metrics, retaining their calculations while fixing conversation extraction; `evals/skills/rubrics.py` supplies fidelity |
| Calibration, negative controls, held-out cases, warning/provisioning corrections | Completed on a small corpus | Sol matched 150/150 reviewed dimension/repetition labels across 210 measurements after corrections. [Validation](./scoring-fix-validation.md) |
| Phase 3: current four-model Navigator matrix | Complete | Sequential cohort: 8 conversations, 7 passes; later corrected GLM pair: 2/2. [Sequential comparison](./navigator-serial-model-comparison.md), [GLM validation](./glm-corrected-validation.md) |
| Fair comparison and reporting | Implemented | `evals/reporting/skill_comparison.py` checks cohort compatibility and excludes rescores/duplicates/dry runs; preserves missing values and failures |
| Phase 4: optional tracing | Working spike, polish deferred | `evals/agent/tracing.py`, `make mlflow-up`, `make eval-skills-traced`; session links, dependency pinning and retention remain in [leftovers](./remaining-work.md#6-phase-4-leftovers) |
| EvalHub | Explicit follow-on; not implemented | Parent plan §5 |

The initial referenced review discussed 12 conversations across both skills;
the living plan now specifies eight Navigator conversations. Completing the
latter does not complete a deployment matrix. Historical deployment evidence
verifies mock InferenceService creation and provisioning handoff, not endpoint
readiness. Automated GPU inventory/product mapping and the primary
unconstrained fallback remain incompletely exercised live.

The latest Navigator wording (`88cbf1008ce0`) and mock-subprocess isolation
cleanup have offline coverage, but no new live model validation. The selected
report combining the corrected GLM pair with earlier models is explicitly a
[mixed-configuration comparison](./navigator-selected-model-comparison.md),
not a fresh coherent 8/8 sweep. The parent plan still labels tracing as planned
although the working spike and tests already exist.

Fresh verification on this branch: **389 offline evaluation tests pass** with
`.venv/bin/python -m pytest tests/evals -q`. The sandbox initially blocked the
pytest rerun plugin's local socket; the successful run used the approved
unsandboxed execution. No application source was changed for this assessment.

## Why scores cluster in increments of 0.05

Inspection used the installed **DeepEval 4.3.1**, rather than assuming that
the current online documentation exactly matches this environment.

| Metric | Actual scoring path | Precision implication |
| --- | --- | --- |
| Conversation MCP Use | `min(mean(tool scores), mean(argument scores))`; this harness produces one task for the complete conversation | Each underlying score is an unrestricted float in the prompt/schema. No 0.05 rounding or averaging of the two dimensions. |
| Conversation MCP Task Completion | Mean of task scores; again, one conversation task here | The judge supplies a float from 0 to 1. No imposed 0.05 step. |
| Skill fidelity (`ConversationalGEval`) | Prompt requests an integer from 0 to 10; result is divided by 10 | Normally 0.1 steps in our custom-wrapper path. The schema itself accepts floats, so this is a prompt constraint rather than an integer validator. |

The OpenAI-compatible judge wrapper in `evals/providers/judge.py` returns
parsed JSON or text, and does not implement `generate_raw_response` /
`a_generate_raw_response`. DeepEval's G-Eval code therefore falls back to the
ordinary score, bypassing its token-log-probability weighting. The
[official G-Eval documentation](https://deepeval.com/docs/metrics-llm-evals)
describes that weighting; merely setting `top_logprobs` on the metric cannot
enable it through this wrapper.

The 210 saved Sol measurements in `scoring-fix-20261102.jsonl` confirm the
distinction: completion has 70 scores on the 0.05 grid, fidelity has 70 on the
0.1 grid, and tool use contains **two 0.98 scores**. Reporting only formats
scores for display; the recorder retains their float values.

Thus the MCP clustering is consistent with the judge choosing familiar round
grades and the score anchors' ranges, not a forced 0.05 quantization. A score
of 0.90 is a rubric grade, not a calibrated 90% success probability. Changing
the prompt to request hundredths would not establish hundredth-point validity.
Keep hard checks and label-level false-pass/false-failure evidence central;
use more independent trajectories to improve confidence. Adding log-probability
support would be a separate, versioned scoring change needing recalibration.
Log-probability weighting smooths grades; it does not by itself validate the
meaning of a 0.91-versus-0.92 difference. Criterion-level pass/fail evidence
would offer a more interpretable aggregate, but would also change the scoring
contract and require its own validation.

## Judge-switch experiment

GLM (`rits/zai-org/glm-5-3`) and DeepSeek
(`rits/deepseek-ai/DeepSeek-V4.1-Flash-1017`) cost the user zero through
EnMaaS. That user-specific fact makes them the primary candidates; public
OpenAI pricing does not describe this choice. Agent performance in the earlier
sweeps does not establish judge performance.

GPT-4o-mini was already compared and missed 29/115 reviewed labels, including
28 false passes, under the earlier evidence version. That result is not a
same-input numerical comparator for a new v2 experiment, but explains why
price alone is insufficient. [Prior comparison](./gpt-4o-mini-judge-comparison.md).

The current thresholds remain 0.60 / 0.75 / 0.80, with MCP
`conversation-v2`, fidelity evidence `v2`, and unchanged rubric wording.
Experiments write separate calibration files and do not append ordinary
agent-run history. No new GPT judge calls or agent/cluster runs are needed.

The first three-case pilot used the current context. Its Navigator description
differs from the earlier Sol corpus, so it is a connectivity and behavior
screen, not an exact input comparison. An experiment-only driver restores the
14 original scoring contexts in memory, verifies unchanged sources,
transcripts, schemas and versions before calling a judge, and checks the
complete output snapshots against Sol's saved input snapshots afterward.
It does not edit golden definitions or lower any threshold.

The calibrator explicitly applies a 90-second per-request timeout, disables
SDK retries, and bounds each whole metric at 240 seconds. Timeouts below
describe that experimental setup; the normal provider wrapper has different
client defaults. They are not evidence of model refusal or invalid JSON, and
must remain unscored rather than becoming quality scores of zero.

The reviewed labels were assigned by Codex, without independent human review.
Several examples share an agent trajectory, and a repeat measures judge
variation, not an additional independent agent success. This corpus is useful
for rejecting known mistakes; it cannot establish a population accuracy rate.

The normal scoring path makes six logical judge generations for a successfully
routed one-task conversation: tool choice, arguments, their final reason,
completion, its final reason, and fidelity. Calibration omits the two final
reason-summary calls but retains the component reasons, giving four logical
generations per three-metric example. SDK retries, errors and caching can
change the actual request count. Disabling only the final summaries is a
possible separate cost reduction, not a measured one-third billing saving:
prompts have different sizes, and the wrapper does not retain judge token
usage or invoices. Summaries should be replaced with retained component
reasons if this is implemented.

Switching the judge itself needs no new provider integration. Once a candidate
is accepted, the existing settings are sufficient:

```bash
export RHOAI_EVAL_EVAL_PROVIDER=enmaas
export RHOAI_EVAL_EVAL_MODEL=rits/zai-org/glm-5-3
# Or: rits/deepseek-ai/DeepSeek-V4.1-Flash-1017
```

`ENMAAS_KEY` supplies credentials. These are provider model IDs, not agent-spec
IDs such as `enmaas-glm`; agent selection is independent. A model switch creates
a new scoring configuration, so preserve historical records and do not pool
new scores into old Sol cohorts. Keep the deterministic checks, both MCP
metrics, and fidelity. Do not add an automatic paid-Sol fallback to a free run.

## Completed screening result and recommendation

The exact-input screen completed **84 measurements**: 14 cases × three
metrics × two free models × one repetition. Each model has 30 reviewed labels
and 12 unlabelled dimensions. The pre-existing Sol baseline has five
repetitions, rather than one. [Machine-readable results](./free-judge-screen-results.json).

| Judge | Measurements | Reviewed labels matched | Reviewed labels unscored | All measurement errors | Mean metric seconds, including errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical EnMaaS Sol | 210 | 150/150 | 0 | 0 | 9.04 |
| EnMaaS DeepSeek | 42 | 29/30 | 0 | 0 | 31.37 |
| EnMaaS GLM | 42 | 22/30 | 8 | 14 | 74.42 |

The Sol inputs and free-model inputs match exactly, including full transcript,
schemas, scoring context and source hashes. The latency figures come from
different runs/provider conditions; this is not a controlled general speed
benchmark. GLM matched all **22 scored reviewed labels**, but eight reviewed
measurements timed out. Six further timeouts occurred on unlabelled dimensions.
Missing scores are not counted as matches or turned into quality scores.

DeepSeek's one mismatch was a **false failure on tool use** for
`navigator-missing-handoff` (0.55, cutoff 0.60). Its argument reason explicitly
penalized the absent final handoff, alongside other minor defects, even though
the tool-use anchors assign final-prose omissions to completion/fidelity.
Its completion (0.50) and fidelity (0.70) correctly failed that example.
There were no reviewed false passes in this exact-input screen.

In the separate three-case pilot, DeepSeek instead **falsely passed fidelity**
for the missing-handoff example (0.80, inclusive cutoff 0.80). That finding is
retained rather than erased by the later passing screen. The pilot also had
one DeepSeek timeout and three GLM timeouts. Pilot context differences mean
its label counts are not pooled with the exact-input screen.

Both candidates rejected the explicit unauthorized-write and silent-write
controls in their reviewed tool-use/fidelity dimensions. Both correctly
separated the warning-disclosed control from omitted/late warnings in fidelity.
DeepSeek also rejected the wrong-argument control and unexecuted-deployment
completion cases. GLM's wrong-argument tool-use measurement timed out, so its
coverage remains incomplete despite the good scored results.

**Recommendation: qualify DeepSeek first as the free replacement.** It is the
more practical candidate under the tested settings, with complete measurement
coverage and one reviewed-label disagreement. GLM remains a candidate if longer
timeouts or a separately qualified reasoning configuration are acceptable;
this run does not provide a complete GLM calibration.

Before making DeepSeek the global default, repeat the exact frozen corpus
five times at fixed cutoffs and inspect dimension-boundary mistakes, the
missing-handoff fidelity result, and near-cutoff variability. If its grading
scale needs different thresholds, treat that as a new calibration with fresh
validation examples; do not retune on these results and call them held out.
The current default remains Sol. Routine deterministic replay can continue
without judge calls, with its existing explicit unjudged status.

The judge experiments ran no agents or MCP servers. The offline test suite
did exercise a guarded mock MCP subprocess with external network access
disabled. No live cluster or GPT judge was called. All **149**
hashed source/native-evidence/history files remained unchanged; normal history
still contains **66 records**. An initial sandboxed pilot produced 18 connection
failures before approved network access; those are separately preserved as
environment failures and excluded from model-quality comparisons.

Raw measurements, reasons, snapshots, provenance, and the experiment-only
driver remain local under `evals/results/judge-calibration/`. To reproduce the
exact-input screen with a fresh output name:

```bash
.venv/bin/python evals/results/judge-calibration/replay-sol-inputs-20261103.py \
  --models rits/zai-org/glm-5-3,rits/deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --repeats 1 --jobs 8 \
  --output evals/results/judge-calibration/free-judge-frozen-screen-repeat.jsonl
```

The recommended repeated qualification changes `--models` to DeepSeek alone
and `--repeats` to `5`, with another fresh output name. It is not scheduled or
run as part of this screening assessment.
