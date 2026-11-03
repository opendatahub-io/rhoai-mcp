# DeepSeek judge migration and MCP item scoring

**Date:** 2026-11-03 · **Source:** working tree based on `9eedeed`

The user selected EnMaaS DeepSeek to contain evaluation cost and selected
item-based MCP scoring rather than asking a holistic judge for more digits.
Fidelity stays on G-Eval's 0–10 scale, with its existing wording/evidence `v2`.

## Implementation

- `EvalConfig`, the example environment, calibration CLI and evaluation workflow
  default to `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017` through EnMaaS. The
  workflow requires `ENMAAS_KEY`; no automatic paid-model fallback is added.
- MCP metrics are `conversation-v3-items`: six weighted items each for tool
  choice, arguments and completion. Ratings are 0/1/2 or permitted n/a, with
  evidence. Python computes fractions and critical caps. The use metric retains
  the minimum of tool/argument scores, and the full ordered conversation stays
  one task. Both required DeepEval MCP metric subclasses remain.
- Records retain full item scorecards and unrounded floats. Summary/attempt
  tables display four decimals for v3. Reason summaries are assembled in code,
  removing two extra judge generations from a normal three-metric evaluation.
  Cached MCP scores are disabled for the new contract.
- The [scoring guide](../JUDGE-SCORING.md) and the fidelity factory comment link
  to [DeepEval's G-Eval documentation](https://deepeval.com/docs/metrics-llm-evals#rubric)
  and document its 0–10 grade, normalization, and plain-score fallback.

## Verification

**413 offline tests pass**, including real DeepEval sync/async execution,
normal score extraction, evidence round-tripping, invalid scorecard rejection,
critical caps, and visible fractional precision. Targeted type checks pass for
nine changed runtime modules. Ruff and format checks pass on the changed
Python files, with the pre-existing UP042 enum findings in `evals/config.py`
excluded. Workflow YAML, relative documentation links, and diff whitespace pass.

The four-case live pilot completed **12 measurements without errors**. It
produced MCP-use `0.954545…`, missing-handoff completion `0.588235…`, wrong-core-
argument use `0.30`, and silent unauthorized-write use `0.30`. The latter two
are explicit critical caps, not an artificial display grid. Missing-handoff
tool use correctly passed, separating the absent prose from tool arguments.

Fidelity falsely passed missing handoff at `0.80` in the pilot. Keeping its
scale does not remove this judge limitation; completion and deterministic
checks reject that case. The repeated validation below retains such failures.

## Repeated-validation configuration

Fourteen frozen examples × three metrics × five repetitions = **210 measurement
attempts**, all using free EnMaaS DeepSeek. The original labels, transcripts,
tool schemas and workflow contexts are preserved. Context descriptions are
restored in memory from the earlier Sol corpus, rather than mixing today's
changed scenario descriptions into old reviewed examples. Input snapshots now
also retain the new metric version and complete item policy/weights/caps.

Cutoffs stay 0.60 / 0.75 / 0.80, provider-default sampling, eight concurrent
measurements, 90 seconds per request, 240 seconds per metric, and no SDK retries
or automatic reruns. Errors remain unscored. No agent or cluster is run for
these experiments; no GPT judge is called. Native artifacts and ordinary
66-record agent history remain separate and unchanged.

The labels were assigned by Codex without independent human review, and the
14 examples are partly correlated. Repetition measures judge variation, not
independent agent successes or population accuracy. Old `conversation-v2`
scores and new item scores must not be pooled.

Raw pilot and repeated evidence is local under
`evals/results/judge-calibration/deepseek-items-*20261103*`; the experiment-only
driver is `qualify-deepseek-items-20261103.py` in that directory. An initial
sandboxed pilot's 12 connection errors are preserved separately as environment
failures, before network access was approved.

## Completed repeated validation

**210 attempts completed: 191 scores and 19 infrastructure errors** (16 request
timeouts and three HTTP 502s). No measurement was automatically rerun. There
were no malformed scorecards or schema errors. Full arrays, mismatch/error
lists, input/source hashes and policy snapshots are in the
[machine-readable results](./deepseek-item-judge-results.json).

| Metric | Reviewed labels | Scored reviewed labels matched | Reviewed labels unscored | Errors across all 70 measurements |
| --- | ---: | ---: | ---: | ---: |
| MCP use | 55 | 41/41 | 14 | 17 |
| Completion | 50 | 49/49 | 1 | 2 |
| Fidelity | 45 | 41/45 | 0 | 0 |
| Total | 150 | 131/135 | 15 | 19 |

Sixty dimensions have no reviewed label. Four errors affected those unlabelled
dimensions, accounting for the difference between 15 unscored reviewed labels
and 19 total errors. Errors are not zero-quality scores and are not omitted
from coverage counts. Scored MCP judgments have no reviewed false passes or
false failures; this is **90/90 scored reviewed labels**, with 15 additional
reviewed MCP labels unscored. It does not establish a population accuracy rate.

Fidelity's only mismatches were four false passes on the missing-handoff
counterexample, each 0.80 at the inclusive 0.80 cutoff. Its fifth score was
0.70. Completion rejected that example in all five repetitions, scoring
0.50–0.588235…. Correctness of the separate MCP dimension boundary also held:
its four measured tool-use scores passed, with one timeout.

MCP use rejected scored reviewed wrong-core-argument and unauthorized-write
controls. Completion rejected scored reviewed incomplete or contradicted
outcomes. Fidelity rejected all five silent-write measurements and all ten
omitted/late warning measurements. Approved provisioning and warning-disclosed
positive controls passed their scored reviewed dimensions. Missing measurements
remain unknown rather than being treated as additional passing examples.

The live scores demonstrate the requested change in granularity: **25 of 53**
scored MCP-use measurements and **38 of 68** completion measurements fall off
the former 0.05 grid. Examples include MCP use `0.954545…` and completion
`0.911765…` (31/34 points). All **174 returned scorecards** independently
recompute from their stored ratings, weights, point totals and caps.
Fidelity remains on its tenth-point scale. Valid round fractions/caps remain
possible; no artificial digit variation is added.

All **143** original artifact/message/history hashes still match, and normal
agent history remains at **66 records**. All 14 original transcript/context/
schema snapshots match the frozen sources. Changed MCP policy and input hashes
identify the distinct scoring configuration; historical Sol records remain unchanged.

The timeout-heavy tool-use measurements limit coverage under eight-way
concurrency and a 90-second request timeout. Normal wrapper client defaults
differ; this batch does not establish a general service availability rate.
The DeepSeek default is implemented per the user's cost decision. Keep the
deterministic checks and all metrics, preserve the fidelity limitation, and
validate any later timeout/sampling/cutoff change as a separate configuration.
No threshold was selected or retuned from this validation, including the
calibrator's diagnostic threshold suggestions.
