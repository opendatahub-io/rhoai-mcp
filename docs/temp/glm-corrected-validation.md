# GLM Navigator validation under the committed corrections

**Date:** 2026-11-02 · **Commit:** `44d6774` · **Run:** `8fa71ecb03e2`

**Both Navigator goldens passed overall**, including the new hardware-ordering
check, table completeness, handoff agreement, and all six judged metrics. Each
completed five customer turns. The earlier missing handoff and premature
recommendations did not recur in this pair. This is one attempt per golden,
not a reliability estimate.

The two cases ran sequentially through EnMaaS using the production
`enmaas-glm` Responses profile. No automatic reruns or deployment goldens ran.
Pytest reported **2 passed, 3 warnings in 444.67 seconds**. The warnings concern
the existing unsupported `allowed-tools` skill frontmatter.

## Quality

| Golden | Attempt | Hard checks | Tool use | Completion | Fitness / fidelity | Overall |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Vague requirements | `fdf292edefce` | Pass | 0.95 | 0.90 | 0.80 | Pass |
| Happy path | `ad0ca6ae4e95` | Pass | 0.80 | 0.95 | 1.00 | Pass |

Mean scores are **0.875 / 0.925 / 0.900**, with complete measurement coverage.
The unchanged thresholds are **0.60 / 0.75 / 0.80**. Vague fidelity is exactly
at its passing cutoff.

Both conversations loaded Navigator, waited for workload confirmation, then
asked for and received A100-80 inventory before recommending. Each first
recommendation returned four A100-80 configurations. Both selected Balanced
and emitted all eleven handoff fields, preserving the selected tool response's
`expected_qps=0.13`. There were no failed MCP calls, model/harness errors,
timeouts, forbidden calls, deployment writes, or errored metrics.

## GPU filtering and remaining coverage

Each case made one initial constrained recommendation and three tighter
duplicate-resolution calls. All used `preferred_gpu_types=["A100-80"]`.
The tighter cost, latency, and quality searches returned empty recommendation
maps; GLM displayed the affected slots as `—` and retained the valid Balanced
recommendation. In particular, the `ttft_max_ms=44` search returned no matches
instead of the H100 results seen before the adapter correction. No returned
slot violated its GPU filter.

Neither initial call missed on cluster hardware, so **the primary
unconstrained fallback was not exercised**. The tighter duplicate-resolution
branch explicitly permits empty slots. These runs therefore verify the
inventory gate, constrained selection, empty duplicate results, table, and
handoff; they do not establish live coverage of every fallback branch. The
explicit unconstrained control remains covered by the
[offline correction evidence](./glm-root-cause-correction.md).

## Usage and time

| Golden | Customer turns | Model responses | MCP calls | Output tokens | Included reasoning tokens | Agent seconds | Scoring seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vague requirements | 5 | 9 | 6 | 8,728 | 6,590 | 154.47 | 43.56 | 198.03 |
| Happy path | 5 | 9 | 6 | 12,869 | 10,554 | 210.60 | 35.73 | 246.33 |

The two agents used **21,597 output tokens**, including 17,144 reasoning tokens.
Total recorded input was 353,884 tokens, including 310,272 cached tokens.
Reasoning and cached input are subsets, not additional amounts to add.
These are agent usage counts, not billed EnMaaS cost or judge usage.

Agent plus scoring time totals 444.36 seconds; pytest's 444.67 seconds also
includes surrounding test overhead. Whole-case intervals include judging and
teardown and show zero overlap. Idle sleep was inhibited with `caffeinate`.
Provider load, caching, and one sample per scenario limit timing comparisons.

The [comparison tables](./glm-corrected-model-comparison.md) and
[JSON export](./glm-corrected-model-comparison.json) include measured coverage,
input/cache tokens, skill loads, errors, and all per-case quality and usage
figures. Both successful cases remain in every applicable denominator.

## Caveats retained from the evidence

Both cases used `min_quality=32` in the tighter quality search. A five-point
increase over 27.1 requires 32.1, hence **33** for the tool's integer argument.
The tool-use judge identified this discrepancy in both cases. No further
skill or code correction was made during this validation.

The vague case also overstates what unsuccessful tighter searches prove about
all possible alternatives and omits an explicit constraint-relaxation question.
The completion/fidelity judges retained those shortcomings. The happy case's
completion judge notes optimistic wording about latency headroom. A passing
overall result does not erase these recorded reasons.

The MCP server used its mock cluster, but the upstream planner independently
attempted optional cluster GPU discovery and logged eight denied node-list
requests (HTTP 401), then continued with the explicit customer-supplied GPU
filter. No cluster inventory was returned. This is a limitation of mock
isolation in the embedded dependency, and its overhead is included in timing.

## Configuration and preservation

Model: `rits/zai-org/glm-5-3`, Responses through
`https://api.enmaas.devshift.net/v1`; temperature, effort, output cap, and
penalties remain unspecified. This does not reproduce Pi.dev's Chat
Completions host/settings. The judge remains EnMaaS `gpt-6.1-sol`, MCP metrics
`conversation-v2`, and fidelity `v2`. Limits remain 300 seconds, 30 model
requests, and 40 tool calls per customer turn; tracing is off.

The clean committed source was frozen across 288 source/configuration/test/
driver files. Skill hash is `96cc8596d4dd`; happy/vague scenario hashes are
`07ced00d7013` / `f349e686f7ff`. Of 142 runtime files compared with the original
sequential roster, only the four intended skill/check/golden/local-adapter
files differ. Model profiles, host instruction, judge, and thresholds match.

All 64 prior history records remain a byte-identical prefix; exactly two
records were appended, bringing history to **66**. The 164 earlier native
files, 46 report/plan files, and prior diagnostic payloads were unchanged
before living-plan updates. Historical roster **7/8** and Chat **1/2** verdicts
remain separate. No production implementation changed during these runs.

[Full records and provenance](./glm-corrected-validation-results.json) retain
source/native hashes, judge reasons, each recommendation call, hardware-gate
results, and measured intervals. Local driver/log/JUnit evidence is under
`evals/results/*glm*corrected*20261102*`; native conversations are under
`evals/results/8fa71ecb03e2/`.

Review these results and the remaining quality-threshold/explanation issues
before choosing another implementation step. No additional paid run is
scheduled. A future four-model comparison requires all four models under a
coherent configuration after the shared corrections; this pair cannot be
pooled with the older roster.
