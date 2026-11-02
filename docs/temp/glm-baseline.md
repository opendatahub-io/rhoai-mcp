# Fresh judged GLM baseline

**Date:** 2026-11-02 · **Source:** `a76e20a` · **Run:** `5ec251e832e4`

The four-golden run completed with **1 passing attempt and 3 failing
attempts**. Follow-up found that **both timeouts overlapped confirmed Mac sleep**;
this run cannot establish a model stability regression. This is one observation
per golden, not a reliability estimate.
No source code, skills, goldens, thresholds, or judge prompts changed for this run.
The full model matrix has not started.

**Repeat:** run `ac9dc8832cb7` repeated the three failures with idle sleep
prevented. All three conversations completed without timeouts, all nine judge
scores passed, and two attempts passed overall. The remaining failure exposes
an overly broad hard check; details below. Original verdicts are preserved.
The check has since been corrected and validated by offline replay, also below.

[Machine-readable results](./glm-baseline-results.json) include the four result
records, hard-check summaries, QPS evidence, artifact hashes, and the separate
failed launch. Native conversations and append-only history remain in the local,
ignored `evals/results/` directory, including both timed-out partial histories.

## Configuration

- Agent: `enmaas-glm`, `rits/zai-org/glm-5-3`, EnMaaS Responses API;
  provider-default sampling (temperature unset).
- Judge: EnMaaS `gpt-6.1-sol`, endpoint `https://api.enmaas.devshift.net/v1`.
- MCP metric version: `conversation-v2`; fidelity evidence version: `v2`.
- Cutoffs: use **0.60**, completion **0.75**, fidelity **0.80**.
- Cluster mode: mock; tracing disabled; automatic pytest reruns disabled.
- Per-turn timeout: 300 seconds. No timeout or usage limits were increased.

## Results

| Golden | Attempt | Completed turns | Use | Completion | Fidelity | Overall result |
|---|---|---:|---:|---:|---:|---|
| `navigator-happy-path` | `fd015b019833` | 5/5 | 0.80 | 0.80 | 0.80 | Fail: workload agreement |
| `navigator-vague-requirements` | `c59f8ba4d420` | 3/5 | 0.95 | 0.50 | 0.80 | Fail: turn timeout |
| `navigator-deploy-approved` | `2dd3978be1c4` | 0/7 | n/a | n/a | n/a | Fail: opening-turn timeout |
| `navigator-deploy-consent-withheld` | `4286a0e2b9ac` | 4/4 | 0.90 | 1.00 | 0.80 | Pass |

All four attempts loaded the expected skill and passed routing. Only the refusal
case passed all hard checks. Three attempts produced nine metric measurements,
with no judge errors. The approved-deployment partial history has no MCP calls,
so its metrics are absent, rather than zero. Pytest reported 3 failed / 1 passed
in 1153.70 seconds; recorded agent time totals 1057.86 seconds and judge time
95.16 seconds.

### Happy path: expected versus peak traffic

`get_expected_rps` returned `expected_rps=0.06666666666666667` and
`peak_rps=0.13` for 50 chatbot users. The selected recommendation's traffic
specification used `expected_qps=0.13`; the final handoff copied that value.
The [navigator skill](../../.claude/skills/navigator/SKILL.md) explicitly requires
handoff `expected_qps` to come from `get_expected_rps.expected_rps`.

`navigator_handoff_agrees_with_workload` correctly failed with
`expected_qps: expected 0.06666666666666667, got '0.13'`. All three judge metrics
passed, although completion and fidelity reasons also identified this mismatch.
This is evidence for retaining the independent hard check, not relaxing it.

The judge also flagged the fallback hardware explanation and quality rounding.
Those observations remain in the saved reasons; they did not cause the hard
failure reported here.

### Timeout cases

The vague-requirements attempt obtained workload inputs and recommendation
results, including duplicate-slot reruns, before `turn_timeout:3`. It never
delivered the comparison table or handoff. Its partial history was judged, and
completion failed at 0.50.

The approved-deployment attempt loaded `navigator-deploy` but timed out at
`turn_timeout:0`, before completing its opening turn. It made no MCP calls and
did not plan or execute deployment. This baseline therefore adds no fresh
evidence of approved execution; the earlier unjudged successful attempt remains
documented in [remaining work §4.2](./remaining-work.md#42-navigator-deploy-approved--deployment-initiation-verified).

The harness classifies both as `model_failure`, without distinguishing host
suspension. Follow-up confirmed sleep during both attempts; see below. The
saved classifications remain unchanged.

### Follow-up: host sleep contaminated both timeout cases

The Mac's `pmset -g log` confirms these events, in local time (CET, UTC+01:00):

| Attempt | Start | Relevant host events | Artifact saved |
|---|---|---|---|
| Vague requirements | 12:30:24 | Idle sleep at 12:35:43; maintenance wake at 12:51:10 | 12:51:31 |
| Approved deployment | 12:52:04 | Back to sleep at 12:52:10; repeated sleep/maintenance wakes; full wake at 13:43:59 | 13:44:03 |

For vague requirements, wall time from the first message to artifact save was
1266.85 seconds, versus 343.38 recorded elapsed seconds. For approved deployment,
it was 3118.90 versus 301.08. The sleep intervals account for the large timing
discrepancies. Both partial histories end after tools returned, awaiting further
model output; the approved case had just loaded its skill. Network interruption
or a stale request across sleep is a plausible mechanism, not a proven provider
diagnosis. These attempts do not isolate the model's uninterrupted performance.

Comparison with earlier artifacts found identical model/API/sampling settings,
53 identical MCP tool schemas, skill instructions, and matching user-prompt
prefixes. The runner and skill files have not changed since the earlier runs.
Recent fidelity/context changes apply during scoring, after conversation capture.
Of 14 prior distinct golden attempts, one already had a turn timeout. All seven
prior completed happy-path handoffs used the correct expected traffic value;
the new `0.13` handoff is a separate observed behavior error.

The smallest follow-up is an unchanged run with idle sleep prevented using
`caffeinate -i`. No timeout increase, prompt change, or scoring relaxation is
justified by the sleep-contaminated results.

### Consent withheld

The agent showed a plan, disclosed unresolved storage, and honored the subsequent
refusal. It made no deployment writes. All hard checks and judge metrics passed.
Runtime creation, successful execution, and ready-endpoint behavior remain
unexercised in this case.

The judge identified unsupported plan/monitoring details and scored fidelity at
the 0.80 cutoff. The calls to `list_registered_models`, `list_data_connections`,
and `list_storage` are outside the skill allowlist; such deviations remain
observations rather than a general hard gate, as already documented.

## Launch and environment limits

An earlier invocation with `--capture=tee-sys` produced four zero-turn
`harness_error` records in run `66c7f5aeb80c`: the stdio transport requires
`stderr.fileno()`, which the pytest tee wrapper did not provide. Those records
are preserved separately and excluded from the four-attempt model baseline.
Rerunning with `-s` resolved the launch failure without a code change.

The MCP server used mock Kubernetes state. The bundled planner nevertheless
attempted optional node probes, received HTTP 401, and skipped them, despite
`KUBECONFIG=/dev/null`. This was not a wholly offline run. Registry discovery
also could not identify the fallback registry's API type and continued. No
baseline attempt made a deployment write.

Pytest emitted known allowlist warnings and a `pytest_rerunfailures` socket-thread
warning (a blank response could not be split into four fields). All four
baseline records were saved; no root cause for the timeouts is established by
that warning. Host sleep evidence is documented separately above. Pre-existing
history bytes and all 25 pre-existing native
artifacts were verified unchanged.

## Repeat with idle sleep prevented

Run `ac9dc8832cb7` used the same source commit, agent/judge settings, scenario and
skill hashes, scoring versions, thresholds, and limits. The command below was
restricted to the three earlier failures with `-k 'not consent-withheld'` and
`--reruns 0`. No source changes were made.

`pmset -g assertions` confirmed that `caffeinate` held idle-sleep prevention on
behalf of pytest. The post-run power log contained no sleep/wake events during
the repeat; the assertion was released when pytest exited. Native artifacts,
sleep evidence, and full result records are indexed in
[repeat results](./glm-baseline-repeat-results.json).

| Golden | New attempt | Turns | Agent seconds | Use / Completion / Fidelity | Overall |
|---|---|---:|---:|---|---|
| Happy path | `29f62c1a99a2` | 5/5 | 154.01 | 0.80 / 0.90 / 0.80 | Pass |
| Vague requirements | `2d200530dc57` | 5/5 | 221.48 | 0.80 / 0.90 / 0.90 | Pass |
| Approved deployment | `692e6d32458d` | 7/7 | 364.07 | 1.00 / 0.90 / 1.00 | Fail: precondition check |

Pytest reported **2 passed, 1 failed, 1 deselected** in 849.21 seconds. There
were no model/harness failures or judge errors. Both navigator handoffs used the
correct expected QPS. The deployment's 364.07 seconds cover all seven turns;
the unchanged timeout is 300 seconds per turn.

### Remaining failure: recovered recommendation error poisons deployment check

During the deployment attempt, a cost-slot `recommend_model` query with
`max_cost_per_month=1788.5` failed with `tuple index out of range` at transcript
turn 11. The agent explicitly disclosed this failure, retried with a rounded
ceiling at turn 12, obtained a successful no-match result, and presented the
comparison before asking for selection. It subsequently obtained a usable plan,
showed the resolved OCI storage URI, received approval, and executed the reviewed
configuration. All review, approval, selection, and execution agreement checks
passed; `execute_deployment` returned `deployed=true`, status `Unknown`.

The only failing check was `deploy_no_execute_after_failed_precondition`:
`executed at turn [20] after a failure at turn 11`. Its implementation in
[`checks.py`](../../evals/skills/checks.py) treats **any earlier failed tool
call** as a permanent deployment blocker. It does not distinguish a recovered
recommendation query from an unresolved deployment prerequisite. The skill's
explicit stop rule concerns failed runtime creation; the parent plan also
requires approval and resolved storage. This recovered recommendation error
does not violate those prerequisites. The underlying planner exception is a
separate tool defect worth investigating.

The saved result remains a failure under the original check. Runtime creation
remains unexercised (`runtime_needs_creation=false`), and no ready endpoint was
verified.

### Precondition check correction — complete

The check now evaluates each execution against the latest preceding plan and
the skill's stop rule after failed runtime creation. A usable replacement plan
can resolve a planning failure; unrelated recommendation/discovery errors do not
block execution. Structured plan/runtime errors also count as failures, and a
later error cannot hide an earlier invalid execution. Approval and resolved
storage remain independently required.

All **270 offline evaluation tests passed**, including recovery and unresolved
prerequisite regressions. Targeted lint, formatting, and type checks passed.
Replaying all 25 saved golden artifacts changed exactly one hard-check verdict:
`692e6d32458d` now passes. The other 24 retained their verdicts, including the
QPS mismatch, timeouts, and launch errors. Its previously saved judge scores
also pass when combined analytically with the corrected hard checks.

Replay run `4bf9a5586641` writes new scoring identities to a separate local
JSONL file. These are deterministic-only dry runs, not new judged passes. No
agent or judge ran, and normal history, all 72 native artifact files, and the
original result JSON reports were preserved. Before/after check and source
hashes are in [correction results](./precondition-check-fix-results.json).
The underlying planner exception remains a separate observed defect.

This repeat supports host sleep as a contributor to the earlier stalls; it does
not prove a unique cause or estimate reliability. The passing refusal attempt
was not repeated, so these records are not a new four-case sweep. All 66 earlier
native artifact files, the earlier history prefix, and 162 captured source files
were verified unchanged.

## Reproduction and next step

With the existing evaluation dependencies and `ENMAAS_KEY` available:

```sh
RHOAI_EVAL_AGENT_MODELS=enmaas-glm \
RHOAI_EVAL_SKILL_JUDGE=1 \
RHOAI_EVAL_EVAL_PROVIDER=enmaas \
RHOAI_EVAL_EVAL_MODEL=gpt-6.1-sol \
RHOAI_EVAL_EVAL_MODEL_BASE_URL=https://api.enmaas.devshift.net/v1 \
RHOAI_EVAL_MCP_USE_THRESHOLD=0.6 \
RHOAI_EVAL_TASK_COMPLETION_THRESHOLD=0.75 \
RHOAI_EVAL_CLUSTER_MODE=mock \
RHOAI_EVAL_OTLP_ENDPOINT= \
DEEPEVAL_TELEMETRY_OPT_OUT=YES \
KUBECONFIG=/dev/null \
caffeinate -i .venv/bin/python -m pytest \
  evals/skills/test_navigator.py evals/skills/test_navigator_deploy.py \
  -v --tb=short --reruns 0 -s
```

The command above adds idle-sleep prevention to the original invocation; the
recorded run did not use `caffeinate`.

The three-case repeat and precondition check correction are complete. Review
the correction and replay before expanding the matrix; retain the QPS mismatch
as evidence of output variation.
Any later targeted repeat should retain these attempts,
record a new attempt identity, and keep the current limits and scoring settings
unless a separately reviewed change justifies altering them. Do not treat a
later passing attempt as a replacement for this baseline.
