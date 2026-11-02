# GLM investigation: hardware consent and planner filtering

**Date:** 2026-11-02 · **Source:** `753e8ca` plus the correction below · **Scope:** Navigator only

**Two concrete defects are now addressed:** GLM inferred permission to skip the
hardware question, and the embedded planner returned hardware outside the
requested filter. A missing deterministic check allowed the former to pass hard
checks. The original missing-handoff failure remains a separate, intermittent
output failure; the API experiments did not establish a fix for it.

[Machine-readable evidence](./glm-root-cause-correction-results.json) contains
the saved-case replay, before/after planner results, four decision probes,
source hashes, usage, and preservation checks. Historical scores are unchanged.

## What caused the GPU failures

The failed Chat happy case (`2aaa97922492`) received workload confirmation but
no GPU inventory or explicit permission for an unfiltered search. It nevertheless
made three unfiltered recommendation calls before the hardware reply. Its visible
answer announced that choice; retained model metadata also shows it interpreting
“go ahead” as consent. The full instruction was available. This was a mistaken
consent inference, rather than a missing tool or lost context.

Separately, `llm-d-planner` deliberately retries without a GPU filter when its
initial search finds no matching configurations. The HTTP adapter already
filters these returned rankings before choosing category winners. The embedded
adapter omitted that step, so an A100-only request could return H100 in all four
slots. Navigator then received a successful-looking constrained response, without
the empty slots that trigger its explicit fallback and hardware warning.

Inspection of the latest eight-case roster and subsequent two Chat cases found
**20 slots outside the requested filter across five conversations**: one Opus,
two GLM Responses, and two GLM Chat cases. Those GPU values were present in tool
responses; they cannot all be attributed to model invention. This is a shared
backend defect, not evidence that GLM's Pi.dev configuration caused it.

## Minimal correction

- [Local planner](../../src/rhoai_mcp/composites/planner/local_client.py): reuse the
  HTTP adapter's existing GPU filter on all four ranked lists before selecting
  their winners. An explicit empty filter retains unconstrained behavior.
- [Checks](../../evals/skills/checks.py) and [goldens](../../evals/skills/goldens.py):
  add `navigator_recommend_after_hardware_confirmation` to both Navigator
  scenarios. Every recommendation must follow their scripted customer hardware
  reply; later valid calls cannot hide an earlier violation. The check covers
  the manual-inventory/explicit-consent branch, leaving automatic discovery
  outside this scenario-specific assertion.
- [Navigator skill](../../.claude/skills/navigator/SKILL.md): two sentences clarify
  that confirming SLOs or saying “go ahead” does not answer the GPU question.
  Ask again and wait unless inventory, lack of knowledge, or explicit permission
  for unfiltered recommendations has been supplied.

No model API, effort, output cap, penalties, shared host instruction, judge,
cutoff, or deployment skill changed. Navigator skill hash is now `96cc8596d4dd`.
Grading intent changes the happy/vague scenario hashes to `07ced00d7013` /
`f349e686f7ff`; both deployment hashes are unchanged.

## Verification

The real bundled planner reproduces the old defect with 50 chatbot users,
`preferred_gpu_types=["A100-80"]`, and `ttft_max_ms=44`: all four slots returned
H100. With the correction, all four are empty. A separate explicit
`preferred_gpu_types=[]` request returns the same H100 configurations as before.
The corrected result and unconstrained control were also verified with cluster
auto-detection disabled and sockets blocked. These are synthetic bundled
benchmarks, not a production hardware suitability assessment.

Offline replay of all ten saved conversations makes the new gate fail only the
known Chat happy case. It passes all eight original roster cases and Chat vague.
The original GLM missing-handoff case retains its three handoff failures. This
is a derived check replay, not a rewritten verdict or new judge score.

Regression tests were written before the production fixes. The broader offline
evaluation/planner suite passed **597 tests, with 9 live tests deselected**.
The final focused run passed **22 tests**, covering ordering, later correction,
assistant-only claims, missing replies, explicit unfiltered consent, all four
GPU categories, missing matches, vendor normalization, and real planner fallback.
Ruff, mypy for the three changed source modules, and skill validation also pass.

```bash
.venv/bin/pytest tests/evals tests/composites/planner tests/integration/test_planner_local.py -m 'not live' -o addopts= -q --reruns 0
```

## Bounded live instruction probe

Four Chat Completions requests used the failed case's saved context immediately
after workload confirmation, before any premature recommendation or GPU reply.
All earlier content/reasoning, 53 tool schemas, model, API, and unspecified
sampling/effort settings were fixed. Only the loaded skill's two-sentence
clarification varied. An offline, socket-blocked control and normalized wire
hashes verified that isolation. SDK retries were disabled; no tool was executed
and no judge ran.

| Order / wording | Behavior | Output tokens | Included reasoning tokens | Seconds |
| --- | --- | ---: | ---: | ---: |
| Original 1 | Premature unfiltered recommendation call | 2,796 | 2,652 | 45.822 |
| Clarified 1 | Asked for GPU inventory and waited | 680 | 552 | 11.535 |
| Original 2 | Asked for GPU inventory and waited | 776 | 631 | 13.954 |
| Clarified 2 | Asked for GPU inventory and waited | 774 | 587 | 12.722 |

All four returned HTTP 200 without errors. The failure recurred once with
original wording; both clarified samples respected the prerequisite. This
supports retaining the targeted clarification, but two samples per variant are
not a reliability estimate. Cache usage varied; these timings do not establish
a speed improvement. Total output was 5,026 tokens, including 4,422 reasoning
tokens. This was neither a full conversation sweep nor full Pi.dev parity.

## What remains

The original Responses handoff omission (`4f2a187d74cd`) remains real. Prior
matched final-turn replays passed on both APIs; original HTTP bytes are absent,
so model generation versus gateway behavior cannot be conclusively separated.
There is no demonstrated API or reasoning-effort fix. See the
[handoff investigation](./glm-handoff-investigation.md) and
[API validation](./glm-api-validation.md).

Review this correction, then validate only the two GLM Navigator goldens as a
new cohort using the corrected backend, skill, and checks. The decision probe
does not exercise the full fallback/table/handoff flow. A future model
comparison must use one coherent cohort for all four models after these shared
changes; the historical roster remains 7/8 and the historical Chat pair 1/2.
No further full sweep or deployment run was performed.

Before living-plan updates, all **209 snapshotted files** were unchanged:
164 native evidence files, 44 report/plan files, and ordinary history. History
remains byte-identical at **64 records**. Only the two living plans are updated
after that verification. Diagnostic drivers, raw requests/responses, and native
probe artifacts remain locally under `evals/results/`, with hashes in the JSON
report; no credentials or authorization headers were recorded.
