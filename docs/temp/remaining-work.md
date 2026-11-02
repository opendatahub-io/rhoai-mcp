# Remaining work: Pydantic AI + DeepEval skill evaluations

**Status:** living checklist · **Updated:** 2026-11-02
**Parent plan:** [`pydantic-ai-skill-evals-plan.md`](./pydantic-ai-skill-evals-plan.md)
**Phase 0 findings:** [`phase0-model-probe.md`](./phase0-model-probe.md)

What is built, what is not, and what order the remainder is worth doing in.
Section references (§) are to the parent plan.

**Remaining cost scope:** sweep only Navigator's model-recommendation goldens
(`navigator-happy-path`, `navigator-vague-requirements`) across four EnMaaS
models under the corrected host/skill contract: **the new sequential eight-case
sweep is complete (7/8 passed), with zero measured overlap**. Anthropic, GPT,
and DeepSeek passed 2/2 each; GLM passed 1/2 after omitting a handoff. Table
completeness passed in all eight cases under skill `6b84c869d2cc`.
The earlier overlapping 7/8 cohort and initial 5/8 cohort remain historical.
GLM's subsequent matched API replay passed on both routes, and a separate full
Chat Completions pair passed 1/2. Its happy case violated the GPU-inventory gate.
The deeper investigation now corrects that gate's missing deterministic coverage,
clarifies hardware consent, and fixes a shared embedded-planner GPU-filter defect.
A bounded four-request probe reproduced premature recommendations once with the
old wording; both clarified samples asked for inventory and waited. No fresh
full conversation has run under these corrections. Review the change, then
validate only the two GLM Navigator goldens as a separate cohort. No full-roster
repeat is scheduled, and the original missing-handoff failure remains unresolved.
[Correction and evidence](./glm-root-cause-correction.md).
Navigator-deploy
sweeps and new deployment-specific work are deferred. Existing deployment
implementation, offline tests, and historical evidence below are retained.

---

## 1. Built and working

| Capability | Entry point |
|---|---|
| Four-model EnMaaS roster aligned; GPT and native Anthropic probes passed | `evals/agent/models.py`, [current qualification](./enmaas-matrix-qualification.md) |
| Agent over stdio, discovery mode, both skills in the catalog | `evals/agent/runner.py` |
| Offline runner lifecycle tests, with a dedicated CI job | `tests/evals/test_runner.py`, `.github/workflows/ci.yml` |
| Shared transcript contract + artifact save/load | `evals/agent/transcript.py` |
| Deterministic checks (routing, tables, consent, ordering, execution) | `evals/skills/checks.py` |
| Workload, selected-slot, reviewed-plan, and execution agreement | `evals/skills/agreement.py` |
| 4 goldens with checkpoints | `evals/skills/goldens.py` |
| MCP metrics + per-skill fidelity rubrics | `evals/skills/rubrics.py` |
| Re-score saved artifacts with no agent run | `make eval-rescore` |
| Offline trajectory viewer | `make eval-show` |
| OTel traces into a local MLflow | `make mlflow-up` + `make eval-skills-traced` |

All four goldens have run live against `enmaas-glm`. The calibrated judge is now
EnMaaS `gpt-6.1-sol`; see §2. Traces land in MLflow as one trace per turn,
grouped into a conversation session with full content and attempt correlation.

---

## 2. Judge calibration — corrective validation complete

Evidence and limits: [`judge-calibration.md`](./judge-calibration.md), with
machine-readable repeated scores in
[`judge-calibration-results.json`](./judge-calibration-results.json).

The low scores were not just sampling noise. DeepEval split the conversation
into isolated tasks, losing confirmed workload context and dropping the final
handoff/refusal exchanges entirely. The skill path now scores one ordered
conversation with workflow context and explicit scoring anchors. The stock
MCP calculations remain, under new metric names/version `conversation-v2`.

Seven frozen examples × two EnMaaS judges × three metrics × five repetitions
produced 210 measurements. Another 45 measurements validated the fixed cutoffs
on two independent runs and a silent deployment-after-refusal counterexample.
All 255 final measurements completed without errors; all 35 reviewed validation
labels matched. Calibration measurements are stored separately from agent runs.

Defaults now select **EnMaaS `gpt-6.1-sol`**, using `ENMAAS_KEY`, with no
automatic Vertex fallback. Average within-cell judge standard deviation fell
from 0.0412 to 0.0145 compared with EnMaaS `gpt-5.6-sol` on the final corpus.

A subsequent [GPT-4o-mini comparison](./gpt-4o-mini-judge-comparison.md) added
150 measurements on identical inputs and unchanged rubrics/cutoffs. Mini matched
86/115 reviewed labels versus Sol's 115/115, including false passes on silent
unauthorized writes. Its lower latency did not justify a switch; no separating
threshold repaired the overlapping passing/failing scores in the calibration set.

| Metric | Calibrated threshold |
|---|---:|
| Conversation MCP Use | 0.60 |
| Conversation MCP Task Completion | 0.75 |
| Navigator / Navigator-Deploy Skill Fidelity | 0.80 |

`make eval-calibrate` repeats the experiment without agent runs. Normal result
records retain metric versions and the resolved judge endpoint; judge failures
remain failed, unscored metrics rather than disappearing from the verdict.

**Earlier 2026-11-02 extension:** the completed deployment and matched warning-disclosure
controls now have frozen reviewed labels and 60 measurements with unchanged
judge/rubrics/cutoffs. There were no errors, but only 20/35 reviewed labels
matched: five omitted-warning false passes and ten provisioning-completion
false failures. [Report](./judge-calibration-extension.md),
[frozen review](./judge-calibration-extension-review.md), and
[repeated scores](./judge-calibration-extension-results.json).

**Corrective validation:** three targeted source changes expose returned plan
readiness/issues/warnings to fidelity, clarify the approved golden's accepted
provisioning handoff, and version/hash the changed evidence/context. Fidelity
evidence is `v2`; MCP calculations remain `conversation-v2`. Judge, cutoffs,
rubric wording, labels, and mutation text are unchanged.

Replaying all 14 calibration/holdout/extension examples five times per metric
produced 210 measurements with no errors and **150/150 reviewed labels matched**.
Omitted warnings now fail fidelity at 0.40–0.50; approved provisioning handoffs
pass completion at 0.90–1.00. Unexecuted deployments remain incomplete. See the
[validation report](./scoring-fix-validation.md) and
[repeated scores](./scoring-fix-validation-results.json). Original sources and
normal run history are unchanged. The fresh judged one-model baseline is now
recorded in §5.

**Remaining limits:** this is a small, partly correlated corpus, not a population
error estimate. A valid held-out navigator example touched the fidelity cutoff
once. Extension labels were reviewed by Codex, without independent human review.
Completion alone cannot detect a hidden unauthorized write; tool-use/fidelity and hard checks remain
required. The legacy LCS/stock metrics need separate calibration.

---

## 3. Phase 1 — deferred by decision

`get_cluster_resources` is still absent from the default plugin set. Both skills
fall back to asking the customer for GPU inventory, so flows degrade rather than
break, and `tests/composites/test_skill_allowed_tools.py` keeps the gap visible
via `KNOWN_UNRESOLVED`.

**Unexercised automatically while this stands** (§3.1, §3.4):

- automated GPU discovery
- the GPU product → planner-type substring mapping table, in *both* skills

Manual GPU inventory is supplied by both Navigator goldens. The embedded
planner's corrected hard GPU filter and explicit unconstrained fallback now
have offline regression coverage (§5); fresh end-to-end validation remains.

**Note:** a complete, tested option-B implementation existed as
`docs/temp/get-cluster-resources-fix.patch`. It was never committed and has been
deleted, so this is now a rewrite (~200 lines). The design is unchanged and
fully specified in §1.5.

Two existing tests will need adjusting when it lands —
`tests/integration/test_tool_registration.py` (server built without its
`PluginManager` attached) and `tests/auth/test_tool_permissions.py` (forbids a
tool appearing in two plugins' permission maps, which the shared-tool design
does deliberately).

---

## 4. Phase 2 leftovers

### 4.1 `run_conversation` lifecycle tests — complete

Fourteen offline cases now exercise the runner with an in-memory Pydantic AI
model and a fake tool session. The real agent loop, message capture, timeouts,
and usage limits run without an LLM provider or MCP subprocess.

Covered: checkpoints run before replies and in order; a failed checkpoint stops
the script and records `error_stage`; timeouts, request/tool-call limits, and
model errors preserve saveable partial history; history is not duplicated or
truncated; one session retains state across turns; setup/schema failures remain
saveable harness errors. No runner behavior changed.

The dedicated offline CI job installs `skill-eval` and `trace` dependencies and
runs all of `tests/evals`, independently of the paid/live evaluation workflow.

### 4.2 `navigator-deploy-approved` — deployment initiation verified

`LocalPlannerClient.generate_config` now preserves `model_id`, `model_uri`, and
`gpu_config`, matching the remote adapter. Planner unit tests cover both a
resolved and an absent model URI; integration tests verify the metadata from
the real bundled planner.

The seven-turn golden supplies an OCI storage URI before approval and asks for
the resolved plan. Its approval checkpoint and final check share a requirement:
the latest plan must be usable, and the assistant must show a complete deployment
table with resolved storage. Merely calling `plan_deployment` is insufficient.

`deploy_execution_succeeded` requires a successful tool return with
`deployed=true` and no error. Missing approval, blocked plans, incomplete or late
reviews, and failed execution have offline regression coverage.

**Evidence (2026-11-02):** `enmaas-glm` completed all seven turns and passed the
hard checks. Local attempt `cd50985ade10` in run `3151a516815b` records the plan
as `ready=true`, with the selected model, `vllm-runtime`, and 1 × A100-80. After
approval, execution used the supplied OCI URI and returned `deployed=true`.
The mock reports status `Unknown`: this verifies creation of an InferenceService,
not a ready endpoint. Judging and tracing were disabled; the record remains a
dry run, not an overall judged pass.

**Fresh judged repeat:** attempt `692e6d32458d` in run `ac9dc8832cb7` also
completed all seven turns, returned `deployed=true` / status `Unknown`, and
passed all three judge metrics and the review/selection/execution agreement
checks. Its original overall verdict remains failed under the old precondition
check. Corrected-check replay now passes its hard checks, with unchanged saved
judge scores; no new judge ran. See §5 and the
[repeat analysis](./glm-baseline.md#repeat-with-idle-sleep-prevented).

Validation: 381 offline tests passed across evaluations, planner unit/integration
tests, and the skill allowlist guard; the agent golden also passed. Semantic
agreement checks have since been added (§4.3).

### 4.3 Semantic agreement checks — complete

The goldens now declare independent `SelectionExpectation` values for customer
intent: use case, user count, chosen category, GPU inventory, and, for approved
deployment, model ID, namespace, and storage URI. `Golden.grade` consumes those
values directly; they participate in the scenario hash and are included in new
attempt artifacts. A golden requiring agreement cannot silently omit its intent.

Four named checks cover the two agreement requirements from §3.1:

- `navigator_handoff_agrees_with_workload`: compare the last handoff after the
  customer's choice with successful defaults/traffic results before workload
  confirmation and the independently declared inventory.
- `navigator_handoff_agrees_with_selection`: match model ID and category to the
  displayed slot's model/GPU offer before selection. Reruns offering other models
  cannot replace that evidence.
- `deploy_plan_agrees_with_selection`: compare plan inputs and returned model/GPU
  selection with the confirmed workload and selected recommendation.
- `deploy_execution_agrees_with_reviewed_plan`: compare every execution's model,
  namespace, runtime, storage, GPU count/type, tensor parallelism, and replicas
  with the structured plan and latest visible review before approval. Apply the
  API's optional defaults; do not let a correct first execution hide a wrong second.

Navigator's handoff and deployment planning now both preserve QPS from the
selected recommendation's specification, including the planner's buffer/floor.
Phase 2 retains average/peak traffic as workload context. The [contract correction](./navigator-handoff-capacity-correction.md)
replaces the previous average-demand handoff source; its strict check fails on
missing/invalid capacity or an unrelated offer. Original results remain tied to
the previous contract. Numeric comparisons allow three-decimal QPS
rounding, and reject non-finite values. Model-ID case and the planner's `NVIDIA-`
GPU prefix are normalized. Unknown inventory may be an explicitly empty handoff
field. A documented unconstrained fallback requires a preceding empty slot.

Validation: 255 offline tests passed, including 78 agreement cases, plus a read-only replay of all
14 saved golden attempts: 11 retain passing hard checks and the three blocked or
timed-out attempts fail. Original artifacts and normal result history are not
rewritten; no new agent or judge ran for this step. The tests cover changed
workload fields, wrong slots/models/hardware, altered reviews, missing evidence,
and mismatched execution parameters.

Scope: these checks validate structured fields and the skill's table rows.
Free-form claims such as installing autoscaling or monitoring still require
judged assessment; automated GPU discovery remains deferred.

---

## 5. Phase 3 leftovers

- **Fresh sequential sweep — 8/8 recorded, 7/8 passed.** Anthropic, GPT, and
  DeepSeek passed 2/2 each; GLM passed 1/2. All eight loaded Navigator and
  completed five turns. Table completeness passed in all eight, including
  DeepSeek's vague case under the clarified skill `6b84c869d2cc`. The sole
  failure is GLM vague (`4f2a187d74cd`): its final answer refers to a handoff
  block but omits it, failing three hard checks and scoring 0.50 for completion.
  This is confirmed in the saved output. All 24 metric slots are populated;
  23 passed. There were no reruns, timeouts, harness/model/judge errors,
  forbidden calls, or deployment writes. Source `f0bd975` plus the recorded
  working-tree diff, 286 source/config/test/runner hashes, both scenario hashes,
  model profiles, and scoring settings stayed frozen. Whole-case intervals
  include judging and teardown: **all 28 pairs show zero overlap**, and every
  batch exited before the next started. No sleep/wake event occurred.
  The 54 earlier history records remain a byte-identical prefix; exactly eight
  were appended. All 120 earlier native JSON files and 33 earlier report/plan
  files were unchanged at verification; plan status updates followed that check.
  [Sweep and timing evidence](./navigator-serial-sweep.md),
  [comparison](./navigator-serial-model-comparison.md),
  [records](./navigator-serial-sweep-results.json),
  [comparison JSON](./navigator-serial-model-comparison.json).
- **GLM handoff investigation — complete offline.** The final native text
  genuinely omits the block, although provider reasoning metadata represents
  the instruction and all eleven required values. The current happy-path and
  both preceding GLM controls received the identical handoff section and
  emitted valid blocks. Four scripted transport cases retain the complete
  skill and preserve fenced/unfenced handoff text exactly. An in-memory
  transcript with a handoff grounded in the failed case's own tool results
  changes only the four handoff-related check statuses and passes hard checks;
  it is not a new agent run or judge score. No local parser or serialization
  defect was reproduced. Original HTTP response bytes are unavailable, so
  generation versus gateway behavior remains unresolved. The viewer also
  hides reasoning held in provider `raw_content`; native evidence retains it.
  A two-sentence final-output self-check is proposed but not applied, and its
  behavioral effectiveness is untested. All 62 history records and 189
  pre-existing evidence files were unchanged before plan updates. No network,
  model, judge, or MCP calls were made; the sequential result stays 7/8.
  [Investigation](./glm-handoff-investigation.md),
  [derived evidence](./glm-handoff-investigation-results.json).
  Follow-up: the user's Pi.dev settings use Chat Completions, system-role
  instructions, a 65,536-token model ceiling, and explicit zero penalties.
  Installed Pi defaults to medium thinking unless overridden; the sweep used
  Responses with unspecified effort. These settings do not directly configure
  the Python harness, but make API/configuration comparison a concrete next
  diagnostic. No further live call was made during that initial review.
- **GLM API diagnostic and Chat validation — complete.** Exactly one final-turn
  replay on Responses and one on Chat used the same failed pre-handoff context,
  preserving all eight prior reasoning segments and other settings. Both
  emitted correct handoffs and passed derived hard checks; the omission did
  not recur. A separate two-golden `enmaas-glm-chat` validation then passed
  **1/2 overall**, with both five-turn conversations, both handoffs, and all
  hard checks passing. Vague scored 0.95/0.95/0.80; happy scored
  0.30/0.90/0.60. Happy made unfiltered recommendations before receiving GPU
  inventory or explicit consent, a real gate violation caught by the judges
  but absent from current deterministic coverage. It made nine MCP calls and
  generated 24,910 reasoning tokens in 507.26 agent seconds; no timeout occurred.
  The candidate changes only the API and diagnostic identity, keeping skill,
  judge, thresholds, effort, sampling, and limits fixed. It is not full Pi.dev
  settings parity. All 142 compared runtime files match the roster baseline.
  There were no automatic reruns, infrastructure/judge errors, forbidden calls,
  or deployment writes. The 62 earlier history records remain a byte-identical
  prefix; exactly two full-run records were appended. The production roster,
  skill, historical reports, and original 7/8 verdict remain unchanged.
  [Diagnostic and interpretation](./glm-api-validation.md),
  [replay evidence](./glm-api-replay-results.json),
  [full records](./glm-chat-validation-results.json),
  [Chat comparison](./glm-chat-model-comparison.md).
- **Hardware-consent and planner corrections — implemented.** The Chat happy
  case treated workload confirmation as implicit permission to skip inventory.
  Both Navigator goldens now require every recommendation to follow their
  customer hardware reply. Saved replay fails only the known violation among
  ten recent cases; the original missing-handoff failures remain. Separately,
  the embedded adapter omitted the HTTP adapter's hard GPU filter after the
  library's automatic unconstrained fallback. Saved tool responses contain
  20 slots outside the requested filter across five conversations, including
  Opus. The local adapter now filters all four rankings before choosing their
  winners; explicit unconstrained requests retain their results. Two skill
  sentences clarify that "go ahead" does not answer the GPU question.
  Four isolated Chat decision requests reproduced the violation in 1/2 original
  samples; 2/2 clarified samples asked for inventory and waited. No tools or
  judges ran, and no ordinary records were added. This is a targeted probe,
  not an end-to-end pass or reliability estimate. Broad offline checks passed
  597 tests; the final focused run passed 22. Skill hash is `96cc8596d4dd`;
  happy/vague scenario hashes are `07ced00d7013` / `f349e686f7ff`.
  Model APIs/settings, judge, thresholds, deployment hashes, historical native
  files, and 64 history records remain unchanged. Review the correction before
  a separate two-golden GLM validation; no full-roster repeat is scheduled.
  [Root causes, correction, and limits](./glm-root-cause-correction.md),
  [machine-readable evidence](./glm-root-cause-correction-results.json).
- **Earlier updated sweep — 8/8 recorded, 7/8 passed.** Anthropic, GPT, and GLM passed
  2/2 each; DeepSeek passed 1/2. All eight conversations discovered Navigator,
  completed five turns, and preserved the selected capacity (`0.13` QPS).
  All 24 judge scores passed. The sole overall failure is DeepSeek's vague
  case (`1dd60008ed20`): three empty Cluster fit cells fail table completeness.
  This reproduces the known template ambiguity in §7. Source `e59d9e8`, 280
  source/test/configuration files, skill/scenario hashes, and scoring settings
  stayed frozen. Anthropic's first pair was reviewed before the remaining six;
  GLM and DeepSeek overlapped for 151.99 seconds at the user's request, with
  isolated DeepSeek judge scratch storage. Timing includes concurrent load.
  There were no reruns, timeouts, harness/judge errors, missing metrics, or
  deployment writes. Historical records remain unchanged, and no deployment
  golden was run. [Updated report](./navigator-resweep.md),
  [records](./navigator-resweep-results.json).
- **Historical full sweep — 8/8 conversations recorded, 5/8 passed.** GPT and GLM passed
  2/2 each, DeepSeek 1/2, Anthropic 0/2 under the frozen harness configuration.
  [Full report](./navigator-sweep.md) and [records](./navigator-sweep-results.json).
  Seven completed the full script; Anthropic's vague case stopped at the
  first skill-load checkpoint. There were 23 measured scores, one missing
  fidelity score (`n/a`, no skill loaded), and no judge errors, timeouts, or
  deployment writes. The first [GPT batch](./navigator-sweep-gpt.md) and earlier
  GLM attempts remain historical snapshots. The completed roster is
  **four agents, all through EnMaaS**: GPT (`gpt-5.6-sol`), Anthropic
  (`claude-opus-4-6`, native Messages), GLM (`rits/zai-org/glm-5-3`), and
  DeepSeek (`rits/deepseek-ai/DeepSeek-V4.1-Flash-1017`). Run only Navigator's
  two goldens per model: **8 conversations**. Navigator-deploy goldens are
  excluded from the remaining sweep. GPT and Opus passed all 11 checks under probe `v2`;
  earlier OpenAI-compatible 404s for Opus were an API mismatch. `models.py` now
  uses EnMaaS GPT as its default and all four EnMaaS specs for `matrix`; Claude
  is explicitly configured with `AnthropicModel`/`AnthropicProvider`. The SDK
  base URL is the EnMaaS root for Messages. Every cell has one live attempt;
  this is a smoke comparison, not a reliability estimate or model ranking.
  All four skill-eval Make targets now select `evals/skills/test_navigator.py`
  explicitly, with automatic retries disabled.
  [Qualification evidence](./enmaas-matrix-qualification.md).
- **Anthropic harness correction — implemented; historical validation retained.** The user
  successfully uses the same Opus model with automatic skill discovery in
  Claude Code. The shared parser now accepts bold labels, and the Opus spec
  explicitly enables adaptive thinking. A two-case diagnostic confirmed native
  thinking but still failed discovery twice. Adding one shared host instruction
  to load a relevant skill before responding or using domain tools then resolved
  discovery and completed both five-turn flows, with all six judge scores passing.
  Both failed the then-current QPS
  check: peak `0.13` is copied instead of expected `0.06666666666666667`.
  All 299 offline tests and targeted lint/format/type checks pass. Free replay of
  the eight original artifacts changes only two formatting-related checks in
  Anthropic's happy path; its QPS failure remains. Four new Navigator conversations
  were recorded across two distinct configurations with no automatic reruns.
  Original sweep records remain unchanged. Review this macro step before changing
  skill wording or spending on further runs. [Correction and validation](./anthropic-correction.md),
  [original investigation](./anthropic-harness-investigation.md).
- **Handoff traffic contract — corrected and verified offline.** Navigator and
  its strict checker now preserve
  `specification.traffic_profile.expected_qps` from the successful response for
  the selected recommendation. The skill explains the buffer/minimum and retains
  Phase 2's average/peak display. Invalid or missing capacity, failed responses,
  and unrelated/later offers cannot substitute other traffic sources. All 330
  offline tests and lint/format/type checks pass. A 12-artifact counterfactual
  replay changes only the QPS check: three capacity handoffs match, six average
  handoffs do not, and three partial attempts stay failed. Original verdicts,
  judge scores, artifacts, and history remain unchanged. Both Navigator scenario
  hashes and the skill hash now identify the new contract. No paid calls were
  made. [Correction and validation](./navigator-handoff-capacity-correction.md),
  [records](./navigator-handoff-capacity-correction-results.json).
- **Fresh judged baseline — captured, follow-up needed.** All four goldens ran
  through the normal judged path with fixed judge/cutoffs and fidelity `v2`.
  Run `5ec251e832e4` passed **1/4**: consent withheld passed; navigator happy
  path failed expected-versus-peak QPS agreement despite passing judge scores;
  vague requirements and approved deployment hit turn timeouts. No approved
  deployment executed. Three attempts yielded nine metric scores with no judge
  errors. Separate launch errors and partial histories are preserved. Follow-up
  confirmed Mac sleep during both timeouts. Repeating the three failures with
  `caffeinate -i` yielded **2/3 overall passes**, three complete conversations,
  no timeouts, and nine passing judge scores. Both navigator handoffs had correct
  QPS. Approved deployment executed successfully but failed the overly broad
  precondition check described below. One repeat is not a reliability estimate;
  [report](./glm-baseline.md), [baseline records](./glm-baseline-results.json),
  and [repeat records](./glm-baseline-repeat-results.json).
- **Precondition check scope — corrected.** Each execution is checked against
  its preceding plan and the stop rule after failed runtime creation. Unrelated
  MCP errors do not block deployment; a usable replacement plan resolves an
  earlier planning failure. Structured prerequisite errors count, and later
  errors cannot hide earlier invalid executions. Approval/storage checks remain.
  All 270 offline evaluation tests and targeted lint/type checks passed. Replay
  of 25 saved artifacts changes only `692e6d32458d` to hard-check pass; original
  verdicts and judge scores are preserved. The separate replay uses new scoring
  identities and remains dry-run data. [Correction evidence](./precondition-check-fix-results.json).
  The planner's `tuple index out of range` on a cost-slot query is a separate
  observed tool defect.
- **Navigator comparison + reporting CLI — implemented offline.** The
  latest [comparison](./navigator-serial-model-comparison.md) and
  [machine-readable report](./navigator-serial-model-comparison.json) use the
  new sequential cohort, retaining skill `6b84c869d2cc` and its 7/8 verdict.
  The [earlier comparison](./navigator-model-comparison.md) and
  [JSON](./navigator-model-comparison.json) preserve the separate
  `25392e419e13` cohort and its original 7/8 verdict.
  Quality highlights tool use, task completion, and fitness (skill fidelity),
  separately from routing, overall passes, and hard checks. Efficiency covers
  user turns, model responses, input/output/reasoning/cache tokens, MCP calls
  and errors, skill loads, and agent/scoring/total seconds. Means show measured
  sample coverage; absent values stay `n/a`. Native usage is summed once per
  response, with saved hashes verified; reasoning and cached input are subsets
  of output and input respectively. Token counts are not billed EnMaaS cost.
  The latest cohort has zero measured overlap; provider load, natural caching,
  and one attempt per model/golden still limit stronger latency claims.
  Select `--cohort` or explicit original `--run-id` values: incompatible skill,
  scenario, source, judge, rubric, metric, or agent settings are rejected, as
  are unequal golden sets or sampling proportions. Dry runs, rescores, and
  duplicate attempts are excluded. No model or judge calls are made.
  Regenerate the latest tables with `make eval-compare ARGS="--cohort docs/temp/navigator-serial-sweep-results.json --artifacts-root evals/results --format markdown --output docs/temp/navigator-serial-model-comparison.md"`;
  use `--format json` with the JSON output path for machine-readable data.
  The [CLI guide](../EVALS.md#navigator-model-comparison) also covers run-ID
  selection and portable artifact paths.
  Validation: all 371 offline evaluation tests pass, along with lint/format
  and type checks for the changed reporting modules. Independent summation
  reconciles native token usage; all 129 snapshotted evidence files and 54
  history records are unchanged. Terminal, Markdown, and JSON paths were
  exercised without model or judge calls.
- **`make eval-rescore` recording.** Re-scores currently append records under a
  *new* `run_id`. The comparison now excludes these using original artifact/run
  identity and deduplicates `attempt_id`; dry runs are excluded separately.
  The writer itself remains deferred. Re-scoring currently writes the current golden's scenario
  and skill hashes onto the new score record. Preserve the original attempt's
  identity separately from the scoring configuration before comparing history;
  the new agreement expectations intentionally change scenario hashes.

---

## 6. Phase 4 leftovers

The spike works end to end. These productisation items are deferred while the
remaining work focuses on the Navigator sweep and its comparison:

- surface the recorded conversation session id as a clickable MLflow link in reports
- decide whether tracing defaults on in CI (it is opt-in via
  `RHOAI_EVAL_OTLP_ENDPOINT` today)
- MLflow currently runs via `uv run --no-project --with mlflow`; if it becomes
  standard, pin it in a dependency group
- trace retention / cleanup for the local SQLite store

---

## 7. Known gaps that are not plan items

Found while building; none are scheduled.

- **Skill template ambiguity — corrected and exercised live.** Navigator's Cluster fit row
  now has four explicit `[cluster fit]` cells. The guidance defines exactly one
  hardware-fit label per column, with `—` for a missing recommendation. The
  table checker and workflow are unchanged. All 18 existing
  table-check tests pass; the actual skill template has complete cell structure.
  Replay of all eight saved conversations preserves every check result, including
  DeepSeek's original empty-cell failure (`1dd60008ed20`). The completed sweep
  remains 7/8 under skill `25392e419e13`. The clarified template
  `6b84c869d2cc` subsequently passed table completeness in all eight cases of
  the new sequential sweep (§5), including DeepSeek's vague case. The template
  edit itself made no paid calls; the subsequent sweep was separately
  authorized. Local verification is retained in
  `evals/results/navigator-cluster-fit-placeholder-20261102-validation.json`.
- **Allowlist deviations observed, not gated** (§3.2). The deploy run called
  `list_data_connections`, `list_storage` and `prepare_model_deployment`, none
  declared in `navigator-deploy`'s `allowed-tools`.
- **Phase 0 spike artifacts** in `evals/results/` predate goldens and carry no
  `golden_id`; `rescore` skips them with a warning rather than guessing.

---

## 8. Suggested order

1. **Review the sequential sweep and comparison (§5).** The full roster is
   complete: 7/8 passed, all eight table checks passed, and all 28 interval
   pairs show zero overlap. No additional sweep is pending. Preserve both
   historical cohorts and their distinct failure cases. These predate the
   hardware-consent/planner corrections and retain their original verdicts.
2. **Review the completed GLM investigation.** The missing block is real;
   instructions and values were available, and no local parser/harness defect
   was reproduced. Preserve the failure and the generation/gateway attribution
   limit. Do not infer a regression from one sample.
3. **Review the completed API diagnostic and hardware corrections.**
   Both APIs passed matched replay; Chat also passed 1/2 full flows, with a
   different failure mode. This does not establish an API fix. The new gate
   catches `2aaa97922492`; the shared planner filter is corrected and a bounded
   probe supports the two-sentence consent clarification. Review the minimal
   code/skill changes and [evidence](./glm-root-cause-correction.md).
4. **Validate GLM on the two Navigator goldens after review.** Use a separate
   cohort with the corrected backend, skill, and grading hashes. Check the
   full inventory/fallback/table/handoff flow, retaining any failure. If a new
   four-model comparison is later requested, every model needs the same shared
   configuration. Pi-style effort/cap/penalty/streaming experiments and
   raw-reasoning viewer support remain separate work.

**Separately deferred:** the Phase 1 `get_cluster_resources` rewrite (§3) remains
outside this branch's current evaluation work by decision. Navigator-deploy
sweeps, new deployment-specific follow-up, and Phase 4 polish (§6) are also
deferred under the current cost scope.
