# Remaining work: Pydantic AI + DeepEval skill evaluations

**Status:** living checklist · **Updated:** 2026-11-02
**Parent plan:** [`pydantic-ai-skill-evals-plan.md`](./pydantic-ai-skill-evals-plan.md)
**Phase 0 findings:** [`phase0-model-probe.md`](./phase0-model-probe.md)

What is built, what is not, and what order the remainder is worth doing in.
Section references (§) are to the parent plan.

---

## 1. Built and working

| Capability | Entry point |
|---|---|
| Model roster set on evidence (7 candidates probed, all qualified) | `evals/agent/probe_tool_calling.py` |
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
normal run history are unchanged. A fresh judged one-model baseline is next.

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

**Permanently unexercised while this stands** (§3.1, §3.4):

- automated GPU discovery
- the GPU product → planner-type substring mapping table, in *both* skills
- cluster-constrained `recommend_model` and its "requires new hardware" fallback

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

The skills deliberately use different traffic sources: navigator handoff QPS
comes from `get_expected_rps`; deployment plan QPS comes from the selected
recommendation's specification. Numeric comparisons allow three-decimal QPS
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

- **Full sweep.** Only `enmaas-glm` has run goldens. The matrix is 3 models × 4
  goldens; `make eval-skills-matrix` exists but has never been run.
- **Fresh judged baseline.** Both fidelity rubrics now have repeated frozen-corpus
  scores. Run all four goldens through the normal judged path with fidelity `v2`
  before expanding to the full model matrix.
- **`comparison.py` + reporting CLI.** Untouched, correctly deferred until
  multi-model records exist. When it lands it must compare only matching
  skill/golden sets under fixed judge, rubric, and metric versions, show sample counts, and
  leave missing values `n/a` rather than substituting `0.0` as the current
  implementation does (§3.5).
- **`make eval-rescore` recording.** Re-scores currently append records under a
  *new* `run_id`. Pooling them with live runs in a future comparison would
  double-count one agent run; the comparison layer must filter on `scoring_id`
  / `dry_run`. Re-scoring also currently writes the current golden's scenario
  and skill hashes onto the new score record. Preserve the original attempt's
  identity separately from the scoring configuration before comparing history;
  the new agreement expectations intentionally change scenario hashes.

---

## 6. Phase 4 leftovers

The spike works end to end. What remains is productisation:

- surface the recorded conversation session id as a clickable MLflow link in reports
- decide whether tracing defaults on in CI (it is opt-in via
  `RHOAI_EVAL_OTLP_ENDPOINT` today)
- MLflow currently runs via `uv run --no-project --with mlflow`; if it becomes
  standard, pin it in a dependency group
- trace retention / cleanup for the local SQLite store

---

## 7. Known gaps that are not plan items

Found while building; none are scheduled.

- **Skill template ambiguity.** The navigator comparison-table template's
  `Cluster fit` row shows one value cell followed by three empty cells. It is
  well-formed (5 cells) but a model copying it literally produces a row the
  `table_cells_complete` check rejects. Whether to tighten the skill wording is
  a judgement call; the goldens will show whether models actually fill it.
- **Allowlist deviations observed, not gated** (§3.2). The deploy run called
  `list_data_connections`, `list_storage` and `prepare_model_deployment`, none
  declared in `navigator-deploy`'s `allowed-tools`.
- **Phase 0 spike artifacts** in `evals/results/` predate goldens and carry no
  `golden_id`; `rescore` skips them with a warning rather than guessing.

---

## 8. Suggested order

1. **Fresh judged one-model baseline (§5)** — all four goldens with the fixed
   judge/cutoffs and fidelity `v2`.
2. **Full model sweep (§5)** — expand once the one-model baseline is understood.
3. **`comparison.py` (§5)** — needs multi-model records to exist first.
4. **Phase 4 polish (§6)** — nothing depends on it.

**Separately deferred:** the Phase 1 `get_cluster_resources` rewrite (§3) remains
outside this branch's current evaluation work by decision.
