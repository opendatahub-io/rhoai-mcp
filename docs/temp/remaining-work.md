# Remaining work: Pydantic AI + DeepEval skill evaluations

**Status:** living checklist · **Date:** 2026-10-31
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
| Shared transcript contract + artifact save/load | `evals/agent/transcript.py` |
| 23 deterministic checks (routing, table, consent, ordering) | `evals/skills/checks.py` |
| 4 goldens with checkpoints | `evals/skills/goldens.py` |
| MCP metrics + per-skill fidelity rubrics | `evals/skills/rubrics.py` |
| Re-score saved artifacts with no agent run | `make eval-rescore` |
| Offline trajectory viewer | `make eval-show` |
| OTel traces into a local MLflow | `make mlflow-up` + `make eval-skills-traced` |

All four goldens have run live against `enmaas-glm`; judged scoring works with
EnMaaS `gpt-5.6-sol` as judge; traces land in MLflow as one trace per
conversation with full content and working `trace_id` correlation.

---

## 2. The blocking issue: judge variance

**Nothing downstream of this is trustworthy until it is resolved**, which is why
it is listed before the phase work.

Re-scoring one **frozen** transcript (`navigator-happy-path`, attempt
`c7b261e66bfb`) with the same judge gave `Multi-Turn MCP Use`:

```
0.250   →   0.150   →   0.250   →   0.217
```

A ±0.1 spread on identical input, against a `0.5` threshold. So "0.25 fails"
currently says more about sampling than about the agent. Note this is *judge*
non-determinism, independent of the agent non-determinism already documented
(no matrix cell can set `temperature`).

Also uncalibrated: `task_completion_threshold` defaults to `0.25`, which is
permissive enough that almost anything passes.

**What to do**

1. Re-score a fixed set of transcripts N times (N≥5) per metric and record the
   spread. `make eval-rescore` makes this cheap — that is what it is for.
2. Decide thresholds from reviewed passing/failing examples plus the observed
   spread, not from whichever number the current agent happens to produce
   (§3.3).
3. Re-open §6 Q2 with evidence: is EnMaaS Sol a good enough judge, or does a
   stronger judge materially reduce variance? Worth answering *before*
   investing in calibration against a noisy judge.
4. Consider recording per-metric variance alongside the score so a single run's
   number is never read as precise.

**Acceptance:** a documented threshold per metric, justified by measured spread
and reviewed examples, with the judge model pinned.

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

### 4.1 `run_conversation` has no tests — highest priority here

The loop that drives checkpoints, stops the script on failure, and preserves
partial history on timeout is now load-bearing for every golden, and nothing
verifies it. Needs a mocked agent.

Cases worth pinning: checkpoints run *before* the reply and in order; a failed
checkpoint stops the script and leaves `error_stage` set; a timeout or
usage-limit breach still yields a saveable partial history; `_extend` does not
duplicate turns; the MCP session stays open across turns.

### 4.2 `navigator-deploy-approved` is one turn short

It reaches `plan_deployment`, then asks for a storage location and runs out of
script. The newly added `deploy_execution_reached` check now fails it honestly
rather than passing a deployment golden that never deployed.

### 4.3 Two §3.1 "agreement" checks are missing

Both are harder than the presence checks already implemented, and both guard the
same failure: the right tool called with the wrong selection.

- The handoff block **agrees with** the confirmed workload and the selected
  recommendation. Currently only presence and completeness are checked.
- Execution parameters **match the reviewed plan** and the confirmed
  model/configuration. §3.1: *"calling the right tool with a different
  selection is not success."*

---

## 5. Phase 3 leftovers

- **Full sweep.** Only `enmaas-glm` has run goldens. The matrix is 3 models × 4
  goldens; `make eval-skills-matrix` exists but has never been run.
- **`navigator-deploy-fidelity`** rubric is written but has never been exercised
  live. Only the navigator rubric has produced a score (0.90).
- **`comparison.py` + reporting CLI.** Untouched, correctly deferred until
  multi-model records exist. When it lands it must compare only matching
  skill/golden sets under a fixed judge/rubric version, show sample counts, and
  leave missing values `n/a` rather than substituting `0.0` as the current
  implementation does (§3.5).
- **`make eval-rescore` recording.** Re-scores currently append records under a
  *new* `run_id`. Pooling them with live runs in a future comparison would
  double-count one agent run; the comparison layer must filter on `scoring_id`
  / `dry_run`.

---

## 6. Phase 4 leftovers

The spike works end to end. What remains is productisation:

- surface `trace_id` as a clickable MLflow link in reports
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

1. **Judge variance (§2)** — invalidates every other number until settled.
2. **`run_conversation` tests (§4.1)** — load-bearing and unverified.
3. **Deploy golden turn + agreement checks (§4.2, §4.3)** — small, and they
   close real holes in what the goldens claim to prove.
4. **Full model sweep (§5)** — only meaningful once thresholds are defensible.
5. **`comparison.py` (§5)** — needs multi-model records to exist first.
6. **Phase 1 rewrite (§3)** — unblocks the unexercised branches; independent of
   the above and gated on the product decision.
7. **Phase 4 polish (§6)** — nothing depends on it.
