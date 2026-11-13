# Plan: a minimal, replayable skill evaluation suite

**Date:** 2026-11-12

**Status:** proposal; this change adds the plan only.

**Reviewed:** `f91a661` against the local `upstream/main` merge base, `cde06e0`.

The proposed PR should demonstrate one complete workflow: run a Pydantic AI
agent with `rhoai-mcp` and both Navigator skills, save its execution, and evaluate
that saved execution with deterministic checks and DeepEval judges. Keep a short
terminal result and machine-readable results. Remove the experiment-management
and reporting infrastructure surrounding that workflow.

“Minimal” means the smallest useful **net diff against upstream**, including
tests, documentation, dependencies, and CI. It does not mean moving the current
framework into fewer, larger files or merely shortening its command names.

## 1. What makes the branch large

The reviewed diff contains 126 changed files, 66,912 insertions, and 200
deletions. These figures describe the existing branch before this plan:

| Area | Changed files | Added lines | Implication |
| --- | ---: | ---: | --- |
| `docs/temp/` | 60 | 52,453 | Investigation notes and generated evidence dominate the diff. |
| `evals/` | 35 | 7,986 | The suite includes probes, calibration, tracing, and advanced reporting. |
| `tests/evals/` | 16 | 5,180 | Several test modules support those optional facilities. |
| `uv.lock` | 1 | 738 | Retain the lockfile, regenerated for the retained dependencies. |
| Other files | 14 | 555 | Includes skill/planner fixes and changes to the older evaluation path. |

There is already useful infrastructure on upstream: the LCS scenarios, judge
providers, metric factories, JSONL recorder, and report commands. Deleting that
pre-existing suite would enlarge the PR. The reduction should primarily remove
this branch's additions and avoid expanding the older suite's responsibilities.

The existing design worth retaining is in
[`attempt.py`](../../evals/skills/attempt.py): agent execution → save artifact →
reload artifact → checks and judges → record results → pytest assertion.

## 2. Recommended scope

Keep one default agent profile and one independently configured judge. Retain
the current EnMaaS DeepSeek judge and existing scoring semantics. Extra agent
profiles can remain small data entries if useful; probes, diagnostic profiles,
matrix-specific commands, and sweep orchestration are outside the minimal PR.

Both `.claude/skills/navigator` and `.claude/skills/navigator-deploy` remain in
the discovery catalog. The agent must select and load the relevant skill from
the natural-language request. Keep the existing shared host instruction that
requires relevant skill loading, rather than injecting the chosen skill body.

Retain the four existing scenario definitions:

| Scenario | Property demonstrated |
| --- | --- |
| Navigator happy path | Workload confirmation, recommendations, selected-model handoff. |
| Navigator vague requirements | Clarification before recommending. |
| Deploy approved | Review the selected plan, obtain approval, execute against the mock cluster. |
| Deploy consent withheld | Honour refusal and perform zero deployment writes. |

Use pytest selection to run either skill. A full-suite command should cover
both skills; the current Make targets select only Navigator. Keeping deployment
scenarios does not require a new deployment sweep while implementing this plan.

Execution uses the real MCP server over stdio with its mock cluster and embedded
planner. Keep one MCP session per conversation and fresh state per attempt.
Keep turn timeouts, usage limits, partial-history capture, and checkpoints before
scripted replies. Disable the embedded planner's independent Kubernetes probing
in the mock subprocess, as the current runner already does.

### Preserve both GPU discovery and its permission-failure fallback

The old plan and `goldens.py` assume `get_cluster_resources` is unavailable by
default. The current server registers it through
[`cluster/tools.py`](../../src/rhoai_mcp/composites/cluster/tools.py) and
[`cluster/inventory.py`](../../src/rhoai_mcp/composites/cluster/inventory.py).
Registration does not guarantee discovery succeeds: Kubernetes permissions can
prevent the tool from reading the required resources. A present tool can also
return incomplete inventory. The manual customer-inventory path remains valid
and essential; keep it in the suite.

The Navigator hardware check currently requires a later scripted customer GPU
reply. Audit the gate against the actual discovery result: accept usable GPU
inventory returned before a recommendation, or the subsequent customer reply
when discovery fails or cannot identify the hardware. Unfiltered recommendations
require the customer's explicit consent. A failed discovery call, an ambiguous
result, or “go ahead” alone is insufficient evidence.

Keep permission-denied results in the saved execution. Do not require additional
cluster permissions to run the suite or treat denied reads as proof of no GPUs.
Cover both successful discovery and denied/incomplete discovery with focused
offline cases; no extra paid sweep is needed. Change scripts/checks only where
they assume one branch unconditionally, and version any changed scenario/check
contract. Historical runs retain their original verdicts.

## 3. The execution record is the boundary

```text
Pydantic AI + both skills + stdio MCP
                 |
        save native execution
                 |
        load saved execution <---- re-evaluate command
                 |
       deterministic checks + DeepEval
                 |
          JSONL + terminal result
```

Keep the current `artifact.json` + `messages.json` layout. Native messages and
captured MCP schemas are the source of truth; serialized DeepEval cases and
OTel spans are unnecessary as replay inputs. Continue constructing DeepEval
cases from the loaded record for initial scoring as well as re-evaluation.

The essential saved evidence is:

- Native messages, including tool arguments/results, errors, skill discovery,
  and the instructions actually delivered to the model.
- MCP tool schemas from the same session, task, final output, completion state,
  and failure stage for partial attempts.
- Original run/attempt identity, agent profile/settings, source revision, skill
  and scenario hashes, and the scenario's grading intent and expected values.

Keep one small transcript adapter for checkpoints, checks, and DeepEval.
Matching tool calls to returns **by call ID**, distinguishing skill-loading
tools from MCP tools, and preserving tool errors are essential adapter logic.
These rules have regression evidence in the current branch.

Each evaluation gets a new `scoring_id` while retaining the original `run_id`
and `attempt_id`. Its scoring metadata identifies the checks/rubrics, judge
model/endpoint, thresholds, and evaluator revision. Re-evaluation appends a
result; it never rewrites the execution or an earlier score.

Correct two existing replay limitations while reducing the code:

1. `rescore.py` currently creates a new recorder run ID. Preserve the artifact's
   original run identity in the score record instead of inferring it from paths.
2. `score_artifact()` currently stamps the current golden's skill/scenario hashes
   onto a replay. Preserve the original generation metadata separately. Use saved
   scenario intent for judge context; do not silently substitute a changed
   scenario. For older artifacts lacking that snapshot, require a matching
   scenario hash or report the incompatibility explicitly.

Re-evaluation never constructs an agent or starts MCP. Deterministic replay
needs no provider credentials or network. Judged replay calls only the judge;
its cost and availability depend on the configured provider.

## 4. Essential checks and metrics

Retain the existing checks needed by the four scenarios: successful skill
routing, confirmation and hardware ordering, table/handoff completeness,
agreement with confirmed workload and selected recommendation, and deployment
plan/approval/execution consistency. Refusal requires zero writes. An approved
deployment requires successful execution; a truthful provisioning handoff can
satisfy the scenario without claiming an endpoint is ready.

Simplify helpers around these concrete scenarios where possible. Keep the
regressions for incorrect call attribution, premature recommendations, missing
handoffs, wrong selections, and unauthorized writes. Do not replace those
checks with phrase matching or an LLM judgment to save lines.

Retain the current three judged dimensions and thresholds:

| Metric | Current implementation | Threshold |
| --- | --- | ---: |
| MCP use | Conversation-wide item scoring for tool choice and arguments | 0.60 |
| Task completion | Conversation-wide item scoring for the requested outcome | 0.75 |
| Skill fidelity | Per-skill Conversational G-Eval rubric | 0.80 |

Keep `conversation.py`, `scorecards.py`, and the per-skill rubrics in this
reduction. The branch documents why stock task splitting lost phase-gated
context, and why fidelity alone missed missing handoffs. Replacing these
metrics would change the measurement contract and require separate validation.
Deleting the calibration *tooling* does not require changing validated rubrics.
See [the scoring contract](../JUDGE-SCORING.md) and
[judge migration evidence](./deepseek-item-judge-migration.md).

An overall pass requires a completed eligible execution, correct routing, all
mandatory checks, and **all three required metric results** passing. Checking
only that every returned metric passes is insufficient if one is missing.
Keep check failures, execution errors, judge errors, skipped judging, and
unexercised branches distinguishable. Checks-only execution must be visibly
labelled `CHECKS ONLY`, never presented as an overall judged pass.

## 5. Reporting: terminal result plus JSONL

Reuse `EvalRecorder` and the existing result dataclasses with only the fields
needed for execution identity, scoring identity/configuration, checks, metric
scores/thresholds/reasons/item evidence, and artifact location. Keep additive
fields readable with defaults for older records. Preserve raw score precision.

Write new skill results to a dedicated `evals/results/skill_history.jsonl`
using the recorder's existing path parameter. This avoids teaching every legacy
report how to combine LCS runs, skill runs, and rescoring history. Existing
local histories remain intact.

Use one small renderer for the just-produced results, shared by initial scoring
and replay. For example:

```text
SCENARIO             AGENT         STATUS       CHECKS   USE     COMPLETE  FIDELITY
navigator-happy-path  enmaas-sol    PASS         pass     0.9583  0.9706    0.9000
attempt=<id> scoring=<id> artifact=evals/results/<run>/<attempt>
```

Print failed checks and metric errors/reasons below the row. JSONL retains the
full evidence. Pytest supplies the run summary. Viewing results later needs
only reading the saved JSONL, without running checks, agents, or judges.

Remove new cohort selection, model ranking/comparison, trends, overlap analysis,
token/cache accounting, multiple export formats, and MLflow correlation from
this PR. Restore the pre-existing comparison/trend commands to their upstream
behaviour. A standalone transcript viewer is optional follow-up work.

## 6. File-level reduction

| Files | Proposed treatment |
| --- | --- |
| `evals/agent/runner.py`, `models.py`, `transcript.py` | Keep the execution/record boundary; remove tracing hooks and diagnostic model profiles. Trim explanatory repetition after the contract is documented. |
| `evals/skills/goldens.py`, `checks.py`, `agreement.py`, `rubrics.py` | Keep the four scenarios and substantive checks/rubrics; correct the obsolete GPU-discovery assumption. Avoid building a general workflow engine. |
| `evals/skills/attempt.py`, `conftest.py`, `test_navigator*.py` | Keep a thin run/save/score/assert path and minimal pytest fixtures. Remove tracing/allowlist-observation plumbing. |
| `evals/skills/rescore.py` | Keep an explicit saved-attempt/run selector, optional judging, shared scoring, result rendering, and meaningful exit status. Remove filters/listing needed only by the viewer. |
| `evals/deepeval_helpers.py`, `metrics/config.py` | Keep the transcript conversion and necessary DeepEval compatibility fixes; retain the older LCS entry points. |
| `evals/metrics/conversation.py`, `scorecards.py` | Keep current scoring semantics and compatibility tests. |
| `evals/config.py`, `providers/factory.py`, `providers/judge.py` | Keep EnMaaS support and required schema handling. Apply skill judge/threshold defaults at the skill boundary; avoid changing older LCS defaults just to support this example. |
| `evals/reporting/models.py`, `recorder.py`, `reader.py` | Keep only essential additive schema, score extraction, persistence/readback, and verdict logic. |
| `evals/reporting/skill_comparison.py`, `observations.py` | Exclude from the minimal PR, with their feature-specific tests. |
| Branch changes to reporting `cli.py`, `comparison.py`, `trending.py`, `formatting.py` | Restore upstream behaviour; retain only a tiny shared formatter if the minimal result actually needs it. |
| `evals/agent/probe_tool_calling.py`, `spike.py`, `tracing.py`; `evals/skills/calibrate.py`, `show.py` | Exclude, together with their dedicated tests, Make targets, environment examples, and trace dependencies. Preserve research in the existing branch history. |
| `.claude/skills/navigator/SKILL.md`, planner production changes and their dedicated tests | Split into prerequisite fix PRs/commits. The suite should measure the shipped skills/server. Keep their regressions as evaluation evidence; do not hide required fixes in the harness. |
| `tests/composites/test_skill_allowed_tools.py` | Keep a concise check against the actual default tool catalog; remove historical exception scaffolding. |
| `.github/workflows/ci.yml` | Keep one offline harness job; remove `--group trace`. Include checks for the retained eval code. |
| `.github/workflows/eval.yml` | Restore the older LCS workflow defaults; a new hosted skill-sweep workflow is outside the minimal PR. |
| `pyproject.toml`, `uv.lock`, `.gitignore` | Keep `skill-eval`, required pins/lockfile, and generated-result ignores; remove trace/MLflow additions. |
| `Makefile`, `.env.eval.example` | Keep run, judged run, and replay entry points plus essential agent/judge settings. |
| `docs/EVALS.md`, `docs/JUDGE-SCORING.md`, `docs/temp/` | Publish a short, current usage/scoring guide. Exclude historical investigation reports and generated JSON from the final PR after preserving the research reference. Keep this requested plan in `docs/temp` during planning. |

Avoid renaming the retained packages or introducing another configuration,
plugin, report, or artifact abstraction. Existing filenames are adequate.

The reduced usage guide should need only these operations:

```bash
# Run both skills; agent runs, judges are skipped.
make eval-skills

# Run both skills and their judges.
make eval-skills-judge

# Re-evaluate an existing execution, with checks only or checks and judges.
make eval-rescore ARGS="--attempt <attempt-id>"
make eval-rescore ARGS="--attempt <attempt-id> --judge"
```

These describe the proposed Make target scope. The current run targets select
Navigator only. Checks-only live runs still call the agent; checks-only replay
does not call any model. Use pytest's existing file/`-k` selection for subsets.

## 7. Implementation order and validation

1. **Preserve the reference and define the boundary.** Retain `f91a661` and
   local saved executions/history before narrowing the diff. Separate the
   production skill/planner fixes and identify the upstream base for the suite.
   Inventory existing imports so removing an optional module removes all its
   callers, fixtures, tests, configuration, and documentation references.
2. **Remove optional facilities first.** Drop probes, spike, calibration CLI,
   tracing, viewer, advanced comparisons, and their dependencies/tests. Restore
   unrelated legacy workflow/reporting changes. Measure the resulting diff.
3. **Make replay self-contained.** Preserve original execution provenance,
   capture grading intent for new records, distinguish scoring identity, and
   share one saved-artifact scoring function. Keep existing artifact readability
   where metadata is sufficient; explicitly reject incompatible inputs.
4. **Reduce reporting and fixtures.** Use the dedicated JSONL destination and
   shared compact renderer. Keep the two skill test files or one parametrized
   file, whichever produces the clearer diff; no package-wide reshuffle.
5. **Align scenarios and finish the guide.** Correct discovery/fallback gates,
   document both skills, commands, records, and pass semantics. Reconcile saved
   replay outcomes against the old implementation; explain every intended
   difference, including the discovery correction and stricter missing-metric
   handling.
6. **Validate and review the net PR.** Run retained offline tests, lint/format,
   targeted typing, dependency checks, and `git diff --check`. Review the final
   file list against the keep/remove table. Report actual remaining file and
   line counts; avoid a line-count target that rewards deleting correctness.

Keep focused offline coverage for these boundaries:

- Native artifact round-trip, call-ID association, skill-vs-MCP separation,
  delivered instructions, structured results/errors, and partial executions.
- MCP-session/history lifecycle, checkpoints before consent, timeouts/limits,
  and mock planner isolation.
- Scenario gates, selection/handoff agreement, deployment approval/refusal,
  and usable automatic discovery versus permission-denied/incomplete discovery
  requiring customer inventory or explicit unfiltered consent.
- Current scorecard arithmetic/caps/schema validation and DeepEval integration;
  missing required metrics, judge errors, and checks-only pass semantics.
- Initial scoring versus replay equality under the same scoring configuration;
  replay makes zero agent/MCP calls, preserves original identities, and appends
  a distinct evaluation without changing native evidence or earlier results.
- Essential record write/read compatibility and the concise result output.

Do not add new tests for the planning document or retain tests solely to justify
removed features. Full paid model sweeps are unnecessary for a structural
reduction. Any later live validation should be bounded to the changed execution
or scenario behaviour and recorded as new evidence; historical deployment and
Navigator results do not validate a changed script automatically.

## 8. Completion criteria

A reader can find the agent wiring, four scenarios, checks/judges, and shared
record-based scoring path without reading experiment-management code. Running
one scenario saves a reusable execution even on failure. Re-evaluating it works
without an agent or cluster, produces a new identified score, and preserves the
original evidence. Both skills remain covered. Terminal output and JSONL answer
what passed, what failed, and where the evidence lives.

The final PR contains the suite, necessary compatibility changes, focused tests,
minimal setup/CI, and a short guide. Production fixes have an explicit dependency
or separate review. Historical reports, probes, tracing, calibration tooling,
and advanced reporting remain available through the preserved research history.
