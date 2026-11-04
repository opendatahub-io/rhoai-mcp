# Navigator sweep with DeepSeek judge

**Date:** 2026-11-03 · **Source:** `7c706ea` plus frozen per-file hashes · **Scope:** Navigator only

**8/8 conversations recorded; 7/8 passed.** GPT, Claude, GLM and DeepSeek all ran afresh through EnMaaS, using the same corrected Navigator skill `88cbf1008ce0`, backend, scenarios and DeepSeek judge. Each model ran vague requirements, then happy path; no pytest reruns were allowed. Earlier Sol-judged sweeps remain historical and are not pooled with these scores.

The [comparison report](./navigator-deepseek-judge-model-comparison.md) presents quality, usage, tool errors and timing. Its [JSON export](./navigator-deepseek-judge-model-comparison.json) preserves original records and scorecards. The [sweep evidence](./navigator-deepseek-judge-sweep-results.json) includes the frozen configuration, source/native hashes, intervals, checks and complete metric reasons.

GPT, Claude and DeepSeek passed 2/2 each. GLM passed its vague-requirements case; its happy-path conversation completed all five turns and passed every hard check, but MCP judging timed out after 180 seconds. Both MCP metric slots retain error sentinels in the original records and display as `n/a` in the comparison; fidelity was not produced. This is incomplete judging, not an observed agent-quality failure. GLM's quality means have 1/2 coverage and must not be ranked directly against the fully scored pairs.

There are **21 valid scores out of 24 expected**, all above their cutoffs; two metric slots errored and one is missing. No judge or conversation was rerun. The recorded 7/8 verdict remains unchanged.

## Execution and results

Zero overlapping case intervals and zero overlapping model batches. All 28 pairs of eight whole-test intervals were checked on one host monotonic clock, including setup, agents, scoring and teardown. Each batch used `caffeinate -i`; 0 sleep/wake events were recorded in the sweep window.

| Agent | Run | Passed | Agent seconds | Scoring seconds | Batch start (UTC) | Batch end (UTC) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| enmaas-sol | `8b1f4d45dbdd` | 2/2 | 97.12 | 216.29 | 2026-11-03T08:45:29.711282+00:00 | 2026-11-03T08:50:45.266886+00:00 |
| enmaas-opus | `01b72fad40b3` | 2/2 | 181.31 | 371.09 | 2026-11-03T08:50:45.279927+00:00 | 2026-11-03T08:59:59.896021+00:00 |
| enmaas-glm | `5ad040f0609d` | 1/2 | 602.65 | 299.82 | 2026-11-03T08:59:59.908491+00:00 | 2026-11-03T09:15:04.793407+00:00 |
| enmaas-deepseek | `da04988dcff6` | 2/2 | 84.60 | 345.57 | 2026-11-03T09:15:04.809154+00:00 | 2026-11-03T09:22:17.916167+00:00 |

Routing passed 8/8; 8/8 completed five turns. Hard checks passed 8/8. Judge measurements: 21/23 passing; missing metric entries: 1. Agent timeouts: 0; model failures: 0; harness errors: 0; cases with judge errors: 1; deployment writes: 0.

## Observations within passing cases

GPT's vague-case completion scored 0.8529: the judge flagged a Performance slot displayed as available after a tighter search returned no match. Claude's happy-case completion also scored 0.8529: a planner error (`tuple index out of range`) was described as finding no cheaper configuration. Claude's tool-use scores of 0.9643 / 0.9545 reflect a small tighter-cost-limit rounding deviation. These are judge observations, not independently reviewed labels; full item evidence is retained. GLM's happy conversation retained a separate planner error (`bad parameter or other API misuse`); its judge scores are unavailable.

All seven available fidelity grades were 1.0, despite the completion caveats above. Fidelity alone therefore does not distinguish these observed defects. DeepSeek's two agent cases scored 1.0 on all metrics; two samples do not establish reliability or a general quality advantage.

## Configuration and interpretation

The judge is EnMaaS `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017`. MCP use and completion use `conversation-v3-items`; fidelity retains G-Eval's 0–10 grade normalized to 0–1 and evidence version `v2`. Cutoffs remain 0.60 / 0.75 / 0.80. Full item scorecards and unrounded numbers are retained. MCP scoring does not reuse cached scores. No paid-judge fallback was configured.

Scenario hashes: {"navigator-happy-path": "07ced00d7013", "navigator-vague-requirements": "f349e686f7ff"}. All models use identical current skill/backend/checks. GPT, GLM and DeepSeek use Responses; Claude uses native Messages with adaptive thinking. Temperature is unspecified for all models. Limits are 300 seconds, 30 model requests and 40 tool calls per scripted turn. The cluster is mocked, embedded planner GPU probing is disabled, both skills are discoverable, and tracing is off. DeepEval scratch storage was fresh for the sweep; provider caches were not reset. SDK retries retain the normal implementation; zero pytest reruns means no repeated conversations.

One attempt per model/scenario is a smoke comparison, not a reliability estimate or statistically established latency ranking. Sequential scheduling removes local evaluation overlap; shared EnMaaS load, natural caching, provider defaults and token accounting remain variable. Agent time excludes scoring. Currency cost is unavailable. Navigator-deploy goldens are excluded.

DeepSeek fidelity has a known missing-handoff false-pass limitation. Completion and deterministic handoff checks remain required. Judge/infrastructure errors remain unscored failures; passing scores do not prove flawless behavior. See the [judge migration validation](./deepseek-item-judge-migration.md) and [scoring contract](../JUDGE-SCORING.md). Changes in both the skill/backend and judge contract prevent attributing differences from older sweeps solely to the judge.

## Recorded failures

- `54a6fd8de32a` (enmaas-glm, navigator-happy-path): Conversation MCP Use: -1.0 — TimeoutError: call timed out after 180s (per attempt). Increase DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE (None disables) or reduce work per attempt.; Conversation MCP Task Completion: -1.0 — TimeoutError: call timed out after 180s (per attempt). Increase DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE (None disables) or reduce work per attempt.

## Provenance and reproduction

The freeze records 292 source/config/test/driver files, model profiles, packages, judge, thresholds and limits. At verification, 172 earlier native JSON files and 56 earlier report/plan files were unchanged. The original 66 history records remain a byte-identical prefix; exactly eight new records were appended. Historical verdicts were not changed or rescored.

Offline validation recomputed all 21 retained item scorecards from their ratings, weights and caps; every scored MCP metric matches its cards. Comparison records match the original cohort exactly, and relative report links resolve. Evaluation preflight passed 67 offline checks covering scoring, comparison reporting, hardware ordering and the judge client.

Reproduce the comparison from saved evidence without model calls:

```bash
make eval-compare ARGS="--cohort docs/temp/navigator-deepseek-judge-sweep-results.json --artifacts-root evals/results --format markdown --output docs/temp/navigator-deepseek-judge-model-comparison.md"
```

Use `--format json` and the JSON output path for machine-readable statistics. The separate sweep evidence retains scheduling verification. Local execution driver: `evals/results/run_navigator_deepseek_judge_sweep_20261103.py`; offline report builder: `evals/results/report_navigator_deepseek_judge_sweep_20261103.py`. Logs, JUnit and timing evidence use `evals/results/navigator-deepseek-judge-sweep-20261103-*`.

After producing both standard comparison formats, run `.venv/bin/python evals/results/annotate_navigator_deepseek_judge_report_20261103.py` to reproduce the presentation with four-decimal means, explicit metric coverage, scheduling notes and passing-score caveats. This offline presentation step verifies that every original selected record remains identical; its path and SHA-256 are retained in the comparison JSON. The execution freeze covers the evaluated code and original report builder; the presentation script was added during the sweep and does not alter the evaluation configuration.
