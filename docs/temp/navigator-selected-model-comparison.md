# Navigator selected-run comparison — mixed configurations

**Requested selection:** latest corrected GLM pair plus the previous sequential GPT, Claude, and DeepSeek pairs, all through EnMaaS. **This is a descriptive comparison across two configurations.** No new evaluation ran.

The standard comparison CLI rejects this selection because the skill/source/scenario versions differ. Each source cohort was validated separately with the existing comparison code, then its selected model rows were presented together. Original hashes, scores, verdicts, and API settings are retained.

## Selection and versions

| Model           |                         Provider model ID |  Source |        Skill |          Run |
| --------------- | -----------------------------------------: | -------: | ------------: | ------------: |
| GPT             |                               gpt-5.6-sol | f0bd975 | 6b84c869d2cc | 6427f23f8fe9 |
| Claude Opus     |                           claude-opus-4-6 | f0bd975 | 6b84c869d2cc | c1aed9dbd1c1 |
| GLM (corrected) |                      rits/zai-org/glm-5-3 | 44d6774 | 96cc8596d4dd | 8fa71ecb03e2 |
| DeepSeek        | rits/deepseek-ai/DeepSeek-V4.1-Flash-1017 | f0bd975 | 6b84c869d2cc | b77ef7e8878d |

Both groups use the same two customer scripts, judge (`enmaas/gpt-6.1-sol`), metric versions, cutoffs (0.60 / 0.75 / 0.80), and limits. GLM additionally has the hardware-consent clarification, corrected embedded GPU filter, and new hardware-ordering check. Its scenario hashes version that grading change. **Recorded hard checks and fidelity are relative to each run's own contract.**

The older Claude vague case received four H100 slots from an A100-filtered tool call, as documented in the [planner investigation](./glm-root-cause-correction.md). That backend defect is fixed only in the newer GLM pair here. The table cannot isolate model quality from that change.

## Quality — means [measured/selected]

| Model           | N | Recorded passes | Routing | Recorded hard checks |    Tool use |  Completion | Fitness / fidelity |
| --------------- | ---: | ---------------: | -------: | --------------------: | -----------: | -----------: | ------------------: |
| GPT             | 2 |             2/2 |     2/2 |                  2/2 | 0.975 [2/2] | 0.975 [2/2] |        0.950 [2/2] |
| Claude Opus     | 2 |             2/2 |     2/2 |                  2/2 | 0.875 [2/2] | 0.900 [2/2] |        0.900 [2/2] |
| GLM (corrected) | 2 |             2/2 |     2/2 |                  2/2 | 0.875 [2/2] | 0.925 [2/2] |        0.900 [2/2] |
| DeepSeek        | 2 |             2/2 |     2/2 |                  2/2 | 0.950 [2/2] | 0.950 [2/2] |        0.900 [2/2] |

All eight selected records passed under their recorded configurations. This does not revise the historical roster's 7/8 verdict or constitute a new 8/8 sweep. The two older GLM attempts were excluded by the requested selection, not retried or rescored.

## Efficiency — means per conversation [measured/selected]

| Model           | Customer turns | Model responses |  Output tokens |      Agent s |   Scoring s |      Total s |
| --------------- | --------------: | ---------------: | --------------: | ------------: | -----------: | ------------: |
| GPT             |      5.0 [2/2] |       9.0 [2/2] |  2,589.0 [2/2] |  50.44 [2/2] | 32.86 [2/2] |  83.30 [2/2] |
| Claude Opus     |      5.0 [2/2] |       9.0 [2/2] |  5,071.5 [2/2] | 114.25 [2/2] | 39.26 [2/2] | 153.51 [2/2] |
| GLM (corrected) |      5.0 [2/2] |       9.0 [2/2] | 10,798.5 [2/2] | 182.53 [2/2] | 39.64 [2/2] | 222.18 [2/2] |
| DeepSeek        |      5.0 [2/2] |       9.0 [2/2] |  5,374.5 [2/2] |  77.96 [2/2] | 33.92 [2/2] | 111.88 [2/2] |

## Usage and tools — means per conversation [measured/selected]

| Model           |    Input tokens | Reasoning tokens |    Cached input | MCP calls | MCP errors | Skill loads |
| --------------- | ---------------: | ----------------: | ---------------: | ---------: | ----------: | -----------: |
| GPT             | 119,718.5 [2/2] |      910.5 [2/2] | 104,359.0 [2/2] | 5.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| Claude Opus     | 189,523.5 [2/2] |    2,596.0 [2/2] |       0.0 [2/2] | 5.5 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| GLM (corrected) | 176,942.0 [2/2] |    8,572.0 [2/2] | 155,136.0 [2/2] | 6.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| DeepSeek        | 174,975.5 [2/2] |    2,808.5 [2/2] | 162,112.0 [2/2] | 5.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |

## Errors — recorded counts

| Model           | Harness | Model | Routing | Judge metrics |
| --------------- | -------: | -----: | -------: | -------------: |
| GPT             |       0 |     0 |       0 |             0 |
| Claude Opus     |       0 |     0 |       0 |             0 |
| GLM (corrected) |       0 |     0 |       0 |             0 |
| DeepSeek        |       0 |     0 |       0 |             0 |

## Per attempt

| Model           |                       Golden |      Attempt | Recorded result | Tool use | Completion | Fitness | Turns | Output tokens | Total s |
| --------------- | ----------------------------: | ------------: | ---------------: | --------: | ----------: | -------: | -----: | -------------: | -------: |
| GPT             | navigator-vague-requirements | 3b16719f4e5c |            Pass |     1.00 |       1.00 |    1.00 |     5 |          2495 |   79.37 |
| GPT             |         navigator-happy-path | 1816d8623b79 |            Pass |     0.95 |       0.95 |    0.90 |     5 |          2683 |   87.22 |
| Claude Opus     | navigator-vague-requirements | 6287b7c43f66 |            Pass |     0.95 |       0.90 |    0.90 |     5 |          6628 |  187.70 |
| Claude Opus     |         navigator-happy-path | c7b9f3ff3194 |            Pass |     0.80 |       0.90 |    0.90 |     5 |          3515 |  119.32 |
| GLM (corrected) | navigator-vague-requirements | fdf292edefce |            Pass |     0.95 |       0.90 |    0.80 |     5 |          8728 |  198.03 |
| GLM (corrected) |         navigator-happy-path | ad0ca6ae4e95 |            Pass |     0.80 |       0.95 |    1.00 |     5 |         12869 |  246.33 |
| DeepSeek        | navigator-vague-requirements | 9e932493be20 |            Pass |     0.95 |       0.95 |    0.90 |     5 |          5659 |  103.81 |
| DeepSeek        |         navigator-happy-path | c94da100902f |            Pass |     0.95 |       0.95 |    0.90 |     5 |          5090 |  119.94 |

## Interpretation and provenance

Both source cohorts ran sequentially; all 28 pairs of selected saved intervals have zero measured overlap, including scoring and teardown. The GLM pair ran later and used a corrected backend, so timing is descriptive. Provider load and cache warmth remain uncontrolled. Two samples per model do not establish reliability or a causal ranking.

GLM's passing pair retains the quality-threshold rounding defect (`min_quality=32` where 33 is required); vague fidelity is at its 0.80 cutoff. Its primary unconstrained fallback was not needed. See the [GLM validation](./glm-corrected-validation.md) for explanation and mock-isolation caveats. Passing scores do not mean flawless behavior; all original judge reasons are retained in the JSON export.

Sources: [earlier sequential cohort](./navigator-serial-sweep-results.json), [corrected GLM cohort](./glm-corrected-validation-results.json), and [this report's JSON export](./navigator-selected-model-comparison.json). Each selected original native artifact was hash-verified. JSON retains per-model provenance, profiles, metric coverage, complete original records, usage, and intervals.

All 223 pre-existing snapshotted evidence files and all 66 history records are unchanged. The strict comparison guard is unchanged. This report made no model or judge calls.

Reproduction uses the local offline driver `evals/results/report_navigator_selected_comparison_20261102.py`; it reads the two frozen cohorts and validates them independently. The source cohort hashes are recorded in JSON. The driver refuses to overwrite either report.

### Metric and measurement notes

- This is a descriptive comparison of explicitly selected historical runs from two configurations, not a fresh four-model sweep or controlled model ranking.

- GPT, Claude, and DeepSeek use the earlier skill/backend/checks; GLM uses the corrected hardware-consent instructions, local GPU filter, and added hardware-ordering check.

- Customer scripts, judge, metric versions, thresholds, and per-turn limits match. Scenario hashes differ because the new check versions grading intent.

- Hard-check passes and fidelity are the original results against each run's own contract. Earlier models were not regraded under the newer gate.

- The original eight-case roster remains 7/8 and the corrected GLM pair remains 2/2; the eight selected passing records do not create a new 8/8 sweep result.

- All means are per conversation with measured/selected coverage. Fitness is skill fidelity; no composite score or cost estimate is calculated.

- Output includes reasoning and tool-call tokens; reasoning is a subset of output. Cached input is a subset of input. Provider tokenizers/accounting differ.

- Agent time excludes judging; total time is agent plus scoring per conversation, not elapsed time for a new experiment.

- All 28 selected whole-case interval pairs have zero measured overlap. These runs occurred at different times, so provider load, cache warmth, and configuration changes still limit timing comparisons.

- There are only two samples per model. No model or judge was called, no attempt rerun or rescored, and no historical record changed.
