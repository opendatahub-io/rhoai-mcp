# Navigator comparison with DeepSeek judge

2026-11-03 · 8 fresh conversations · 7/8 overall passes. Two scenarios per model, fully sequential, with no conversation reruns. These are smoke results, not a reliability ranking.

Judge coverage: 21/24 valid scores; 2 errored metric slots and 1 missing metric slot. Infrastructure failures remain unscored and retain their original overall verdicts.

GLM's happy-path conversation passed the hard checks but hit the 180-second MCP judge timeout; its fidelity metric was not produced. GLM's quality means therefore cover only its vague-requirements case.

[Sweep configuration, findings and provenance](./navigator-deepseek-judge-sweep.md) · [Original cohort evidence](./navigator-deepseek-judge-sweep-results.json) · [Comparison JSON with complete scorecards](./navigator-deepseek-judge-model-comparison.json)

## Navigator model comparison

Skill 88cbf1008ce0 | source 7c706ea | judge enmaas/rits/deepseek-ai/DeepSeek-V4.1-Flash-1017 | rubric v2

8 conversations; 7/8 overall passes. Cutoffs: tool_use: 0.60 (conversation-v3-items); completion: 0.75 (conversation-v3-items); fitness: 0.80 (v2).

### Quality — means [measured/selected]

| Model           | N | Overall | Routing | Hard checks |    Tool use |  Completion | Fitness (fidelity) |
| --------------- | ---: | -------: | -------: | -----------: | -----------: | -----------: | ------------------: |
| enmaas-sol      | 2 |     2/2 |     2/2 |         2/2 | 1.0000 [2/2] | 0.9265 [2/2] |        1.0000 [2/2] |
| enmaas-opus     | 2 |     2/2 |     2/2 |         2/2 | 0.9594 [2/2] | 0.9265 [2/2] |        1.0000 [2/2] |
| enmaas-glm      | 2 |     1/2 |     2/2 |         2/2 | 1.0000 [1/2] | 1.0000 [1/2] |        1.0000 [1/2] |
| enmaas-deepseek | 2 |     2/2 |     2/2 |         2/2 | 1.0000 [2/2] | 1.0000 [2/2] |        1.0000 [2/2] |

### Efficiency — means per conversation [measured/selected]

| Model           |     Turns | Model responses |  Output tokens |      Agent s |    Scoring s |      Total s |
| --------------- | ---------: | ---------------: | --------------: | ------------: | ------------: | ------------: |
| enmaas-sol      | 5.0 [2/2] |       9.0 [2/2] |  2,586.5 [2/2] |  48.56 [2/2] | 108.15 [2/2] | 156.71 [2/2] |
| enmaas-opus     | 5.0 [2/2] |       9.0 [2/2] |  3,776.0 [2/2] |  90.66 [2/2] | 185.55 [2/2] | 276.20 [2/2] |
| enmaas-glm      | 5.0 [2/2] |       9.0 [2/2] | 19,601.0 [2/2] | 301.33 [2/2] | 149.91 [2/2] | 451.24 [2/2] |
| enmaas-deepseek | 5.0 [2/2] |       9.0 [2/2] |  4,339.5 [2/2] |  42.30 [2/2] | 172.78 [2/2] | 215.08 [2/2] |

### Usage and tools — means per conversation [measured/selected]

| Model           |    Input tokens | Reasoning tokens |    Cached input | MCP calls | MCP errors | Skill loads |
| --------------- | ---------------: | ----------------: | ---------------: | ---------: | ----------: | -----------: |
| enmaas-sol      | 120,581.5 [2/2] |      803.0 [2/2] | 104,949.0 [2/2] | 5.5 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| enmaas-opus     | 187,541.0 [2/2] |    1,353.5 [2/2] |       0.0 [2/2] | 5.0 [2/2] |  0.5 [2/2] |   1.0 [2/2] |
| enmaas-glm      | 186,821.5 [2/2] |   17,233.5 [2/2] | 150,016.0 [2/2] | 6.0 [2/2] |  0.5 [2/2] |   1.0 [2/2] |
| enmaas-deepseek | 173,511.5 [2/2] |    1,983.5 [2/2] | 160,576.0 [2/2] | 5.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |

### Errors — counts

| Model           | Harness | Model failures | Routing failures | Errored metrics |
| --------------- | -------: | --------------: | ----------------: | ---------------: |
| enmaas-sol      |       0 |              0 |                0 |               0 |
| enmaas-opus     |       0 |              0 |                0 |               0 |
| enmaas-glm      |       0 |              0 |                0 |               2 |
| enmaas-deepseek |       0 |              0 |                0 |               0 |

### Per attempt

| Model           |                       Golden |      Attempt | Overall | Tool use | Completion | Fitness | Turns | Output tokens | Total s |
| --------------- | ----------------------------: | ------------: | -------: | --------: | ----------: | -------: | -----: | -------------: | -------: |
| enmaas-deepseek |         navigator-happy-path | f99c32e8c061 |    Pass |   1.0000 |     1.0000 |    1.00 |     5 |          4241 |  221.65 |
| enmaas-deepseek | navigator-vague-requirements | 6f95514b2138 |    Pass |   1.0000 |     1.0000 |    1.00 |     5 |          4438 |  208.52 |
| enmaas-glm      |         navigator-happy-path | 54a6fd8de32a |    Fail |      n/a |        n/a |     n/a |     5 |         21271 |  513.74 |
| enmaas-glm      | navigator-vague-requirements | 80afe84f338c |    Pass |   1.0000 |     1.0000 |    1.00 |     5 |         17931 |  388.73 |
| enmaas-opus     |         navigator-happy-path | 67bb2808c0e1 |    Pass |   0.9545 |     0.8529 |    1.00 |     5 |          3416 |  277.89 |
| enmaas-opus     | navigator-vague-requirements | cc0fc7809fa6 |    Pass |   0.9643 |     1.0000 |    1.00 |     5 |          4136 |  274.51 |
| enmaas-sol      |         navigator-happy-path | 195d98d96393 |    Pass |   1.0000 |     1.0000 |    1.00 |     5 |          2482 |  137.55 |
| enmaas-sol      | navigator-vague-requirements | e97b7cb3ce1b |    Pass |   1.0000 |     0.8529 |    1.00 |     5 |          2691 |  175.86 |

### Failures and model settings

- enmaas-sol: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-sol", "model_name": "gpt-5.6-sol", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-opus: {"agent_backend": "pydantic-ai", "api": "messages", "base_url": "https://api.enmaas.devshift.net", "extra_settings": {"anthropic_thinking": {"type": "adaptive"}}, "id": "enmaas-opus", "model_name": "claude-opus-4-6", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-glm: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-glm", "model_name": "rits/zai-org/glm-5-3", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-deepseek: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-deepseek", "model_name": "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- 54a6fd8de32a: Conversation MCP Use: TimeoutError: call timed out after 180s (per attempt). Increase DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE (None disables) or reduce work per attempt.
- 54a6fd8de32a: Conversation MCP Task Completion: TimeoutError: call timed out after 180s (per attempt). Increase DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE (None disables) or reduce work per attempt.

Excluded records: {"dry_run": 0, "duplicate": 0, "last_limit": 0, "outside_selection": 0, "rescore": 0}

### Notes


- Fitness means the existing skill-fidelity score; no composite ranking is calculated.
- Hard checks pass when all applicable measured checks pass. Conditional n/a or unexercised checks do not establish branch coverage.
- All averages are per selected conversation. Each value shows measured/selected coverage; missing or errored scores stay n/a, never zero. Partial coverage means are not directly comparable to full coverage.
- Turns are completed scripted user turns; model responses count saved responses, including tool-use steps.
- Token counts sum the retained per-response agent usage across the conversation, including repeated context. Output includes reasoning and tool-call tokens; reasoning is a subset, not an extra charge to add. Cached input is a subset of input. Failed requests without saved usage cannot be counted.
- Tokenizers and provider accounting differ. These are recorded token counts, not billed EnMaaS cost. SDK price estimates are not reported as billing.
- Agent time excludes judging. Total time = agent + scoring per conversation; it is not cohort elapsed time. Timing includes provider latency, mock MCP work, and startup.
- One attempt per cell is a smoke comparison, not a reliability estimate. Failures remain in quality and efficiency denominators.
- This is a fresh cohort using the corrected Navigator skill/backend and DeepSeek conversation-v3-items judge. Historical Sol scores are separate evidence; differences cannot be attributed solely to judge selection.
- Scheduling verified across 8 whole-case intervals including judging and teardown: 28 interval pairs, 0 case overlaps, 0 batch overlaps.
- The known DeepSeek fidelity false-pass limitation remains relevant. Deterministic handoff checks and completion are required alongside fidelity; a passing score is not evidence of flawless behavior.
- The following non-perfect passing scores are judge observations, not independent human labels. Full item ratings and evidence remain in the JSON export.
- Passing-score caveat — enmaas-sol / navigator-vague-requirements (e97b7cb3ce1b), Conversation MCP Task Completion 0.8529: completion.accuracy: 1/2 — [10] shows the Performance duplicate re-run with ttft_max_ms=44 returned 'No configurations matched'; [11] nevertheless lists Mistral Small 24B in the Performance column as 'Available ✓' instead of '—', and its prose omits that Performance re-run failure. Other claims, such as the balanced handoff capacity, are supported.; completion.transparency: 1/2 — [7] honestly reports unavailable GPU inventory and asks for GPU types; [10] explains duplicate tightening and says missing matches will be shown as unavailable; however [11] hides the Performance re-run's no-match result by displaying the original Mistral recommendation as available.
- Passing-score caveat — enmaas-opus / navigator-vague-requirements (cc0fc7809fa6), Conversation MCP Use 0.9643: arguments.constraints: 1/2 — [10] Cost rerun used max_cost_per_month=1788, but the tighter constraint requested by the skill would be 2555 * 0.7 = 1788.5; this is a minor rounding/tightening deviation. The [10] min_quality=33 correctly rounds ceil(27.1 + 5).
- Passing-score caveat — enmaas-opus / navigator-happy-path (67bb2808c0e1), Conversation MCP Use 0.9545: arguments.constraints: 1/2 — [10] Cost duplicate used max_cost_per_month=1788, but 2555.0 * 0.7 = 1788.5, so it rounded down/stricter than the skill's specified threshold; [10] Quality duplicate used min_quality=33, correctly ceil(27.1+5). Units and types were otherwise valid.
- Passing-score caveat — enmaas-opus / navigator-happy-path (67bb2808c0e1), Conversation MCP Task Completion 0.8529: completion.accuracy: 1/2 — Most table values match [9], including Llama 3.1 8B at 76ms TTFT/$2,555 and Mistral Small 24B at 63ms/$10,220. However [10] cost re-run failed with 'tuple index out of range', yet [11] states no cheaper configuration was found under ~$1,789/month; that conclusion is unsupported by the available result.; completion.transparency: 1/2 — [7] honestly discloses that cluster GPU inventory could not be read and asks the user; [13] does not claim deployment. But [10]'s cost-query failure is not disclosed, and [11] presents it as a valid no-cheaper-result finding.
- Incomplete judging — enmaas-glm / navigator-happy-path (54a6fd8de32a): Conversation MCP Use: TimeoutError: call timed out after 180s (per attempt). Increase DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE (None disables) or reduce work per attempt.; Conversation MCP Task Completion: TimeoutError: call timed out after 180s (per attempt). Increase DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE (None disables) or reduce work per attempt.; Missing metric: Navigator Skill Fidelity [Conversational GEval]. This retains the original overall failure, but is not evidence of an agent-quality failure. Partial-coverage means must not be treated as directly comparable to fully scored model pairs.
