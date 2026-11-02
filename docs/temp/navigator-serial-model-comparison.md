## Navigator model comparison

Skill 6b84c869d2cc | source f0bd975 | judge enmaas/gpt-6.1-sol | rubric v2

8 conversations; 7/8 overall passes. Cutoffs: tool_use: 0.60 (conversation-v2); completion: 0.75 (conversation-v2); fitness: 0.80 (v2).

**Scheduling verified: zero overlap across all eight cases, including judging and teardown.** All 28 interval pairs were checked. [Timing evidence](./navigator-serial-sweep.md#scheduling-verification).

### Quality — means [measured/selected]

| Model           | N | Overall | Routing | Hard checks |    Tool use |  Completion | Fitness (fidelity) |
| --------------- | ---: | -------: | -------: | -----------: | -----------: | -----------: | ------------------: |
| enmaas-opus     | 2 |     2/2 |     2/2 |         2/2 | 0.875 [2/2] | 0.900 [2/2] |        0.900 [2/2] |
| enmaas-sol      | 2 |     2/2 |     2/2 |         2/2 | 0.975 [2/2] | 0.975 [2/2] |        0.950 [2/2] |
| enmaas-glm      | 2 |     1/2 |     2/2 |         1/2 | 0.800 [2/2] | 0.725 [2/2] |        0.850 [2/2] |
| enmaas-deepseek | 2 |     2/2 |     2/2 |         2/2 | 0.950 [2/2] | 0.950 [2/2] |        0.900 [2/2] |

### Efficiency — means per conversation [measured/selected]

| Model           |     Turns | Model responses |  Output tokens |      Agent s |   Scoring s |      Total s |
| --------------- | ---------: | ---------------: | --------------: | ------------: | -----------: | ------------: |
| enmaas-opus     | 5.0 [2/2] |       9.0 [2/2] |  5,071.5 [2/2] | 114.25 [2/2] | 39.26 [2/2] | 153.51 [2/2] |
| enmaas-sol      | 5.0 [2/2] |       9.0 [2/2] |  2,589.0 [2/2] |  50.44 [2/2] | 32.86 [2/2] |  83.30 [2/2] |
| enmaas-glm      | 5.0 [2/2] |       9.0 [2/2] | 15,300.5 [2/2] | 283.87 [2/2] | 38.86 [2/2] | 322.73 [2/2] |
| enmaas-deepseek | 5.0 [2/2] |       9.0 [2/2] |  5,374.5 [2/2] |  77.96 [2/2] | 33.92 [2/2] | 111.88 [2/2] |

### Usage and tools — means per conversation [measured/selected]

| Model           |    Input tokens | Reasoning tokens |    Cached input | MCP calls | MCP errors | Skill loads |
| --------------- | ---------------: | ----------------: | ---------------: | ---------: | ----------: | -----------: |
| enmaas-opus     | 189,523.5 [2/2] |    2,596.0 [2/2] |       0.0 [2/2] | 5.5 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| enmaas-sol      | 119,718.5 [2/2] |      910.5 [2/2] | 104,359.0 [2/2] | 5.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| enmaas-glm      | 181,763.5 [2/2] |   13,057.5 [2/2] | 152,832.0 [2/2] | 6.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| enmaas-deepseek | 174,975.5 [2/2] |    2,808.5 [2/2] | 162,112.0 [2/2] | 5.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |

### Errors — counts

| Model           | Harness | Model failures | Routing failures | Errored metrics |
| --------------- | -------: | --------------: | ----------------: | ---------------: |
| enmaas-opus     |       0 |              0 |                0 |               0 |
| enmaas-sol      |       0 |              0 |                0 |               0 |
| enmaas-glm      |       0 |              0 |                0 |               0 |
| enmaas-deepseek |       0 |              0 |                0 |               0 |

### Per attempt

| Model           |                       Golden |      Attempt | Overall | Tool use | Completion | Fitness | Turns | Output tokens | Total s |
| --------------- | ----------------------------: | ------------: | -------: | --------: | ----------: | -------: | -----: | -------------: | -------: |
| enmaas-deepseek |         navigator-happy-path | c94da100902f |    Pass |     0.95 |       0.95 |    0.90 |     5 |          5090 |  119.94 |
| enmaas-deepseek | navigator-vague-requirements | 9e932493be20 |    Pass |     0.95 |       0.95 |    0.90 |     5 |          5659 |  103.81 |
| enmaas-glm      |         navigator-happy-path | 6d6d24c66128 |    Pass |     0.80 |       0.95 |    0.90 |     5 |         12495 |  286.47 |
| enmaas-glm      | navigator-vague-requirements | 4f2a187d74cd |    Fail |     0.80 |       0.50 |    0.80 |     5 |         18106 |  358.98 |
| enmaas-opus     |         navigator-happy-path | c7b9f3ff3194 |    Pass |     0.80 |       0.90 |    0.90 |     5 |          3515 |  119.32 |
| enmaas-opus     | navigator-vague-requirements | 6287b7c43f66 |    Pass |     0.95 |       0.90 |    0.90 |     5 |          6628 |  187.70 |
| enmaas-sol      |         navigator-happy-path | 1816d8623b79 |    Pass |     0.95 |       0.95 |    0.90 |     5 |          2683 |   87.22 |
| enmaas-sol      | navigator-vague-requirements | 3b16719f4e5c |    Pass |     1.00 |       1.00 |    1.00 |     5 |          2495 |   79.37 |

### Failures and model settings

- enmaas-opus: {"agent_backend": "pydantic-ai", "api": "messages", "base_url": "https://api.enmaas.devshift.net", "extra_settings": {"anthropic_thinking": {"type": "adaptive"}}, "id": "enmaas-opus", "model_name": "claude-opus-4-6", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-sol: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-sol", "model_name": "gpt-5.6-sol", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-glm: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-glm", "model_name": "rits/zai-org/glm-5-3", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-deepseek: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-deepseek", "model_name": "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- 4f2a187d74cd: navigator_handoff_present: 
- 4f2a187d74cd: navigator_handoff_agrees_with_workload: no handoff emitted after the customer's selection
- 4f2a187d74cd: navigator_handoff_agrees_with_selection: chosen_category: expected 'balanced', got None; model_id: expected 'meta-llama/llama-3.1-8b-instruct', got None
- 4f2a187d74cd: Conversation MCP Task Completion: The score is 0.5 because the application clarified the workload, confirmed latency targets, presented recommendations, and acknowledged the user's balanced-model selection, but failed to complete the required navigator handoff. It omitted the mandatory handoff block containing capacity-preserving expected_qps and other deployment inputs, despite directing the user to paste 'the block above.'

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
- Scheduling verified for this cohort: all eight complete case intervals, including judging and teardown, were measured on one host monotonic clock. All 28 pairs were checked; there were zero case or batch overlaps. See navigator-serial-sweep-results.json for the intervals and verification evidence.
