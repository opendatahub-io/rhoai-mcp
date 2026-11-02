## Navigator model comparison

Skill 25392e419e13 | source e59d9e8 | judge enmaas/gpt-6.1-sol | rubric v2

8 conversations; 7/8 overall passes. Cutoffs: tool_use: 0.60 (conversation-v2); completion: 0.75 (conversation-v2); fitness: 0.80 (v2).

Timing is descriptive: overlapping batches prevent a controlled latency ranking.

### Quality — means [measured/selected]

| Model           | N | Overall | Routing | Hard checks |    Tool use |  Completion | Fitness (fidelity) |
| --------------- | ---: | -------: | -------: | -----------: | -----------: | -----------: | ------------------: |
| enmaas-opus     | 2 |     2/2 |     2/2 |         2/2 | 0.850 [2/2] | 0.925 [2/2] |        0.900 [2/2] |
| enmaas-sol      | 2 |     2/2 |     2/2 |         2/2 | 0.975 [2/2] | 0.925 [2/2] |        0.900 [2/2] |
| enmaas-glm      | 2 |     2/2 |     2/2 |         2/2 | 0.975 [2/2] | 0.900 [2/2] |        0.800 [2/2] |
| enmaas-deepseek | 2 |     1/2 |     2/2 |         1/2 | 0.900 [2/2] | 0.900 [2/2] |        0.900 [2/2] |

### Efficiency — means per conversation [measured/selected]

| Model           |     Turns | Model responses |  Output tokens |      Agent s |   Scoring s |      Total s |
| --------------- | ---------: | ---------------: | --------------: | ------------: | -----------: | ------------: |
| enmaas-opus     | 5.0 [2/2] |       9.0 [2/2] |  3,946.5 [2/2] |  94.28 [2/2] | 36.64 [2/2] | 130.91 [2/2] |
| enmaas-sol      | 5.0 [2/2] |       9.0 [2/2] |  2,684.5 [2/2] |  54.45 [2/2] | 37.39 [2/2] |  91.84 [2/2] |
| enmaas-glm      | 5.0 [2/2] |       9.0 [2/2] | 16,491.5 [2/2] | 282.37 [2/2] | 40.80 [2/2] | 323.16 [2/2] |
| enmaas-deepseek | 5.0 [2/2] |       9.0 [2/2] |  5,505.0 [2/2] |  44.98 [2/2] | 29.82 [2/2] |  74.80 [2/2] |

### Usage and tools — means per conversation [measured/selected]

| Model           |    Input tokens | Reasoning tokens |    Cached input | MCP calls | MCP errors | Skill loads |
| --------------- | ---------------: | ----------------: | ---------------: | ---------: | ----------: | -----------: |
| enmaas-opus     | 187,091.0 [2/2] |    1,627.5 [2/2] |       0.0 [2/2] | 5.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| enmaas-sol      | 119,636.5 [2/2] |      972.0 [2/2] | 104,235.5 [2/2] | 5.5 [2/2] |  0.5 [2/2] |   1.0 [2/2] |
| enmaas-glm      | 186,308.5 [2/2] |   14,192.5 [2/2] | 154,880.0 [2/2] | 6.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |
| enmaas-deepseek | 175,073.0 [2/2] |    2,953.0 [2/2] | 162,208.0 [2/2] | 5.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |

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
| enmaas-deepseek |         navigator-happy-path | fae31aebd41b |    Pass |     0.80 |       0.90 |    0.90 |     5 |          5797 |   80.11 |
| enmaas-deepseek | navigator-vague-requirements | 1dd60008ed20 |    Fail |     1.00 |       0.90 |    0.90 |     5 |          5213 |   69.49 |
| enmaas-glm      |         navigator-happy-path | 02539ef5343e |    Pass |     1.00 |       0.90 |    0.80 |     5 |         19274 |  372.89 |
| enmaas-glm      | navigator-vague-requirements | 1e271f5c8836 |    Pass |     0.95 |       0.90 |    0.80 |     5 |         13709 |  273.43 |
| enmaas-opus     |         navigator-happy-path | 9e84339148e4 |    Pass |     0.80 |       0.90 |    0.90 |     5 |          3440 |  119.74 |
| enmaas-opus     | navigator-vague-requirements | 6293513c587d |    Pass |     0.90 |       0.95 |    0.90 |     5 |          4453 |  142.09 |
| enmaas-sol      |         navigator-happy-path | e6939d8fabba |    Pass |     0.95 |       0.95 |    0.90 |     5 |          2408 |   94.16 |
| enmaas-sol      | navigator-vague-requirements | 2b4f92693ff5 |    Pass |     1.00 |       0.90 |    0.90 |     5 |          2961 |   89.52 |

### Failures and model settings

- enmaas-opus: {"agent_backend": "pydantic-ai", "api": "messages", "base_url": "https://api.enmaas.devshift.net", "extra_settings": {"anthropic_thinking": {"type": "adaptive"}}, "id": "enmaas-opus", "model_name": "claude-opus-4-6", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-sol: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-sol", "model_name": "gpt-5.6-sol", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-glm: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-glm", "model_name": "rits/zai-org/glm-5-3", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- enmaas-deepseek: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-deepseek", "model_name": "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- 1dd60008ed20: table_cells_complete: 'Cluster fit': empty cell(s) at column [2, 3, 4]

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
- Batches overlapped: enmaas-glm / enmaas-deepseek: 151.99s. Concurrent load prevents a controlled latency ranking.
