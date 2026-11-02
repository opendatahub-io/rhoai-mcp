## Navigator model comparison

Skill 96cc8596d4dd | source 44d6774 | judge enmaas/gpt-6.1-sol | rubric v2

2 conversations; 2/2 overall passes. Cutoffs: tool_use: 0.60 (conversation-v2); completion: 0.75 (conversation-v2); fitness: 0.80 (v2).

### Quality — means [measured/selected]

| Model      | N | Overall | Routing | Hard checks |    Tool use |  Completion | Fitness (fidelity) |
| ---------- | ---: | -------: | -------: | -----------: | -----------: | -----------: | ------------------: |
| enmaas-glm | 2 |     2/2 |     2/2 |         2/2 | 0.875 [2/2] | 0.925 [2/2] |        0.900 [2/2] |

### Efficiency — means per conversation [measured/selected]

| Model      |     Turns | Model responses |  Output tokens |      Agent s |   Scoring s |      Total s |
| ---------- | ---------: | ---------------: | --------------: | ------------: | -----------: | ------------: |
| enmaas-glm | 5.0 [2/2] |       9.0 [2/2] | 10,798.5 [2/2] | 182.53 [2/2] | 39.64 [2/2] | 222.18 [2/2] |

### Usage and tools — means per conversation [measured/selected]

| Model      |    Input tokens | Reasoning tokens |    Cached input | MCP calls | MCP errors | Skill loads |
| ---------- | ---------------: | ----------------: | ---------------: | ---------: | ----------: | -----------: |
| enmaas-glm | 176,942.0 [2/2] |    8,572.0 [2/2] | 155,136.0 [2/2] | 6.0 [2/2] |  0.0 [2/2] |   1.0 [2/2] |

### Errors — counts

| Model      | Harness | Model failures | Routing failures | Errored metrics |
| ---------- | -------: | --------------: | ----------------: | ---------------: |
| enmaas-glm |       0 |              0 |                0 |               0 |

### Per attempt

| Model      |                       Golden |      Attempt | Overall | Tool use | Completion | Fitness | Turns | Output tokens | Total s |
| ---------- | ----------------------------: | ------------: | -------: | --------: | ----------: | -------: | -----: | -------------: | -------: |
| enmaas-glm |         navigator-happy-path | ad0ca6ae4e95 |    Pass |     0.80 |       0.95 |    1.00 |     5 |         12869 |  246.33 |
| enmaas-glm | navigator-vague-requirements | fdf292edefce |    Pass |     0.95 |       0.90 |    0.80 |     5 |          8728 |  198.03 |

### Failures and model settings

- enmaas-glm: {"agent_backend": "pydantic-ai", "api": "responses", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-glm", "model_name": "rits/zai-org/glm-5-3", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}

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
- One attempt per golden is a bounded validation, not a reliability estimate.
- This uses the production Responses profile, not full Pi.dev Chat Completions parity.
- Skill wording, local GPU filtering, and grading intent differ from the historical roster; do not pool these cases into its comparison or revise its 7/8 verdict.
- Model API, effort, output cap, penalties, host instruction, judge, thresholds, and runtime limits remain unchanged.
- A fresh two-case pass would not establish that intermittent missing handoffs cannot recur.
