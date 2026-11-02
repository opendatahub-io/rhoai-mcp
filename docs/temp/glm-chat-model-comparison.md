## Navigator model comparison

Skill 6b84c869d2cc | source 29266d6 | judge enmaas/gpt-6.1-sol | rubric v2

2 conversations; 1/2 overall passes. Cutoffs: tool_use: 0.60 (conversation-v2); completion: 0.75 (conversation-v2); fitness: 0.80 (v2).

### Quality — means [measured/selected]

| Model           | N | Overall | Routing | Hard checks |    Tool use |  Completion | Fitness (fidelity) |
| --------------- | ---: | -------: | -------: | -----------: | -----------: | -----------: | ------------------: |
| enmaas-glm-chat | 2 |     1/2 |     2/2 |         2/2 | 0.625 [2/2] | 0.925 [2/2] |        0.700 [2/2] |

### Efficiency — means per conversation [measured/selected]

| Model           |     Turns | Model responses |  Output tokens |      Agent s |   Scoring s |      Total s |
| --------------- | ---------: | ---------------: | --------------: | ------------: | -----------: | ------------: |
| enmaas-glm-chat | 5.0 [2/2] |      10.0 [2/2] | 20,291.0 [2/2] | 348.33 [2/2] | 39.00 [2/2] | 387.33 [2/2] |

### Usage and tools — means per conversation [measured/selected]

| Model           |    Input tokens | Reasoning tokens |    Cached input | MCP calls | MCP errors | Skill loads |
| --------------- | ---------------: | ----------------: | ---------------: | ---------: | ----------: | -----------: |
| enmaas-glm-chat | 230,185.0 [2/2] |   17,128.0 [2/2] | 191,488.0 [2/2] | 7.5 [2/2] |  0.0 [2/2] |   1.0 [2/2] |

### Errors — counts

| Model           | Harness | Model failures | Routing failures | Errored metrics |
| --------------- | -------: | --------------: | ----------------: | ---------------: |
| enmaas-glm-chat |       0 |              0 |                0 |               0 |

### Per attempt

| Model           |                       Golden |      Attempt | Overall | Tool use | Completion | Fitness | Turns | Output tokens | Total s |
| --------------- | ----------------------------: | ------------: | -------: | --------: | ----------: | -------: | -----: | -------------: | -------: |
| enmaas-glm-chat |         navigator-happy-path | 2aaa97922492 |    Fail |     0.30 |       0.90 |    0.60 |     5 |         28760 |  545.15 |
| enmaas-glm-chat | navigator-vague-requirements | e8d402ddd2ae |    Pass |     0.95 |       0.95 |    0.80 |     5 |         11822 |  229.52 |

### Failures and model settings

- enmaas-glm-chat: {"agent_backend": "pydantic-ai", "api": "chat", "base_url": "https://api.enmaas.devshift.net/v1", "extra_settings": {}, "id": "enmaas-glm-chat", "model_name": "rits/zai-org/glm-5-3", "sampling": "provider-default (temperature not requested; cannot be set uniformly across the matrix)", "temperature": null}
- 2aaa97922492: Conversation MCP Use: The score is 0.3 because the application failed to satisfy key hardware-selection requirements: it requested unfiltered recommendations before obtaining GPU inventory or explicit consent, then accepted H100 results from an A100-80-filtered call without validating the mismatch or making the required unconstrained fallback call. Although it used appropriate tools, generally supplied schema-valid arguments, reran recommendations after learning the GPU type, preserved the selected expected_qps of 0.13, and avoided deployment actions, these strengths did not outweigh the unresolved prerequisites and result-validation failures.
- 2aaa97922492: Navigator Skill Fidelity [Conversational GEval]: The navigator skill provides the specification. The assistant announced all three phases, obtained workload/SLO confirmation before recommendations, used readable prose and markdown tables, waited for a model choice, and ended with the /navigator-deploy handoff without planning or executing deployment. However, it violated the GPU-inventory gate: after asking which GPUs were available, it treated the customer's unanswered hardware question as permission to run unconstrained recommendations. The customer had neither said they did not know nor explicitly requested unfiltered results. Later A100-80 filtering did not undo that violation. The revised table also included an H100 fallback without a corresponding new unconstrained fallback call being shown.

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
- One replay per API and one full attempt per golden do not establish reliability or a causal API fix.
- Responses and Chat both passed the final-turn replay.
- The API-only Chat profile is not full Pi.dev parity; effort, cap, penalties, streaming and host differ from Pi.
- Do not pool these two cases into the existing eight-case roster or revise its 7/8 verdict.
- No production model spec or skill wording was changed.
