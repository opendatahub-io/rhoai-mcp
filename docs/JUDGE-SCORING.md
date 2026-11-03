# Judge configuration and score interpretation

The default judge is EnMaaS `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017`,
using `ENMAAS_KEY`. Normal scoring and `make eval-calibrate` default to this
model. There is no automatic fallback to a paid model. For normal scoring,
environment variables and `.env.eval` override defaults; replace any previous Sol setting:

```bash
export RHOAI_EVAL_EVAL_PROVIDER=enmaas
export RHOAI_EVAL_EVAL_MODEL=rits/deepseek-ai/DeepSeek-V4.1-Flash-1017
# Export ENMAAS_KEY using your existing credential setup.
make eval-skills-judge
```

Agent selection is independent of judge selection. The agent default remains
GPT; use `RHOAI_EVAL_AGENT_MODELS=enmaas-deepseek` to select DeepSeek for the
agent as well. EnMaaS GLM/DeepSeek is free for the account used in this branch's
experiments; billing for other accounts can differ. CI's judge defaults also
select EnMaaS DeepSeek and require the repository secret `ENMAAS_KEY`.

## MCP use and completion

The skill path uses **`conversation-v3-items`**. The metrics still subclass
`MultiTurnMCPUseMetric` and `MCPTaskCompletionMetric`, but replace holistic
numeric grading with explicit scorecards. Each item receives `0` (not met or
materially contradicted), `1` (partially met with an evidenced defect), or `2`
(fully met). A permitted `null` means genuinely inapplicable, with a reason;
an unfulfilled requirement is not inapplicable.

Each rating includes concrete evidence referring to the ordered conversation.
Python computes `sum(weight × rating) / sum(weight × 2)` over applicable items.
Weights express importance, not statistical precision; scores are not probabilities.

| Tool choice | Weight | Arguments | Weight | Completion | Weight |
| --- | ---: | --- | ---: | --- | ---: |
| Authorization | 3 | Workload grounding | 4 | Final requested outcome | 5 |
| Selection | 3 | Hardware grounding | 3 | Mandatory deliverables | 4 |
| Ordering | 2 | Target/configuration | 3 | Accuracy | 3 |
| Error handling | 2 | Schema validity | 2 | Confirmed requirements | 2 |
| Necessity | 1 | Secondary constraints | 1 | Transparency | 2 |
| Efficiency | 1 | Cross-call consistency | 1 | Clarity | 1 |

MCP use is the minimum of tool-choice and argument scores. Completion scores
the whole evolving conversation as one task. Code-enforced caps prevent
averaging away missing authorization (maximum 0.30), contradicted core arguments
(0.30), or incomplete final outcomes/missing mandatory deliverables (0.60).
Absent final prose cannot lower tool/argument ratings. A backend returning an
incorrect GPU despite a correct requested filter is not itself an argument error.
The judge can still misread evidence or assign the wrong item rating; these
caps do not replace the existing deterministic checks.

With all items applicable, denominators are 24, 28 and 34. A partially met
efficiency item produces tool choice `23/24 = 0.958333…`; a partially met clarity
item produces completion `33/34 = 0.970588…`. Raw fractions are retained.
V3 per-attempt and summary tables display four decimals so these fractions
remain visible; raw records retain full floats. Round scores remain valid when
supported by the items. No noise or forced
non-round digits are added to make scores appear precise.

Normal records retain the full cards under `metrics[].scorecards`: ratings,
evidence, weights, earned/possible points, uncapped score and cap. DeepEval's
reason payload also retains them for exports. Human-readable reasons are
assembled from item evidence without extra LLM summary calls. The standard
three-metric skill path uses four logical judge generations, down from six;
SDK retries can change actual requests. MCP score-cache reuse is disabled.

Keep the existing 0.60 / 0.75 / 0.80 cutoffs during validation. New judge/rubric
scores must not be pooled with historical Sol/`conversation-v2` cohorts. The
legacy LCS path still uses stock MCP metrics and needs separate calibration.

## Fidelity and G-Eval's default scale

**DeepEval's standard G-Eval grading scale is 0–10**, normalized to 0–1 by
division by 10. The official [G-Eval rubric documentation](https://deepeval.com/docs/metrics-llm-evals#rubric)
constrains rubric ranges to 0–10. In installed DeepEval 4.3.1, the
`conversational_g_eval/templates/generate_evaluation_results.txt` prompt requests
an integer, and `ConversationalGEval` divides that grade by 10.

Our OpenAI-compatible judge wrapper does not implement raw token-logprob
output, so DeepEval uses the plain-grade fallback rather than its optional
probability-weighted grading. A comment beside the fidelity factory in
`evals/skills/rubrics.py` records this behavior and the documentation link.

Fidelity stays on this scale, with evidence version `v2`. Changing a holistic
scale to 0–30 would offer more grades without defining what distinguishes
26 from 27. It would require custom scoring/normalization and fresh validation.
An average of two fidelity grades can have 0.05 steps: `(0.90 + 1.00) / 2 = 0.95`.
That is a cohort mean, distinct from a single 0–10 judge grade.

DeepSeek has a known fidelity limitation at the existing 0.80 cutoff: it passed
a missing mandatory Navigator handoff in four of five validation repetitions.
The new completion metric rejected that case in all five. Keep the deterministic
handoff check and all required metrics; fidelity alone is insufficient evidence
of a complete handoff. See the [migration validation](./temp/deepseek-item-judge-migration.md).

The earlier MCP 0.05 clustering meant the judge itself emitted familiar round
grades such as 0.90 and 0.95 in its JSON. Code did not round them to a 0.05 grid;
saved Sol results also contain 0.98. Item-based MCP scoring now gives finer
fractions an explicit derivation, while retaining the judge's uncertainty about
whether each rubric item was actually met.
