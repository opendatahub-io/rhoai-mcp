# Skill evaluation scoring

The skill suite uses an independent EnMaaS DeepSeek judge by default. Override `RHOAI_EVAL_EVAL_PROVIDER`, `RHOAI_EVAL_EVAL_MODEL`, `RHOAI_EVAL_EVAL_MODEL_BASE_URL`, or the threshold variables when needed. The agent and judge can use different models. The legacy LCS suite retains its own defaults and stock metrics.

| Required metric | Threshold | Evidence |
| --- | ---: | --- |
| Conversation MCP use | 0.60 | Tool choice and arguments across the conversation |
| Conversation task completion | 0.75 | Requested outcome and mandatory deliverables |
| Skill fidelity | 0.80 | Per-skill conversational G-Eval rubric |

MCP use takes the minimum of weighted tool-choice and argument scorecards. Each item is rated 0, 1, 2, or inapplicable with evidence; Python computes the weighted score and applies caps for serious defects. Completion likewise grades the whole conversation, with caps for missing outcomes or deliverables. Fidelity uses DeepEval's 0–10 grade normalized to 0–1. The raw score, threshold, reason, item evidence, and metric version are stored in JSONL.

An overall pass also requires a completed execution, successful skill routing, and all mandatory deterministic checks. A missing metric or judge error cannot pass. `CHECKS ONLY` identifies a run without judging. A check marked `unexercised` or `n/a` is not evidence that the branch passed.

GPU discovery counts only when it returns identifiable inventory before a filtered recommendation. Denied or incomplete discovery requires customer inventory; an unfiltered recommendation requires explicit customer consent. Execution and earlier scores are never rewritten during replay, so updated check contracts may produce different results for the same attempt. The current check version is `v2-gpu-discovery`.

The migration evidence and its fidelity limitation are preserved in `docs/temp/deepseek-item-judge-migration.md` at commit `f91a661`. Historical scores use their original checks and judge versions and should not be silently combined with new scores.

A local checks-only comparison against the preserved `f91a661` evaluator covered 15 saved attempts with matching scenario hashes: their deterministic check statuses were unchanged. Fifty-two artifacts had incompatible scenario hashes and twelve named scenarios no longer present; these were excluded. The offline discovery tests cover the newly accepted discovery path and the stricter unfiltered-consent rule. Missing required metric results now fail a judged replay even if every returned metric passed. Existing score records remain unchanged.
