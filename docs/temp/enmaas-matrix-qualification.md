# Four-model EnMaaS sweep: roster and Anthropic qualification

**Date:** 2026-11-02 · **Decision:** GPT + Anthropic + GLM + DeepSeek, all via EnMaaS.

The runtime default and matrix now match the requested roster. The full sweep is
**2 skills × 2 goldens × 4 models = 16 conversations**, run one model at a time
for review. This step qualified the Anthropic route and adapter; it did not run
the full sweep or any new judge evaluations.

| Spec | Model | Explicit API | Provider base URL |
|---|---|---|---|
| `enmaas-sol` | `gpt-5.6-sol` | OpenAI Responses | `https://api.enmaas.devshift.net/v1` |
| `enmaas-opus` | `claude-opus-4-6` | Anthropic Messages | `https://api.enmaas.devshift.net` |
| `enmaas-glm` | `rits/zai-org/glm-5-3` | OpenAI Responses | `https://api.enmaas.devshift.net/v1` |
| `enmaas-deepseek` | `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017` | OpenAI Responses | `https://api.enmaas.devshift.net/v1` |

All entries use `ENMAAS_KEY`. The one-model default is `enmaas-sol`. Historical
direct-provider specs remain explicit diagnostics outside the default and matrix.
The judge stays **EnMaaS `gpt-6.1-sol`**, with unchanged 0.60 / 0.75 / 0.80
cutoffs, MCP `conversation-v2`, and fidelity `v2`.

## Anthropic requires its native API

The EnMaaS `/v1/models` catalog returned 23 IDs and omitted Anthropic. Probing
`claude-opus-4-6` on Responses and Chat returned HTTP 404; a native
`POST /v1/messages` request succeeded with HTTP 200. These were API-route
differences, not missing Opus access. Earlier Sonnet checks on the incompatible
routes also returned 404 and are retained in the evidence.

The framework explicitly builds `AnthropicModel` with `AnthropicProvider`, using
the EnMaaS root URL and `ENMAAS_KEY`. The SDK adds `/v1/messages` itself; including
`/v1` in its base URL would duplicate the path. GPT/GLM/DeepSeek keep
`OpenAIResponsesModel` and their `/v1` base URLs. This follows Pydantic AI's
[custom Anthropic provider configuration](https://pydantic.dev/docs/ai/models/anthropic/).
The skill-eval dependency group now includes `pydantic-ai-slim[anthropic]`.

## Probe results and corrections

The shared three-turn probe now supports native Messages with tool-use/result
pairing. It checks nested arguments, enum/array shapes, grounding in tool returns,
and stopping to answer from the returned recommendations.

| Candidate | Probe version | Checks | Elapsed | Verdict |
|---|---|---:|---:|---|
| EnMaaS `gpt-5.6-sol`, Responses | `v2` | 11/11 | 9.37 s | Qualified |
| EnMaaS `claude-opus-4-6`, Messages | `v2` | 11/11 | 20.85 s | Qualified |

Both rejected `temperature=0`; the probe records its fallback to provider
defaults. Runtime matrix specs leave temperature unset from the start. GLM and
DeepSeek retain their [historical Phase 0 qualification](./phase0-model-probe.md).
They were not re-probed in this step.

Two probe defects surfaced and were corrected explicitly:

- The SDK adds nullable response fields, including `toolset_name`. Echoing those
  nulls as input made the gateway reject the second request. Messages replay now
  omits null SDK-only fields. The initial failed attempt is preserved.
- The original final-answer check required a literal `Llama-3.1-8B` token even
  though the fixture also returns Mistral. One Opus attempt scored 10/11 under
  that check; its truncated saved prose cannot establish a corrected verdict.
  Probe `v2` accepts either returned model with readable name separators, records
  full final prose, and rejects unrelated models. Fresh GPT/Opus attempts above
  used `v2`; earlier outcomes were not rewritten. This changes the Phase 0 probe,
  not the skill goldens, hard checks, judge metrics, or calibrated cutoffs.

## Pydantic AI + stdio adapter spike

Attempt `6bbc867e93c6`, run `phase0-enmaas-opus-20261102`, completed all five turns
in **100.06 seconds** through the explicit native Messages model/provider.
All **18 connectivity/adapter checks passed**: 53 MCP schemas loaded, the
14,467-character navigator skill body was delivered, recommendation calls were
attributed correctly, typed DeepEval objects were built, and the transcript,
schemas, skill body, and scoring case survived artifact save/reload unchanged.

This is a smoke test, not a judged skill golden or a reliability estimate. The
spike also records five planner HTTP 400 results, followed by successful
recommendations, and `explore_cluster` outside the skill's declared allowlist.
The bundled planner's optional real node probes returned 401 despite mock MCP
state. These observations remain visible for the four goldens to assess; no
behavior or skill prompt was tuned here. No deployment ran.

## Validation and next step

All **280 offline evaluation tests passed**. Tests exercise the actual native
Messages request path, EnMaaS credential selection, paired tool results, omission
of null SDK fields, matrix/default coverage, and the corrected answer check.
Targeted lint, formatting, and type checks passed; the lockfile resolved offline.
Pytest collection confirms 16 full-matrix cases without executing them.

[Machine-readable evidence](./enmaas-matrix-probe-results.json) retains the catalog,
route failures, earlier probe failures, successful versioned probes, adapter
checks and observations, package versions, and source/artifact hashes. Native
artifacts and individual raw probe files remain under local `evals/results/`.
Existing calibration, baseline, and result history were not rewritten.

The next macro step is four judged goldens for one matrix entry, keeping the
judge and settings fixed. For example, with `ENMAAS_KEY` available:

```sh
RHOAI_EVAL_AGENT_MODELS=enmaas-opus \
RHOAI_EVAL_SKILL_JUDGE=1 \
RHOAI_EVAL_EVAL_PROVIDER=enmaas \
RHOAI_EVAL_EVAL_MODEL=gpt-6.1-sol \
RHOAI_EVAL_EVAL_MODEL_BASE_URL=https://api.enmaas.devshift.net/v1 \
RHOAI_EVAL_MCP_USE_THRESHOLD=0.6 \
RHOAI_EVAL_TASK_COMPLETION_THRESHOLD=0.75 \
RHOAI_EVAL_CLUSTER_MODE=mock \
RHOAI_EVAL_OTLP_ENDPOINT= \
DEEPEVAL_TELEMETRY_OPT_OUT=YES \
KUBECONFIG=/dev/null \
caffeinate -i uv run --group skill-eval pytest evals/skills -v -s --reruns 0
```

Use one explicit spec per reviewed batch; `matrix` selects all four. Each attempt
keeps its own artifact and scoring identity, including failures. Do not pool this
qualification spike with judged golden records or interpret a partial family
set as the complete 16-conversation sweep.
