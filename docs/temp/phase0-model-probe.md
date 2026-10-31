# Phase 0: model probe and spike results

**Date:** 2026-10-31 · **Plan:** [`pydantic-ai-skill-evals-plan.md`](./pydantic-ai-skill-evals-plan.md) §4 Phase 0, §6 Q1

Outcome of the Phase 0 tool-calling probe and the connectivity/adapter spike.
This file sets the model roster in `evals/agent/models.py`.

---

## 1. Model roster

Reproduce with:

```bash
uv run --group skill-eval python -m evals.agent.probe_tool_calling --json /tmp/probe.json
```

The probe drives three turns of OpenAI-style tool calling with a non-trivial
argument shape (nested object, enum, enum array), requires the second call to be
grounded in a value returned by the first, and requires the model to stop calling
tools and answer in prose. All 11 checks must pass.

| Candidate | Model | Endpoint | API | Verdict | Probe time |
|---|---|---|---|---|---|
| `openai-sol` | `gpt-5.6-sol` | OpenAI | Responses | **qualified** | 8.9 s |
| `openai-sol-chat` | `gpt-5.6-sol` | OpenAI | Chat + `reasoning_effort=none` | **qualified** | 5.3 s |
| `openai-luna` | `gpt-5.6-luna` | OpenAI | Responses | **qualified** | 7.9 s |
| `enmaas-sol` | `gpt-5.6-sol` | EnMaaS | Responses | **qualified** | 8.4 s |
| `enmaas-sol-chat` | `gpt-5.6-sol` | EnMaaS | Chat + `reasoning_effort=none` | **qualified** | 6.3 s |
| `enmaas-glm` | `rits/zai-org/glm-5-3` | EnMaaS | Chat | **qualified** | 11.3 s |
| `enmaas-deepseek` | `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017` | EnMaaS | Chat | **qualified** | 25.9 s |

Nothing was excluded. The plan's §6 Q1 worry — that a hosted build might not
tool-call reliably enough to enter the matrix — did not materialise.

### 1.0 The API choice is not ours to make

An earlier draft of this document presented Responses vs Chat Completions as an
open decision. It mostly is not. Three independent facts converge on Responses:

- **Pydantic AI already defaults to it.** `infer_model('openai:<anything>')`
  returns `OpenAIResponsesModel` — blanket, for every model name, including
  `gpt-4o` and `gpt-3.5-turbo`. Chat Completions requires explicitly asking for
  it via the `openai-chat:` prefix or `OpenAIChatModel`.
- **OpenAI directs this family there.** `gpt-5.6-*` rejects function tools on
  `/v1/chat/completions` outright (§1.1).
- **EnMaaS serves Responses for every roster model**, including the open-weight
  ones — verified: both return `['reasoning', 'function_call']` output items.

So every matrix cell runs on Responses, matching the library default. Chat
Completions is not a peer alternative; for `gpt-5.6` it is only reachable by
switching reasoning off, which is a downgrade on a phase-gated task. The
`openai-sol-chat` spec is retained purely as a diagnostic for isolating a
suspected Responses-API problem, and is deliberately outside `MATRIX_SPEC_IDS`.

### 1.1 The gpt-5.6 family needs the Responses API

The first probe run failed all three gpt-5.6 candidates identically:

```
400 - Function tools with reasoning_effort are not supported for gpt-5.6-sol in
/v1/chat/completions. To use function tools, use /v1/responses or set
reasoning_effort to 'none'.
```

So this family offers a choice, not a default:

- **Responses API, reasoning on** — what OpenAI directs tool use to. Rejects an
  explicit `temperature`, so `temperature=0` is unavailable and run-to-run
  variance is higher.
- **Chat Completions, `reasoning_effort='none'`** — accepts `temperature=0` and
  is cheaper and faster, but gives up reasoning on a phase-gated multi-step task.

`openai-sol` (Responses, reasoning on) is the matrix default, matching both the
Pydantic AI default and OpenAI's own direction. The Chat variant is reachable for
a diagnostic run, but it is not a matrix cell:

```bash
RHOAI_EVAL_AGENT_MODELS=openai-sol-chat
```

### 1.2 EnMaaS is a clean OpenAI-compatible endpoint

`https://api.enmaas.devshift.net/v1`, credential in `ENMAAS_KEY`. It serves 23
models across four owners (`openai`, `curvebender`, `vertex`, plus its own). It
proxies the same `gpt-5.6-sol` with identical behaviour to the direct endpoint,
including the same Responses-API requirement — so `enmaas-sol` is kept as a
control cell: a failure there that does not reproduce on `openai-sol` is a
gateway problem, not a model problem.

The plan's §2.3 placeholders resolve to `rits/zai-org/glm-5-3` and
`rits/deepseek-ai/DeepSeek-V4.1-Flash-1017`.

### 1.3 Temperature: not set anywhere, deliberately

**`temperature=0` cannot be set uniformly across this matrix.** The gpt-5.6
family rejects the parameter outright:

```
400 - Unsupported parameter: 'temperature' is not supported with this model.
```

while `glm-5-3` and `DeepSeek-V4.1-Flash` accept it on Responses (both verified).

Plan §2.3 says to "set `temperature=0` where supported". Followed literally that
pins two matrix cells and leaves the third free, so part of any observed
difference between `openai-sol` and `enmaas-glm` would be a sampling-config
artifact rather than a property of the models. **No matrix cell requests a
temperature**; all three run on provider defaults and are therefore treated
alike. This is a deliberate deviation from the literal §2.3 wording, in service
of its actual intent.

| Spec | API | `temperature` requested |
|---|---|---|
| `openai-sol`, `openai-luna`, `enmaas-sol` | Responses | none — parameter rejected by the model |
| `enmaas-glm`, `enmaas-deepseek` | Responses | none — accepted, but left unset for parity |
| `openai-sol-chat` *(diagnostic, non-matrix)* | Chat | `0.0` |

Two caveats to carry forward:

- This buys uniform **requests**, not uniform sampling. Provider defaults need
  not match each other, so the cells are not actually sampling alike.
- **No configuration makes this matrix deterministic.** Re-running a scenario
  will not reproduce a transcript. That is a standing reason to treat one sweep
  as a smoke comparison rather than a ranking, and to use explicit repeated
  attempts for any stronger claim (plan §2.3).

Each result record carries this explicitly via `AgentModelSpec.describe()`,
which emits a `sampling` string alongside the raw value so a bare
`"temperature": null` is not mistaken for determinism that was attempted and
failed.

---

## 2. Spike acceptance

```bash
uv run --group skill-eval python -m evals.agent.spike --model openai-sol
```

All of Phase 0's acceptance criteria pass, on **all four** matrix-relevant models
(`openai-sol`, `enmaas-sol`, `enmaas-glm`, `enmaas-deepseek`): stdio connection,
skill discovery and loading, `recommend_model` reached, a second turn in the same
agent context, typed DeepEval objects with correct turn attribution, and an
artifact that reloads into an identical transcript.

Routing worked first try on every model: each discovered `navigator` from the
catalog without being told. §6 Q4's worry about the confusable skill-description
pair has not shown up yet, though one run per model is not evidence of stability.

---

## 3. Corrections to the plan

Three plan assumptions turned out to be wrong. All three were silent failures —
they produce plausible-looking output rather than an error.

### 3.1 §3.6: the skill body is *not* the `load_capability` result

The plan says to capture "the `load_capability` result containing the skill body
actually delivered". In `pydantic-ai-harness` 0.64 that result is only an
acknowledgement:

```python
{'message': "Capability 'navigator' loaded. Its instructions, if any, follow."}
```

The real 14,467-character body arrives as a separate `InstructionDeltaPart`
stamped `InstructionId(source=CapabilityInstructionSource(id='navigator'))`, and
the routing catalog arrives as an `InstructionBaselinePart` under
`capability:deferred_capabilities`.

A fidelity rubric pointed at the tool result would have been shown one sentence
and asked to judge adherence to a document it never saw. `AgentTranscript`
captures both instruction streams instead; `skill_body()` reads the delta.

### 3.2 §1.3: harness tools cannot be told apart by name alone

`LoadCapabilityCallPart` is a **subclass** of `ToolCallPart` and reports the same
`part_kind='tool-call'`. An `isinstance(part, ToolCallPart)` check therefore
counts skill loading as an MCP interaction, inflating MCP-use scores. The adapter
checks the subclass first.

### 3.3 `load_capability` arguments are a JSON string keyed by `id`

Not a dict, and not keyed by `name`: `'{"id":"navigator"}'`. Routing detection
that looked for a `name` key would have reported "no skill loaded" on every
successful run.

### 3.4 Confirmed as written

- Importing the DeepEval scoring path loads **zero** `pydantic_ai` modules.
- `Turn._mcp_interaction` is a derived read-only property; the
  `deepeval_helpers.py` workaround raised on assignment and has been deleted.
- §1.5's blocker was real and reproduced live: see §4.

---

## 4. §1.5 blocker confirmed, and deliberately left in place

A default server exposes **53** tools, and `get_cluster_resources` is the single
entry across both skills' `allowed-tools` that does not resolve. It is listed in
the committed tool surface of both RHAIRFE-1705 and RHAIRFE-1706
(`docs/mcp-api-stability.md`), so a default server does not serve its own
published contract.

The effect reproduced end-to-end. The agent followed the skill
correctly all the way to Phase 3 and then, lacking the tool, fell back to asking
the customer to type in their own hardware:

> Before querying the planner, I need the available cluster hardware. Which GPU
> types does your OpenShift AI cluster have — for example, **H100, H200,
> A100-80, A100-40, L4, or B200**?

### Decision: evaluate without changing the server

Plan §1.5 proposed fixing this first (option B). **That is deferred.** The server
is unchanged, the default surface stays at **53** tools, and the evals run
against the configuration customers actually get. Rationale: exposing a
contractually-committed tool is a product decision with its own review, and it
should not ride along inside eval scaffolding.

A complete, tested implementation of option B is preserved at
[`get-cluster-resources-fix.patch`](./get-cluster-resources-fix.patch) — shared
handler, `cluster-composites` ownership with a `training` fallback, permission
carry-over, a dedup fix in `PluginManager.collect_tool_permissions()`, and 14
tests. It applied cleanly and took the surface to 54 tools with the agent
calling the tool itself. Apply with `git apply`.

Two existing tests need adjusting when that patch lands, because the shared-tool
design intentionally changes what they assert:
`tests/integration/test_tool_registration.py` builds a server without attaching
its `PluginManager` (so `server.plugins` is empty while every plugin registers
tools — not a state `initialize()` can produce), and
`tests/auth/test_tool_permissions.py` forbids any tool appearing in two plugins'
permission mappings. Both adjustments are in the patch.

### What this costs the evaluations

Both skills define an ask-the-customer fallback, so the flows degrade rather than
break, and the scripted goldens answer the GPU question to reach
`recommend_model`. The cost is **permanently unexercised coverage**:

- the automated GPU-discovery path,
- the GPU product → planner-type substring mapping table in both skills,
- cluster-constrained `recommend_model` and its "requires new hardware" fallback
  slot.

Every run takes the ask branch. Per plan §3.1 that must be *recorded as
unexercised*, not read as passing — the spike prints the branch taken, and the
Phase 2 goldens must carry the same flag.

`tests/composites/test_skill_allowed_tools.py` keeps the drift guard active with
`get_cluster_resources` listed in a `KNOWN_UNRESOLVED` allowlist, plus two tests
that fail if the exemption ever goes stale (the tool becomes available, or no
skill declares it any more).

### navigator-deploy needed a skill fix

The two skills were not equally robust to the tool's absence, and only
`navigator` was safe:

- `navigator` — *"If `get_cluster_resources` is available in this session, call
  it"*, with an explicit ask path and the rule *"Tool unavailability alone is
  not permission to recommend without a hardware filter."*
- `navigator-deploy` — **"Call `get_cluster_resources`."** Unconditional. Its
  only fallback covered the tool *succeeding* with missing/empty `gpu_info`.
  Its pre-flight checks `get_use_case_defaults`, which is present by default, so
  pre-flight passed and the gap went undetected. Absence was undefined
  behaviour in a skill that subsequently writes to the cluster.

`navigator-deploy` Phase 3 Step 1 now mirrors `navigator`: availability guard,
the unavailable/fails/unmappable branch, an instruction not to substitute other
cluster tools or infer hardware, and an explicit wait-for-the-answer gate.
Verified live — the agent reproduces the new wording and asks instead of
improvising.

**Open item for Phase 2:** one run reached Phase 3 with `recommend_model` in its
tool list, and an ad-hoc re-run did not get that far, so whether the call ever
precedes the customer's GPU answer is unresolved. That ordering is exactly what
the §3.1 deterministic checks must decide; the variance between those two runs
is the argument for scripted checkpoints over manual inspection.

---

## 5. Pinned versions

From `uv.lock`, dependency group `skill-eval`:

| Package | Version |
|---|---|
| `deepeval` | 4.3.1 |
| `pydantic-ai-slim[openai,mcp]` | 2.64.0 |
| `pydantic-ai-harness[skills]` | 0.64.0 |
| `mcp` | 1.29.1 |
| `fastmcp-slim` | 3.4.11 |
| `openai` | 3.39.0 |
| `pydantic` | 2.14.1 |

Re-run the probe and the adapter tests when bumping any of these.

---

## 6. Not done

Phase 2 onward: checkpoints and goldens, the deterministic checks of §3.1, the
`recorder.py`/`reader.py` changes, judged metrics, the `make eval-skills`
targets, and OTel. No judge has run, so **no quality scores exist yet** — every
result above is connectivity and plumbing.
