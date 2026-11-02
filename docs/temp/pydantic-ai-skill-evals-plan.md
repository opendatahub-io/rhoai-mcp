# Plan: Pydantic AI + DeepEval skill evaluations (first iteration)

**Status:** implementation plan · **Date:** 2026-10-31 · **Updated:** 2026-11-02

Complement the LightSpeed Core (LCORE) compose stack that `evals/` currently needs with a
small Pydantic AI agent that wires `rhoai-mcp` over **stdio** and exposes the two
agentskills.io skills in `.claude/skills/`. The remaining sweep covers **Navigator
(model recommendation) only: 1 skill × 4 models × 2 goldens = 8 conversations**.
The models are GPT, Anthropic Claude, GLM, and DeepSeek, **all through EnMaaS**.
GPT and Anthropic are required family representatives; GLM and DeepSeek are the
two open-weight entries.
Start with one model before expanding the matrix. Keep both MCP metrics
and the Navigator skill-fidelity judge. Persisted transcripts are the scoring input, so
the same path re-scores them offline; OTel is planned. EvalHub orchestration is a follow-on, not designed here.

**Cost scope:** the remaining paid work is the eight Navigator conversations,
reviewed in batches of two per model. Navigator-deploy sweeps, new deployment
validation/calibration, and deployment-specific follow-up are deferred. Existing
deployment code, offline tests, calibration evidence, and historical results remain
available. Both skills stay in the discovery catalog so routing is still tested.

---

## 1. Research findings

### 1.1 Pydantic AI is the right framework

Pydantic AI fits the required agent loop without introducing a graph or workflow engine:

| Criterion | Verdict |
|---|---|
| Works with existing DeepEval metrics | Yes — build MCP test cases from the transcript; the tracing integration is not needed (§1.3) |
| MCP client over stdio | Yes — `MCPToolset(StdioTransport(...))`, first-party |
| Loads agentskills.io skills from `.claude/skills` | Yes — official `Skills` capability, no parsing code needed |
| Swap model incl. OpenAI-compatible third parties | Yes — `OpenAIProvider(base_url=…, api_key=…)`, one line |
| Fit with this codebase | Excellent — repo is already pydantic + pydantic-settings throughout |

Keep pytest and DeepEval rather than adding another evaluation framework. Native skill
loading avoids maintaining a custom skill parser or discovery tool. There is no need for
a broader framework comparison before proving this small path works.

### 1.2 Skills load natively — and discovery is the default

The official [`pydantic-ai-harness[skills]`](https://pydantic.dev/docs/ai/harness/skills/)
reads SKILL.md libraries from explicitly configured directories such as `.claude/skills`;
it does not automatically scan conventional locations:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Skills
from evals.agent.models import get_spec

agent = Agent(
    model=get_spec('enmaas-sol').build(),
    capabilities=[LocalWorkspace('.'), Skills('.claude/skills')],
)
```

It handles frontmatter validation, name normalisation, the catalog, and
`disable-model-invocation`. Its default behaviour is **progressive disclosure**: the model
sees each skill as name + description and pulls the body with a `load_capability` tool.
That is the mode we evaluate — see §2.

`allowed-tools` is listed among frontmatter fields the harness "accepts for compatibility
but does not implement". Per the agentskills spec the field is optional anyway, so we
treat it as an observation rather than an enforcement point (§3.2).

### 1.3 DeepEval is a scorer only — it never instruments the agent

DeepEval has two non-overlapping scoring paths:

| | Test cases built | Metrics |
|---|---|---|
| Tracing (the Pydantic AI integration) | Automatically, from OTel spans | `TaskCompletionMetric`, `StepEfficiencyMetric`, … |
| [MCP](https://deepeval.com/docs/evaluation-mcp) | **Manually** — `MCPServer` + `MCPToolCall` + `ConversationalTestCase` | `MCPUseMetric`, `MultiTurnMCPUseMetric`, `MCPTaskCompletionMetric` |

The Pydantic AI integration does not populate `mcp_servers` / `mcp_tools_called`, so the
MCP metrics require the manual path regardless.

**We take the MCP path and skip the tracing integration entirely** (§2.1). The payoff is
verified: importing the whole scoring path loads **zero** `pydantic_ai` modules.

```
>>> from deepeval.metrics import MultiTurnMCPUseMetric, MCPTaskCompletionMetric, ConversationalGEval
>>> [m for m in sys.modules if m.startswith('pydantic_ai')]
[]
```

That avoids the integration-specific DeepEval ↔ Pydantic AI version coupling, which has
a narrow published compat table and a silent failure mode
([#3145](https://github.com/confident-ai/deepeval/issues/3145), where a message-shape
mismatch made tool calls vanish from evaluations without error).

**The remaining pin to watch is `mcp`.** DeepEval validates by isinstance against
`mcp.types` — *"must be a list of 'MCPToolCall' with result of type 'CallToolResult' from
mcp.types"* — and Pydantic AI depends on `mcp` via fastmcp. So both sides pin the same
protocol SDK: `deepeval ← mcp → fastmcp ← pydantic-ai`. Dependency and type conflicts can
surface visibly here, but that does **not** prove the transcript adapter is faithful.
Contract tests must also catch silently missing or misattributed calls.

**Reuse with changes:** keep the metric factories and judge providers. Adapt
`evals/deepeval_helpers.py` to a shared transcript contract, but fix its call association
and preserve the additional skill-loading turns. Update reporting explicitly (§3.5).

Three verified gotchas for whoever adapts that module:

- **`Turn._mcp_interaction` is now a derived read-only property** in deepeval 4.3.1. The
  workaround at `deepeval_helpers.py:127-129` (`turn._mcp_interaction = True`) raises
  `AttributeError: property '_mcp_interaction' of 'Turn' object has no setter`. Delete it
   — the property already derives `True` whenever `mcp_tools_called` is non-empty.
- **Call association currently uses list position.** The loop at
  `deepeval_helpers.py:100-119` consumes one entry from `result.tool_calls` for every
  assistant tool call. A local probe with `load_capability` followed by `recommend_model`
  attached the recommendation to the skill-loading turn. Match calls and returns by
  **call ID**, and distinguish harness tools from MCP tools. Retain skill-loading
  instructions in the fidelity transcript without counting them as MCP interactions.
- **DeepEval test cases do not directly round-trip through JSON.** `model_dump()` works, but
  `model_validate()` fails twice: `MCPServer` is not a pydantic model so nested dicts never
  rehydrate, and the validator rejects dict-shaped `mcp_tools_called`. So never persist
  DeepEval objects — persist the native Pydantic AI messages, which do round-trip, and
  rebuild the DeepEval objects from them (§3.6).

### 1.4 The stdio server is self-contained — verified locally

```
$ uv run rhoai-mcp --transport stdio --mock-cluster
planner.planner - INFO - Loaded bundled BLIS benchmarks
→ 53 tools; recommend_model / get_use_case_defaults / get_expected_rps /
  plan_deployment / execute_deployment / list_use_cases all present and callable
```

`planner_mode` already defaults to `LOCAL` (embedded `llm-d-planner`), so no planner
backend, no llama-stack, no LCS, no compose. That is the whole point of this change.

### 1.5 Blocker: `get_cluster_resources` is not enabled by default

Both skills call **`get_cluster_resources`** in Phase 3 Step 1 and name it in
`allowed-tools`. It is absent from the 53 tools a default server exposes — but it is **not
missing**. It exists at `src/rhoai_mcp/domains/training/tools/discovery.py:92` and works
correctly; against the mock cluster it returns exactly what the skills expect:

```json
"gpu_info": {
  "product": "NVIDIA-A100-SXM4-80GB",
  "products": ["NVIDIA-A100-SXM4-80GB"],
  "total": 8, "available": 8, "nodes_with_gpu": 2
}
```

`products` maps cleanly through the skills' substring table ("A100" + "80" → `A100-80`).
Commit `a917f79` added the field for precisely this purpose. **The skill logic is right.**

The problem is packaging: the tool belongs to the **`training`** domain plugin, which is
not in the default `enabled_plugins` (`projects, connections, storage,
planner-composites, model_registry, inference, cluster-composites, meta-composites`).
Enabling `training` makes it appear and takes the server from 53 to 73 tools.

Two things make this worse than a config nit:

- `docs/mcp-api-stability.md` lists `get_cluster_resources` in the committed tool surface
  for **both RHAIRFE-1705 and RHAIRFE-1706**. A contractually-committed navigator tool
  ships disabled by default, so a default server breaks both skills at Phase 3.
- There is a domain-boundary smell underneath: the model-serving/navigator flow depends on
  a *training*-domain tool for GPU inventory.

#### Fix: option B — expose it from `cluster-composites`

| Option | Assessment |
|---|---|
| **A.** Add `training` to default `enabled_plugins` | One line, but pulls in ~20 tools to get one. Tool-count bloat degrades agent tool-selection, and the omission looks deliberate. |
| **B.** Expose `get_cluster_resources` from `cluster-composites` | **Chosen.** |
| **C.** Amend the skills to use `explore_cluster` | Does not work as-is: `explore_cluster` exposes `gpu_available` and `gpu_type`, and `gpu_type` is `"nvidia.com/gpu"` — the resource key, not the product. Useless for the GPU mapping table. Would require extending `explore_cluster`, rewriting both skills, *and* amending the stability doc. |

`cluster-composites` is enabled by default, already imports `TrainingClient` in six
places, and already calls `TrainingClient.get_cluster_resources()` at
`composites/cluster/tools.py:251` to build `explore_cluster`'s output. The dependency is
wired; it simply is not surfaced as a tool. Registering a thin `@mcp.tool()` wrapper there
— reusing the existing response shape from `domains/training/tools/discovery.py` — gives:

- one extra tool instead of twenty
- no SKILL.md edits, since the name both skills already use is preserved
- no change to the stability contract
- a fix for every consumer of the default plugin set, not just the evals

**Register the name exactly once in every plugin configuration.** Unconditionally
registering it in both places would break `test_no_tool_name_collisions()` in
`tests/integration/test_tool_registration.py` when all plugins are enabled. Default
plugin sets do not guarantee mutual exclusion.

Extract the existing response-building implementation into a shared helper. Let
`cluster-composites` own registration when enabled; retain a training-only fallback when
`cluster-composites` is absent. Determine ownership from the loaded plugin set, not hook
execution order. This preserves training-only users without duplicating the handler or
the registered name.

Carry the existing `nodes:list` permission mapping — currently at
`domains/permissions.py:145` — into `CLUSTER_PERMISSIONS` in
`composites/permissions.py`, retaining it for the training fallback. Test default,
training-only, and combined/all-plugin configurations: exactly one tool, the same
response shape, and the appropriate permission mapping in each.

### 1.6 DeepEval has structured local exports; native transcripts fill the MCP gap

Verified against installed **4.3.1** and the
[local-storage docs](https://deepeval.com/docs/evaluation-local-backend-storage):

- **`DisplayConfig(results_folder=...)` saves structured JSON test runs**, including
  scores, reasons, and conversation turns. It is separate from the older rendered
  HTML/Markdown `file_output_dir` interface and its `print_results` gate.
- The default JSON backend also keeps a rolling `.deepeval/.latest_run_full.json`.
  SQLite is available, but adds no value for this iteration. `CacheConfig` serves a
  different purpose: avoiding repeated metric work.
- **The legacy MCP fields used here are omitted by that export.** A local probe showed
  `mcp_tools_called` disappearing when converted to the exported API turn representation.
  A DeepEval JSON report alone is therefore insufficient to reconstruct our MCP case.

Keep our JSONL summary for comparisons. Save the native Pydantic AI message history and
the MCP tool-schema snapshot as the durable input to the adapter (§3.6). DeepEval's JSON
export is a useful additional report for scored attempts, not the source of truth for
agent trajectories. None of this requires a collector or a hosted service.

---

## 2. Design

### 2.1 Mode: discovery (the Pydantic AI default)

The agent gets both skills in its catalog and a natural-language task. It must pick the
right skill, load it, and follow it. No slash command, no body injected up front.

```python
agent = Agent(
    model=model_spec.build(),                                  # §2.3
    capabilities=[
        LocalWorkspace('.'),
        Skills('.claude/skills'),                              # both skills in the catalog
    ],
    toolsets=[MCPToolset(StdioTransport(
        command='uv', args=['run', 'rhoai-mcp', '--transport', 'stdio', '--mock-cluster'],
    ))],
)
async with agent:                                             # one server per conversation
    result = await agent.run("I need an LLM for our customer support bot, about 50 agents.")
    # Check the expected checkpoint before sending the next scripted reply.
    # Continue within this context, using message_history=result.all_messages().
```

Resolve the workspace and subprocess working directory from the repository root rather
than the caller's current directory. Keep the MCP connection open across **all turns**
of a conversation with `async with agent:`. Without an explicit context, toolsets open
and close as needed; a stdio server can restart between runs and lose mock-cluster state.
Start a fresh server for each scenario/model attempt so deployments do not leak into the
next scenario. Fetch tool schemas from this same connection; the skill fixtures must
override the existing HTTP/SSE schema-discovery fixtures instead of requiring LCORE.
See the [MCP lifecycle docs](https://pydantic.dev/docs/ai/mcp/client/#usage).

**OTel is planned but never a prerequisite** (§3.6). When enabled, use Pydantic AI's native instrumentation with
our own provider/exporter, not `DeepEvalInstrumentationSettings`. The latter owns a
global tracer provider and Confident AI exporter. Avoiding it keeps scoring independent
of the Pydantic AI tracing integration (§1.3).

Expose the complete default MCP tool set, including the GPU-inventory fix in §1.5; do
not filter tools to the expected skill. This tests discovery from a natural-language
request with both skills available.

Both skills are phase-gated, so use **scripted replies with checkpoints**. Each scenario
step defines the expected state before its reply: relevant tool calls or their absence,
required output structure, and the confirmation being requested. Check those properties
without demanding exact prose. Only then send the associated reply. If a checkpoint
fails, stop the script, save the partial transcript, and record the failure; do not send
a generic "yes" that could approve an unrelated action.

Thread history through `message_history=result.all_messages()`. When aggregating,
either use the final cumulative history or append each run's `new_messages()`; do not
concatenate successive `all_messages()` snapshots and duplicate earlier turns. Apply
explicit request/tool-call limits and a conversation timeout, preserving partial history
on failure. Fixed scripts reduce variation; record provider-default sampling and expect
model outputs to vary (§2.3).

### 2.2 Layout

```
evals/
├── agent/
│   ├── models.py        # Planned matrix: enmaas-sol | enmaas-opus | enmaas-glm | enmaas-deepseek
│   ├── runner.py        # build Agent + Skills + MCPToolset(stdio); drive scripted turns
│   └── transcript.py    # save/load native history + schemas; adapt to shared transcript
├── skills/
│   ├── conftest.py      # stdio agent/session and schemas; override HTTP/LCS fixtures
│   ├── goldens.py       # scenarios + checkpoints + scripted user replies
│   ├── test_navigator.py
│   └── test_navigator_deploy.py
├── deepeval_helpers.py  # shared conversion; call-ID matching and typed MCP results
├── metrics/, providers/ # reuse MCP metric factories and judge providers; add GEval rubrics
├── reporting/          # extend recorder, reader, models, and comparison handling
├── results/            # JSONL summaries + per-attempt transcript/report artifacts
└── scenarios/, lcs_client.py          # existing LCORE path, untouched
```

Introduce a small shared transcript contract (`task`, `final_output`, `tool_calls`,
`messages`, `turns`, `tool_names_used`) and tool-call records containing call IDs, names,
arguments, results, and error information. Adapt Pydantic AI's message parts to it; keep
the LCS path compatible. This includes behavioral adapter fixes (§1.3), not only type
annotations. Match parallel calls and returns by ID and preserve message order. Preserve
typed MCP content, structured results, and error flags when building DeepEval objects;
capture these at the MCP call boundary if native message normalization loses them.

Test mixed harness/MCP calls, out-of-order parallel returns, failed calls, and multi-turn
history without duplicates before running paid evaluations. The persisted artifact is the
scoring input (§3.6); the shared representation and DeepEval objects are derived views, so
these same adapter tests also cover save/load fidelity.

### 2.3 Model matrix

```python
@dataclass(frozen=True)
class AgentModelSpec:
    id: str
    model_name: str
    base_url: str | None
    api_key_env: str
    api: Literal["chat", "responses"] = "responses"

    def build(self) -> Model:
        if self.api == "messages":
            return AnthropicModel(
                self.model_name,
                provider=AnthropicProvider(
                    api_key=os.environ[self.api_key_env], base_url=self.base_url
                ),
            )
        provider = OpenAIProvider(api_key=os.environ[self.api_key_env], base_url=self.base_url)
        model_class = OpenAIResponsesModel if self.api == "responses" else OpenAIChatModel
        return model_class(self.model_name, provider=provider)
```

The required Phase 3 roster is four agent models, all through EnMaaS and using
`ENMAAS_KEY`. GPT/GLM/DeepSeek use the OpenAI-compatible `/v1/responses` endpoint;
Claude explicitly uses the native `/v1/messages` endpoint:

| Planned spec id | Family | `model_name` | API / qualification |
|---|---|---|---|
| `enmaas-sol` | GPT | `gpt-5.6-sol` | Responses; qualified, rechecked 2026-11-02 |
| `enmaas-opus` | Anthropic Claude | `claude-opus-4-6` | Native Messages; user-selected, 11/11 probe checks passed |
| `enmaas-glm` | Open-weight GLM | `rits/zai-org/glm-5-3` | Responses; qualified, goldens exercised |
| `enmaas-deepseek` | Open-weight DeepSeek | `rits/deepseek-ai/DeepSeek-V4.1-Flash-1017` | Responses; Phase 0 qualified |

Use `OpenAIResponsesModel` with `OpenAIProvider` for GPT/GLM/DeepSeek, and
`AnthropicModel` with `AnthropicProvider` for Claude; do not rely on model-name
inference. The Anthropic SDK appends `/v1/messages` itself, so its provider base URL
is **`https://api.enmaas.devshift.net`**, without `/v1`. The other specs retain
`https://api.enmaas.devshift.net/v1`. This explicit model/provider pairing follows
the [Pydantic AI Anthropic configuration](https://pydantic.dev/docs/ai/models/anthropic/).

The OpenAI-compatible `/v1/models` response omits Anthropic, and Opus returned 404
on Responses/Chat; its native Messages request succeeded. These observations
establish an API mismatch, not missing model access. The native 11-check probe
and Pydantic stdio/skill adapter spike are the qualification evidence; see
[the report](./enmaas-matrix-qualification.md) for results and limits.

`evals/agent/models.py` now selects these four entries for `matrix` and defaults
to `enmaas-sol` for a one-model run. The skill-eval dependency group includes the
Anthropic extra. No direct OpenAI or direct Anthropic endpoint belongs in this
sweep. Historical direct-provider specs remain explicit diagnostics outside the
default and matrix.

Selected by `RHOAI_EVAL_AGENT_MODELS` (comma-separated) to parametrise pytest, so CI can
run one model per job. Leave temperature unset for every matrix entry and record
provider-default sampling explicitly, following the qualified Phase 0 configuration.
Uniform requests do not guarantee uniform provider sampling. Do not silently change
settings to recover a failed run. **Hold the calibrated judge fixed:** EnMaaS
`gpt-6.1-sol`, cutoffs 0.60 / 0.75 / 0.80, MCP `conversation-v2`, fidelity `v2`, via
the existing `RHOAI_EVAL_EVAL_*` settings and `create_judge_llm()`.
Record the requested model identifier, provider-reported version when available, and
endpoint identity without credentials. One sweep is a smoke comparison, not evidence of
stable model rankings; use explicit repeated attempts for stronger conclusions rather
than automatically retrying failures until they pass.

---

## 3. What we measure

### 3.1 Deterministic checks (no judge, free)

Read straight off the transcript.

**Routing** — record successful `load_capability` calls in order, whether the expected
skill was loaded, and any unintended skill loads. If routing fails, record
`routing_failure` and fail the overall attempt. Leave skill-fidelity scoring `n/a` with
a reason; report routing success across **all attempts**, not just those that reach a
judge. MCP metrics still run wherever their transcript requirements are met (§3.3).

**Navigator** — `get_use_case_defaults` and `get_expected_rps` called; `recommend_model`
called at least once and not before the workload profile is confirmed; the output table
has exactly the 4 recommendation columns (plus the row-label column) and 8 named data
rows from the skill template, with a value or explicit `—` in every cell and no
unreplaced `[value]` or `[cluster fit]` placeholders; a `<!-- navigator-handoff … -->` block is emitted
after the customer's choice, parses, and agrees with the confirmed workload and selected
recommendation. `plan_deployment`, `execute_deployment`, and deployment writes through
other tools must never be called by this skill.

**Strengthen the table checker rather than copying it verbatim.** The current helper in
`evals/scenarios/test_navigator_table_completeness.py` accepts rows with only one
recommendation cell. Validate one comparison table's exact headers, row labels, and cell
counts; do not mix it with earlier workload tables. Add malformed-table examples to
exercise those checks.

**Navigator-deploy (retained implementation reference; outside the remaining sweep)** —
`plan_deployment` is reached; the plan is shown before requesting
deployment approval; neither `execute_deployment` nor runtime creation occurs before
the scripted affirmative reply. When `runtime_needs_creation` is true, successful
`create_serving_runtime` must precede execution. An absent confirmation, an unresolved
storage URI, or failed runtime creation must prevent deployment. The withheld-consent
scenario must have zero deployment writes even after the final user reply. Check that
execution parameters match the reviewed plan and the confirmed model/configuration;
calling the right tool with a different selection is not success.

These are hard checks independent of judge scores. Collect named check results and
failure details before asserting in pytest, so a failed check cannot erase the attempt
from the saved results (§3.5). Record which conditional branches each attempt actually
visited and mark the rest unexercised, rather than counting a conditional assertion as
coverage — with a fixed mock state (§3.4) several branches will stay unexercised, and
that must be visible rather than silently read as passing. Targeted storage- and
runtime-failure scenarios need mock-state control and are out of scope here.

### 3.2 `allowed-tools` as a reported dimension, not a gate

The field is optional per the agentskills spec and unimplemented by the harness, so we do
not filter on it. Instead, record per run: tools called, and which of them fall outside
the loaded skill's declared `allowed-tools`. Normalize Claude's
`mcp__rhoai-mcp__<name>` spelling to the server tool name, and report harness tools such
as `load_capability` separately. General allowlist deviations are observations; explicit
workflow boundaries and consent requirements remain hard checks (§3.1).

Worth having because the skill bodies do make real boundary claims — `navigator` ends with
"Do not attempt to plan or execute a deployment" — and because comparing declared
allowlists against the server's actual tool list is exactly what surfaced §1.5. Add one
cheap no-LLM unit test asserting every `allowed-tools` entry resolves to a real tool
under the default plugin configuration; that would have caught the drift the day it landed.

### 3.3 Judged metrics

**Both MCP metrics remain required in the scored evaluation suite**, alongside one
skill-specific GEval. Build their inputs through the corrected shared adapter:

| Metric | Why |
|---|---|
| `MultiTurnMCPUseMetric` | Right tools **and** right arguments (score = min of both) |
| `MCPTaskCompletionMetric` | Did the conversation actually accomplish the task? |
| `ConversationalGEval` (one per skill) | Adherence to the skill document — the thing no stock metric knows |

The two GEval rubrics are what answer the original question, "is the use case as described
in the skill actually being delivered":

- **`navigator-fidelity`** — announces phases; waits at each gate; tables not raw JSON;
  discovers GPU inventory or asks when unavailable; hands off instead of deploying.
- **`navigator-deploy-fidelity`** — surfaces plan `issues`/`warnings` before asking; never
  deploys without an explicit yes; explains failures instead of silently retrying.

Run both MCP metrics on every attempt whose captured transcript satisfies their input
requirements, including routing failures and failed deterministic checks. Do not skip
them to reduce cost or replace them with GEval. If a metric cannot run on an empty or
incomplete trajectory, record `n/a` and its specific missing precondition. Judge errors
are errors, not zero scores. Skill fidelity is conditional on successful routing, and
its rubric receives the expected skill document and full relevant conversation,
including loaded instructions.

Make scenario intent explicit in the scripted user messages and case scenario/context:
honoring a user's decision to withhold deployment is successful completion of that
scenario. Validate this interpretation with the MCP task-completion judge during
calibration. Hard consent checks still decide whether the workflow was followed,
regardless of the judge's interpretation.

Keep the existing `mcp_use_threshold` / `task_completion_threshold` knobs. Calibrate
thresholds against reviewed passing and failing examples with a fixed judge, not only
against whichever agent model scores highest. Report every metric separately; avoid a
combined number that could hide a workflow violation.

### 3.4 The remaining matrix — Navigator only, 8 conversations

| | EnMaaS GPT | EnMaaS Anthropic | EnMaaS GLM | EnMaaS DeepSeek |
|---|---|---|---|---|
| **navigator** | 2 goldens | 2 goldens | 2 goldens | 2 goldens |

Run these **two Navigator goldens per model**:

| Golden | Purpose |
|---|---|
| navigator: happy path | Confirm workload, show recommendations, confirm selection, emit handoff |
| navigator: vague requirements | Ask for missing intent/scale before proceeding; then complete the flow |

**Historical sweep:** all **8/8 planned attempts are recorded; 5/8 passed**. GPT and GLM
passed both goldens, DeepSeek one, and Anthropic neither under the frozen
configuration. [Full sweep report](./navigator-sweep.md) and
[records](./navigator-sweep-results.json). There were no timeouts or judge errors;
Anthropic's vague case stopped after one turn at the skill-load checkpoint, with
fidelity `n/a`. Earlier batches and GLM attempts remain historical evidence.

**Review before comparison:** the user reports successful automatic discovery
with the same Opus model in Claude Code. The [harness investigation](./anthropic-harness-investigation.md)
found omitted native thinking configuration, a catalog/checkpoint contract gap,
and bold table labels causing two coupled parser failures. These harness
corrections are now implemented: bold labels are accepted, Opus explicitly uses
adaptive thinking, and every model receives a shared instruction to load a
relevant skill before responding or using domain tools. Thinking alone did not
resolve discovery in a two-case diagnostic; the shared instruction then restored
discovery and completion of both full scripts, with all six judge scores passing.
Both emitted peak instead of average QPS and failed the then-current hard
check. That stage passed all 299 offline tests.
[Correction and separate live evidence](./anthropic-correction.md). Review this
macro step before skill-text changes or further paid validation. Original sweep
records and verdicts remain unchanged; do not pool configurations silently.

**Handoff investigation:** the [offline contract trace](./navigator-handoff-qps-investigation.md)
shows Navigator's average-demand handoff conflicts with the recommendation and
deployment API's buffered capacity. Both planner adapters overwrite capacity
with the handoff value, without restoring the buffer/floor. The [correction](./navigator-handoff-capacity-correction.md)
is now implemented: the handoff and strict check use the selected successful
recommendation's `specification.traffic_profile.expected_qps`. Phase 2 still
displays average/peak traffic. All 330 offline tests pass, including missing or
invalid source evidence, capacity floors, source provenance, and the planning
contract. A separate 12-artifact counterfactual replay changes only the QPS
check; original verdicts and judge scores remain unchanged. Updated Navigator
skill and scenario hashes identify the new contract. That correction used no
paid calls.

**Updated sweep complete:** all four EnMaaS models ran again under the shared
host instruction and corrected handoff contract: **8/8 recorded, 7/8 passed**.
Anthropic's first pair passed and was reviewed before the remaining six; its
pair counts toward the eight. GPT and GLM also passed 2/2 each, DeepSeek 1/2.
All 24 judge scores passed, and every conversation completed all five turns
with correct capacity QPS. DeepSeek's vague case failed only on three empty
Cluster fit cells, reproducing the known template ambiguity.

Commit `e59d9e8`, Navigator skill `25392e419e13`, scenario hashes
`eaa225a67e99` / `f1f282812e97`, 280 source/test/configuration files, and the
judge/cutoffs/limits stayed fixed. GLM and DeepSeek overlapped for 151.99 seconds
at the user's request; DeepSeek's judge scratch storage was isolated. Treat
timing as descriptive under concurrent load. There were no reruns, timeouts,
harness/judge errors, missing scores, or deployment writes. Historical records
are preserved, and Navigator-deploy remains outside the paid scope.
**Subsequent template clarification:** the Cluster fit row now has explicit
`[cluster fit]` cells in all four recommendation columns, with one fit label per
column and `—` for a missing recommendation. All 18 existing table-check
tests pass, and replay preserves every check result in the eight saved
conversations. The clarified skill hash is `6b84c869d2cc`; no live evaluation
has used it yet. Keep the completed 7/8 sweep tied to `25392e419e13` when
building comparison/reporting. No paid call was made for this one-row change.
The next implementation step is the comparison/reporting CLI.
[Updated report](./navigator-resweep.md),
[records](./navigator-resweep-results.json).

**Goldens take the default mock state as given.** `--mock-cluster` builds a fixed
`create_default_cluster_state()` at startup (`server.py:216-218`); there is no seeding
hook, no env var, and no scenario file. Keep unvisited branches visible rather than
adding state control for this sweep. Navigator-deploy's existing approved and
consent-withheld goldens remain available for offline regression and historical
replay; further paid deployment evaluation is deferred.

The remaining full sweep is **1 skill × 2 goldens × 4 models = 8 conversations**.
A run omitting a required model is a partial sweep. Run one model's two goldens per
reviewable macro step, with the same scoring configuration. Prevent idle sleep for
local runs (`caffeinate -i` on macOS), disable pytest capture (`-s`) for stdio, and
disable automatic retries (`--reruns 0`). Repeated attempts
multiply this count and each gets its own record and artifact.

Use `evals/skills/test_navigator.py` as the explicit pytest target. The
`eval-skills`, `eval-skills-judge`, `eval-skills-matrix`, and `eval-skills-traced`
Make targets now select only that file. Do not target the whole `evals/skills/`
directory for the remaining paid sweep. Both skills remain discoverable by the
agent; only Navigator tasks are scripted and judged.

Keep `outcome` distinct from pass/fail: `harness_error` (setup, auth, adapter, or judge
failure, with `error_stage`), `model_failure` (unrecoverable model-response/argument failure or
usage-limit exhaustion), `routing_failure`, and an `evaluated` attempt that can pass or
fail. Record timeouts with their stage and available diagnostics rather than treating
every timeout as the same failure. Preserve routing and check results even if a later
judge fails. A model failure is an evaluation result, not automatically a harness bug.

### 3.5 Reporting

DeepEval computes scores and reasons; **our recorder writes comparison summaries**.
`evals/reporting/recorder.py` already extracts `score` / `success` / `threshold` /
`reason` per metric and appends a JSONL line. Keep that path and optionally enable
DeepEval's structured JSON export as a companion report (§1.6).

`EnvironmentRecord.agent_model` already exists and is left empty by the LCS recorder;
populate it, with `agent_backend="pydantic-ai"`. Extend records with `skill`, `golden_id`,
`skills_loaded`, `outcome`, `error_stage`/diagnostics, named deterministic-check results,
an `attempt_id`, a `scoring_id` (a re-score reuses the `attempt_id`), an artifact path,
and optional `trace_id` (§3.6). Keep the skill and
scenario hashes, effective model/judge settings, and rubric version with the artifact
so the comparison configuration is recoverable.

**In Phase 2, change only the write and read paths** — not the comparison layer:

- `recorder.py`: capture agent execution as well as scoring time, save partial artifacts
  and one summary for every attempt, and handle missing/error metric values. Persist
  before the final pytest assertion. Preserve results from a dry run without inventing
  judge scores. An empty metric list must not make a judged attempt pass automatically.
- `reader.py`: explicitly load the new fields and provide defaults for older LCS records;
  it currently selects known fields and would otherwise discard the new metadata.

Getting these two right is what makes the data trustworthy later; a correct record can
always be re-read, while a lossy one cannot be recovered. Overall pass requires
successful routing, all hard checks, and all required metrics passing; absent required
scores cannot produce a pass. Dry runs are recorded as such.

**Defer `comparison.py` and the CLI to Phase 3**, when a matrix exists to compare. One
model and a handful of conversations are read directly from the JSONL and the
per-attempt artifacts; a cross-model ranking view built before there are cross-model
results is speculative work. When it does land it must: compare matching skill/golden
sets and fixed judge/rubric configurations within a selected sweep; refuse to pool
unrelated scenarios or arbitrary recent records into a ranking; show sample counts and
leave missing values `n/a` rather than substituting `0.0` as the current comparison
does; and report routing success, pass rate, errors, and per-metric means separately,
with a coverage denominator on conditional judged quality.

The remaining comparison scope is Navigator's happy-path and vague-requirements
goldens across the four EnMaaS models. Select those records explicitly; historical
Navigator-deploy records do not enter this comparison. Deployment reporting and
additional paid validation are deferred under the cost scope.

Retain the LCS reporting behavior through backward-compatible defaults. The new
`make eval-skills` entry point selects a coherent sweep. Existing `make eval-compare`
keeps working on LCS records and is not extended to this matrix until Phase 3.

### 3.6 Debugging artifacts

**The baseline is a saved transcript, not a running collector.** Write one artifact per
attempt under `evals/results/<run_id>/<attempt_id>/`, including partial failed attempts:

- Native Pydantic AI message history, preserving tool-call IDs, user replies, intermediate
  assistant output, tool results, and the `load_capability` result containing the skill
  body actually delivered. Save the configured skill files/catalog as well, so routing
  failures still have a record of the descriptions offered to the model.
- MCP tool schemas from the same server session, and any captured MCP result/error
  records needed to preserve information lost by native normalization (§2.2).
- Scenario/checkpoint definition, check results, effective agent/judge configuration,
  skill/scenario hashes, rubric version, dependency versions, `trace_id` when tracing is
  on (§4), and failure diagnostics. Exclude credentials from configuration snapshots.
  Keeping `trace_id` in the artifact lets a re-score — which runs no agent and so has no
  trace of its own — point back at the original attempt's trace.

Use Pydantic AI's native JSON serialization and `ModelMessagesTypeAdapter` for reloading;
see [storing and loading messages](https://pydantic.dev/docs/ai/core-concepts/message-history/#storing-and-loading-messages-to-json).

**The artifact is the scoring input — in the live run too.** Every attempt goes agent →
write artifact → load artifact → adapt → score. That is one scoring path rather than two,
and a lossy artifact then fails the live run instead of being discovered months later.
Re-scoring is the same path minus the agent, which is what makes rubric calibration (§3.3)
and table-checker work (§3.1) a controlled comparison on frozen transcripts instead of on
a fresh conversation each time.

Explicit non-goals: no artifact database, no agent replay, no provenance graph beyond the
shared `attempt_id`, no artifact migration — a stale-schema artifact fails to load and is
re-run. DeepEval's JSON report stays supplementary (§1.6).

**OTel (planned, §4):** use native `Instrumentation(settings=InstrumentationSettings(...))`
with our own `TracerProvider`, `BatchSpanProcessor`, and `OTLPSpanExporter`. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` for the exporter and flush/shut down the provider at the
end of the run. The environment variable **alone does not configure exporting**. Follow
the [raw OTel setup](https://pydantic.dev/docs/ai/integrations/logfire/#otel-without-logfire).

When enabled, explicitly select the instrumentation format supported by the locked
version (the initial proposal uses `version=5`) and `include_content=True`. The adapter
and scoring path consume native messages, not version-sensitive span attributes.
Verify the following in the trace spike:

| Attribute / span | What it answers |
|---|---|
| `gen_ai.system_instructions` | Initial instructions and skill catalog; do not assume the loaded body is here |
| `load_capability` tool result | **The skill body as actually delivered in discovery mode** |
| `pydantic_ai.all_messages` | Conversation history on the agent run span |
| `gen_ai.input.messages` / `gen_ai.output.messages` | Per-turn prompt and completion |
| Tool spans | Name, arguments, and return values |
| `gen_ai.tool.definitions` | Tool schemas offered to the model |

The [Skills docs](https://pydantic.dev/docs/ai/harness/skills/#how-it-works) describe
loading the body as a tool result, rather than injecting it into the initial system
instructions. Inspect that result when debugging skill delivery.

**Amended after the MLflow spike.** The original instruction here was to wrap the
whole conversation in one parent span so a single `trace_id` covered all its agent
runs, treating separate traces per `agent.run()` as the failure to avoid. MLflow's
model is the opposite, and it is the one to follow: *"Each trace in the session
represents one turn: the user's input and the system's response"*, with turns linked
by a shared session id. A single-trace conversation makes its root a wrapper span
carrying no inputs or outputs — a case MLflow's own guidance names — leaving the trace
list blank and burying each turn's new user message under the cumulative history.

So: wrap **each scripted turn** as its own root span, and group them with
`session.id`. MLflow scans spans for `session.id` or `gen_ai.conversation.id` and
takes the first found, so setting `session.id` to the `attempt_id` makes the session
key the same one the record and artifact already use — correlation is preserved
without a separate `trace_id`. Stamp `run_id`, `attempt_id`, `golden_id` and the turn
index on each turn span, and set `mlflow.spanInputs`/`mlflow.spanOutputs` to the
turn's prompt and reply so the trace list is readable; the agent-run span does not
supply these, because Pydantic AI puts messages on model-request spans only. Scores
and reasons remain in the scoring records. An unavailable collector must not prevent
transcript capture or normal evaluations.

---

## 4. Phasing

Test-first per `CLAUDE.md`.

**Phase 0 — pin and spike.** Add a `skill-eval` dependency group:
`pydantic-ai-slim[openai,mcp]`, `pydantic-ai-harness[skills]>=0.64,<1`, and
`deepeval>=4.3,<5`. The `mcp` extra is required for the FastMCP client; the server's
existing `mcp` dependency alone does not provide it. Resolve compatible versions in
`uv.lock` and record the tested versions; do not rely on open-ended constraints as
evidence of compatibility. Keep tracing SDK/exporter dependencies optional.

Remove the leftover untracked `evals/pydantic_ai/` spike directory (stale `.venv`,
caches, and orphaned bytecode) so it does not collide with the `evals/agent/` and
`evals/skills/` layout in §2.2.

**Probe tool-calling fidelity across candidate models here, not in Phase 3.** A short
curl or one-off script against each candidate endpoint — multi-turn OpenAI-style tool
calling with a non-trivial argument shape — decides which models enter the matrix. The
point is to establish the roster on evidence rather than to shrink it: a candidate that
cannot reliably emit tool calls produces `model_failure` rows that say nothing about the
skills, so it is better excluded knowingly than discovered mid-sweep. Record which
candidates passed, which failed and how, and keep the matrix at every model that passed.

*Acceptance:* one agent/model connects over stdio, loads a skill, makes a
`recommend_model` call, and continues a second turn within the same agent context. Build
typed DeepEval objects from that live run with the call attributed to the correct turn.
Separately, save the attempt artifact and reload it into a readable transcript (§3.6).
A full phase-gated skill flow is not required for this connectivity/adapter spike.

Skipping DeepEval's tracing integration avoids that integration's compatibility table;
it does not remove our own adapter and dependency compatibility obligations. Re-run the
spike and adapter tests when bumping core/harness, FastMCP, `mcp`, or DeepEval.

**Phase 1 — unblock GPU discovery via option B (§1.5).** Extract a shared handler and
register `get_cluster_resources` once, with cluster ownership and a training-only
fallback. Carry over permission mappings. Test default, training-only, and combined
configurations and retain the all-plugin collision check. No skill or contract changes.

**Phase 2 — both skills, one model, deterministic checks (implementation history).** Build the runner, artifact
save/load path, corrected shared adapter, stdio fixtures, checkpoints, and the
`recorder.py` / `reader.py` changes only (§3.5). Scoring reads the artifact from the
start, so the single path costs nothing to adopt here. Delete the stale `turn._mcp_interaction = True`
workaround. Start with one scenario per skill against `enmaas-sol`, then add the
vague-requirements and handoff/withheld-consent cases. Verify that failures and partial
transcripts are recorded before pytest assertions, that branch checks actually execute
and report unexercised branches honestly, and that the reader retains new metadata.
No judge cost yet; dry-run records identify this mode explicitly.

**Phase 3 — Navigator-only model sweep and comparison.** Use the calibrated
conversation MCP use/completion metrics and Navigator fidelity rubric. Run
Navigator's happy-path and vague-requirements goldens across GPT, Anthropic, GLM,
and DeepSeek, all via EnMaaS: **1 skill × 2 goldens × 4 models = 8 conversations**.
Review one model's two conversations before progressing to the next model.
Anthropic uses the explicitly qualified native Messages configuration (§2.3); do
not silently omit the family or switch its provider. The runtime matrix is aligned
with the four-model roster. Keep the calibrated EnMaaS judge and scoring settings
fixed. The first sweep is now complete (8 attempts, 5 passes); review the
Anthropic harness/checker findings in §3.4 in separate macro steps before
extending `comparison.py` and the CLI (§3.5) for the Navigator records.
Report routing failures, missing scores, and errors without biasing comparisons.
The skill-eval Make targets select `evals/skills/test_navigator.py`; use free
deterministic replay for checker work. Further paid Navigator-deploy runs and
deployment-specific follow-up are deferred.

**Phase 4 — trace viewing (deferred under the cost scope).** The retained design
adds native OTel provider/exporter setup and a local
collector. Acceptance: one conversation trace contains the skill-loading result, MCP
arguments and results, and correlation IDs matching its saved record. Planned work, but
sequenced after the Navigator sweep and comparison: no phase above depends on it, and an unavailable collector
must never prevent transcript capture or normal evaluation runs (§3.6).

LCORE stays as-is throughout. Deciding its fate is a separate conversation once this path
has produced some signal.

---

## 5. Follow-on: EvalHub (not designed here)

Noted so the design does not paint us into a corner.

The [contrib DeepEval adapter](https://github.com/eval-hub/eval-hub-contrib) will not fit
— it is dataset-driven (static `input` / `actual_output` + judge), with no agent loop and
no MCP. We would need a
[BYOF adapter](https://developers.redhat.com/articles/2026/06/09/bring-your-own-evaluation-framework-evalhub):
implement `run_benchmark_job(config: JobSpec, callbacks) -> JobResults` in a UBI9
container, running the MCP server as a stdio subprocess inside it.

The one thing that matters for *this* iteration: `JobSpec.model.url` and
`JobSpec.model.name` map exactly onto `AgentModelSpec` (§2.3). Keeping the model matrix
parameterised by `(base_url, api_key, model_name)` is what makes the later adapter a thin
shim rather than a rewrite.

---

## 6. Open questions

1. **Anthropic through EnMaaS — API resolved** — user-selected `claude-opus-4-6`
   works on native Messages and passes the tool-calling probe. Configure
   `AnthropicModel`/`AnthropicProvider` explicitly, using the EnMaaS root base URL.
   GPT, Anthropic, GLM, and DeepSeek are the required four-model roster. Qualification
   is a connectivity/tool-use smoke test, not a passing four-golden result.
   [Probe evidence](./enmaas-matrix-qualification.md).
2. **Judge model — resolved** — calibrated EnMaaS `gpt-6.1-sol` is fixed across agent
   families, with the reviewed cutoffs and versions in §2.3. See
   [judge calibration](./judge-calibration.md).
3. **Which trace collector** — `otel-tui` is an option documented by Pydantic and the
   cheapest place to start. Verify generic OTLP support before choosing MLflow or
   another collector; their integration versions and deployment requirements must not
   block the transcript-based phases, which is why this is Phase 4.
4. **Skill descriptions are now tested artifacts.** In discovery mode the model sees only
   `name` + `description`, and ours are a confusable pair — "LLM model selection" vs.
   "deploys a *chosen* LLM model". If routing fails, the fix may be a frontmatter edit
   rather than a model change. Worth knowing that before reading the first results.
