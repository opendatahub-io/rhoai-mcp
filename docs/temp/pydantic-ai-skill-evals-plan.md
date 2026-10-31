# Plan: Pydantic AI + DeepEval skill evaluations (first iteration)

**Status:** proposal · **Date:** 2026-10-31 · **Scope:** first iteration, deliberately small

Complement/Replace the LightSpeed Core (LCORE) compose stack that `evals/` currently needs with a
small Pydantic AI agent that wires `rhoai-mcp` over **stdio** and exposes the two
agentskills.io skills in `.claude/skills/`. **Six evaluations: 2 skills × 3 models**
(OpenAI, EnMaaS/GLM, EnMaaS/DeepSeek). EvalHub orchestration is noted at the end as a
follow-on, not designed here.

---

## 1. Research findings

### 1.1 Pydantic AI is the right framework

[DeepEval's integrations](https://deepeval.com/integrations) cover LangChain, Pydantic AI,
OpenAI Agents, LangGraph, AgentCore, Strands, Google ADK, LlamaIndex and CrewAI. Pydantic
AI wins on four axes at once; nothing else wins on more than two:

| Criterion | Verdict |
|---|---|
| First-class DeepEval integration | Yes — verified in the installed `deepeval 4.3.1` (`deepeval.integrations.pydantic_ai`) |
| MCP client over stdio | Yes — `MCPToolset(StdioTransport(...))`, first-party |
| Loads agentskills.io skills from `.claude/skills` | Yes — official `Skills` capability, no parsing code needed |
| Swap model incl. OpenAI-compatible third parties | Yes — `OpenAIProvider(base_url=…, api_key=…)`, one line |
| Fit with this codebase | Excellent — repo is already pydantic + pydantic-settings throughout |

Rejected: **OpenAI Agents SDK** (OpenAI-shaped, more friction pointing at EnMaaS),
**LangGraph / CrewAI** (abstractions we would not use — contradicts the KISS and
minimal-dependencies rules in `CLAUDE.md`), **LlamaIndex / Strands / Google ADK** (no
advantage here).

### 1.2 Skills load natively — and discovery is the default

[`pydantic-ai-harness[skills]`](https://pydantic.dev/docs/ai/harness/skills/) (v0.64.0,
official) reads SKILL.md libraries straight off the filesystem, with `.claude/skills`
named in its own conventional-locations example:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Skills

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('.'), Skills('.agents/skills')],
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

That removes the one sharp version coupling in this stack — DeepEval ↔ Pydantic AI existed
*only* through `deepeval.integrations.pydantic_ai`, and it is the coupling with a narrow
published compat table and a silent failure mode
([#3145](https://github.com/confident-ai/deepeval/issues/3145), where a message-shape
mismatch made tool calls vanish from evaluations without error).

**The remaining pin to watch is `mcp`.** DeepEval validates by isinstance against
`mcp.types` — *"must be a list of 'MCPToolCall' with result of type 'CallToolResult' from
mcp.types"* — and Pydantic AI depends on `mcp` via fastmcp. So both sides pin the same
protocol SDK: `deepeval ← mcp → fastmcp ← pydantic-ai`. That is a far better seam than the
old one: a conflict surfaces as a visible dependency or type error, not as silently
missing tool calls.

**Reuse:** `evals/deepeval_helpers.py` already does the manual construction for
`LCSResult`. Feed it a Pydantic AI transcript instead and it works, along with
`evals/metrics/config.py`, `evals/providers/*` and `evals/reporting/*`.

Two verified gotchas for whoever retypes that module:

- **`Turn._mcp_interaction` is now a derived read-only property** in deepeval 4.3.1. The
  workaround at `deepeval_helpers.py:127-129` (`turn._mcp_interaction = True`) raises
  `AttributeError: property '_mcp_interaction' of 'Turn' object has no setter`. Delete it
   — the property already derives `True` whenever `mcp_tools_called` is non-empty.
- **DeepEval test cases do not round-trip through JSON.** `model_dump()` works, but
  `model_validate()` fails twice: `MCPServer` is not a pydantic model so nested dicts never
  rehydrate, and the validator rejects dict-shaped `mcp_tools_called`. Build the objects,
  do not try to reload them.

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

Keep the training-domain registration as-is so nothing regresses for servers that enable
`training`; the two can coexist, since the plugin sets are disjoint by default. Worth a
test asserting the tool is reachable under the *default* plugin configuration — that is
the condition that actually broke.

### 1.6 DeepEval writes almost nothing useful to disk

Verified against 4.3.1, because it determines where debugging information comes from:

- `DisplayConfig.file_output_dir` accepts **`md` or `html` only** — `file_type="json"`
  raises `ValueError` — and the export is gated behind `print_results=True`
  (`deepeval/evaluate/evaluate.py:311`).
- The artifact is a rendered console report: input, final output, metric scores,
  thresholds, judge reasons. **No tool arguments, no tool results, no intermediate turns.**
- `.deepeval/` holds no run history; `CacheConfig` exists to skip re-running identical
  metric/test-case pairs, not for inspection.
- `evaluate()` returns an `EvaluationResult` in memory. Richer inspection is designed to
  go to Confident AI, their hosted platform.

So a bad score on its own gives you a number and the judge's prose. Everything about *how
the conversation got there* has to come from the OTel trace (§3.6).

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
        Instrumentation(settings=InstrumentationSettings(version=5)),
    ],
    toolsets=[MCPToolset(StdioTransport(
        command='uv', args=['run', 'rhoai-mcp', '--transport', 'stdio', '--mock-cluster'],
    ))],
)
result = await agent.run("I need an LLM for our customer support bot, about 50 agents.")
```

**Use Pydantic AI's native instrumentation, not `DeepEvalInstrumentationSettings`.** The
latter is a real DeepEval class (`deepeval.integrations.pydantic_ai.instrumentator`) and
subclasses `InstrumentationSettings`, so it would drop in — but its constructor *"creates
a `TracerProvider` … sets the global tracer provider"* and exposes no parameter for
supplying your own provider or exporter. Its OTLP exporter is Confident AI's, authenticated
with an `x-confident-api-key` header. If we want spans in our own collector we have to own
the pipeline, and it will not hand it over. Skipping it also drops the DeepEval ↔
Pydantic AI version coupling (§1.3).

Pin `version=5` explicitly rather than inheriting the default: versions 2–4 are deprecated
and warn, and **version 6 is opt-in and changes message role structure for tool results**.
Anything parsing these spans is version-sensitive. (`Instrumentation(settings=...)` in
`capabilities` requires pydantic-ai ≥ 1.95; older 1.x used `Agent(instrument=...)`.)

Export OTLP with the standard `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable — the
collector is a deployment choice, not a code dependency (§3.6).

This is the production path for a chat assistant: not many type `/navigator`, instead they typically describe
a problem. All 53 tools are exposed — the agent cannot know which skill's allowlist
applies before it has chosen one.

Both skills are phase-gated and wait for the customer, so drive them with **canned
scripted replies** threaded through `message_history=result.all_messages()`. Deterministic
and reproducible, which matters when the point is comparing three models.

### 2.2 Layout

```
evals/
├── agent/
│   ├── models.py        # AgentModelSpec: openai | enmaas-glm | enmaas-deepseek
│   ├── runner.py        # build Agent + Skills + MCPToolset(stdio); drive scripted turns
│   └── transcript.py    # pydantic-ai messages → Transcript (in-memory; not persisted)
├── skills/
│   ├── goldens.py       # scenarios + scripted user replies
│   ├── test_navigator.py
│   └── test_navigator_deploy.py
├── deepeval_helpers.py  # REUSED — retyped LCSResult → Transcript protocol
├── metrics/, providers/, reporting/   # REUSED as-is
└── scenarios/, lcs_client.py          # existing LCORE path, untouched
```

The only change to existing code: `deepeval_helpers.py` and `reporting/recorder.py` are
typed against `LCSResult`. Introduce a `Transcript` protocol (`task`, `final_output`,
`tool_calls`, `messages`, `turns`, `tool_names_used`) that both satisfy, and retype those
two modules. Both backends then share one set of helpers.

### 2.3 Model matrix

```python
@dataclass(frozen=True)
class AgentModelSpec:
    id: str
    model_name: str
    base_url: str | None
    api_key_env: str

    def build(self) -> Model:
        provider = OpenAIProvider(api_key=os.environ[self.api_key_env], base_url=self.base_url)
        return OpenAIChatModel(self.model_name, provider=provider)
```

| id | `model_name` | `base_url` | key env |
|---|---|---|---|
| `openai-gpt4o` | `gpt-4o` | — | `RHOAI_EVAL_OPENAI_API_KEY` |
| `enmaas-glm` | e.g. `glm-4.6` | `RHOAI_EVAL_ENMAAS_BASE_URL` | `RHOAI_EVAL_ENMAAS_API_KEY` |
| `enmaas-deepseek` | e.g. `deepseek-v3` | `RHOAI_EVAL_ENMAAS_BASE_URL` | `RHOAI_EVAL_ENMAAS_API_KEY` |

[Red Hat MaaS](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/govern_llm_access_with_models-as-a-service/deploy-and-manage-models-as-a-service_maas)
is OpenAI-compatible, so all three are the same code path — only the tuple changes.

Selected by `RHOAI_EVAL_AGENT_MODELS` (comma-separated) to parametrise pytest, so CI can
run one model per job. Set `temperature=0` on all three. **Hold the judge model fixed**
across them, via the existing `RHOAI_EVAL_EVAL_*` settings and `create_judge_llm()` — if
the judge moves with the agent the comparison means nothing.

---

## 3. What we measure

### 3.1 Deterministic checks (no judge, free)

Read straight off the transcript.

**Routing** — did it load the right skill? Record `skills_loaded` (ordered) and whether
the expected one was loaded. This is a *precondition*, not a quality score: if the wrong
skill loaded, record the run as `routing_failure` and leave the judged metrics `n/a`
rather than `0`. Scoring "never loaded navigator" as "followed navigator badly" would be
misleading and unactionable.

**Navigator** — `get_use_case_defaults` and `get_expected_rps` called; `recommend_model`
called at least once and not before the workload profile is confirmed; the output table
has the 4 columns and 8 rows from the skill template with no empty cells (reuse
`_assert_table_completeness()` from the existing scenario verbatim); a
`<!-- navigator-handoff … -->` block is emitted and parses.

**Navigator-deploy** — `plan_deployment` is reached; `execute_deployment` is not called
before the scripted `"deploy"` turn; `create_serving_runtime` precedes it when
`runtime_needs_creation` is true.

### 3.2 `allowed-tools` as a reported dimension, not a gate

The field is optional per the agentskills spec and unimplemented by the harness, so we do
not filter on it. Instead, record per run: tools called, and which of them fall outside
the loaded skill's declared `allowed-tools`. Reported as a number, not an assertion.

Worth having because the skill bodies do make real boundary claims — `navigator` ends with
"Do not attempt to plan or execute a deployment" — and because comparing declared
allowlists against the server's actual tool list is exactly what surfaced §1.5. Add one
cheap no-LLM unit test asserting every `allowed-tools` entry resolves to a real tool; that
would have caught the drift the day it landed.

### 3.3 Judged metrics

Built from the transcript via the existing `deepeval_helpers.py`:

| Metric | Why |
|---|---|
| `MultiTurnMCPUseMetric` | Right tools **and** right arguments (score = min of both) |
| `MCPTaskCompletionMetric` | Did the conversation actually accomplish the task? |
| `ConversationalGEval` (one per skill) | Adherence to the skill document — the thing no stock metric knows |

The two GEval rubrics are what answer the original question, "is the use case as described
in the skill actually being delivered":

- **`navigator-fidelity`** — announces phases; waits at each gate; tables not raw JSON;
  asks about GPU inventory rather than inventing it; hands off instead of deploying.
- **`navigator-deploy-fidelity`** — surfaces plan `issues`/`warnings` before asking; never
  deploys without an explicit yes; explains failures instead of silently retrying.

Keep the existing `mcp_use_threshold` / `task_completion_threshold` knobs. Calibrate
thresholds against the strongest model once there is data — do not guess up front.

### 3.4 The matrix — 6 runs

| | OpenAI gpt-4o | EnMaaS GLM | EnMaaS DeepSeek |
|---|---|---|---|
| **navigator** | run | run | run |
| **navigator-deploy** | run | run | run |

Start with **2 goldens per skill** (happy path, vague-requirements path). Each run emits
one `EvalRecord` per (skill, golden, model), so `make eval-compare` works immediately.

Expect the smaller models to fail routing or the phase gates. That is a *result*, not a
harness bug. Keep three outcomes distinguishable in the record: `harness_error` (crash,
timeout, auth), `routing_failure`, and a normal scored run.

### 3.5 Reporting

DeepEval computes scores and reasons; **our recorder writes them**.
`evals/reporting/recorder.py` already extracts `score` / `success` / `threshold` /
`reason` per metric and appends a JSONL line — that is the file you will grep, not
anything DeepEval produces (§1.6).

`EnvironmentRecord.agent_model` already exists and is left empty by the LCS recorder;
populate it, with `agent_backend="pydantic-ai"`. Add to `EvalRecord`: `skill`,
`golden_id`, `skills_loaded`, `outcome`, and **`trace_id`** (§3.6). That is enough for
`python -m evals.reporting.cli compare` to produce the cross-model table.

### 3.6 Debugging a bad score

Scores come from DeepEval; the trajectory comes from the OTel trace. With
`include_content=True` (the Pydantic AI default, alongside `include_binary_content=True`),
the spans carry what the skill-wording loop needs:

| Attribute / span | What it answers |
|---|---|
| `gen_ai.system_instructions` | **The skill body as actually delivered** |
| `pydantic_ai.all_messages` | Full conversation, on the agent run span |
| `gen_ai.input.messages` / `gen_ai.output.messages` | Per-turn prompt and completion |
| Tool spans | Name, arguments, **and return values** |
| `gen_ai.tool.definitions` | The tool schemas offered to the model |
| MCP `call_tool` / `list_tools` spans | Which MCP calls were made |

Mapped onto the failure modes in §3.1: wrong skill loaded (the `load_capability` span and
its args), phase gate skipped (ordering of output vs. tool spans), `recommend_model` called
with bad arguments (tool span args), `gpu_info.products` misread (tool return value), table
rendered with empty cells (final output message). The first row is the one that makes this
worth wiring: you read the instructions the model received rather than inferring them.

**Correlate the two systems.** The judge's score and reason are not in the trace. Stamp the
same `run_id` / `golden_id` on the trace and on the `EvalRecord`, and persist `trace_id` in
the record, so a bad score is one lookup away from its trajectory instead of a timestamp
hunt. Cheap, but only if designed in from the start.

**Accepted trade-off: no offline re-scoring.** We do not persist DeepEval test cases, so
iterating on a `ConversationalGEval` rubric means re-running the agent, not re-judging a
recording. With six runs against a mock cluster that is affordable; it is recorded here so
the cost is chosen rather than discovered.

The collector is a deployment choice behind `OTEL_EXPORTER_OTLP_ENDPOINT` — `otel-tui` for
zero-infra local viewing, MLflow or anything else later, with no code change.

---

## 4. Phasing

Test-first per `CLAUDE.md`.

**Phase 0 — pin and spike (half a day).** Add a `skill-eval` dependency group:
`pydantic-ai-slim[openai]`, `pydantic-ai-harness[skills]>=0.64,<1`, `deepeval>=4.3,<5`,
plus the OTLP exporter. One agent, one model, one task, one tool call.

*Acceptance:* a `recommend_model` call appears in `result.all_messages()`, **and its
arguments and return value are visible in an exported OTLP span.** The second half is what
§3.6 depends on, so prove it before building on it.

Because we skip DeepEval's instrumentation (§2.1), pydantic-ai needs no ceiling — the
compat table and #3145 no longer apply. Two pins still matter:
`pydantic-ai-harness` is a `0.x` that iterates faster than core, and `mcp` is the shared
seam between DeepEval and Pydantic AI (§1.3). Treat a bump of either as cause to re-run
this spike.

**Phase 1 — unblock `get_cluster_resources` via option B (§1.5).** Register a
`get_cluster_resources` tool in `cluster-composites`, reusing the response shape already
built in `domains/training/tools/discovery.py`. Test-first: a failing test asserting the
tool is reachable under the **default** plugin set. No skill or contract changes.

**Phase 2 — harness + navigator + one model.** `evals/agent/*`, the `Transcript` retype,
`test_navigator.py` with the deterministic checks, against `openai-gpt4o`. No judge cost
yet. Fast feedback.

Two fixes land here while retyping `deepeval_helpers.py` (§1.3): delete the stale
`turn._mcp_interaction = True` workaround, which now raises on 4.3.1, and add the
`trace_id` / correlation plumbing from §3.6.

**Phase 3 — judged metrics, second skill, three models.** Adds the GEval rubrics,
`test_navigator_deploy.py`, EnMaaS credentials, and the 6-cell matrix. New targets:
`make eval-skills`, `make eval-skills-matrix`.

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

1. **EnMaaS specifics** — exact `base_url` and model identifiers for GLM and DeepSeek, and
   does each handle OpenAI-style tool calling well enough? Tool-calling fidelity varies a
   lot across hosted builds; worth a short curl probe before Phase 3.
2. **Judge model** — keep `gemini-2.5-flash` on Vertex, or something stronger for the
   GEval rubrics? A weak judge on a nuanced rubric adds noise.
3. **Which local OTel backend?** `otel-tui` is the zero-infra option Pydantic themselves
   name. MLflow is attractive later because EvalHub provides it automatically — but note
   MLflow's first-party route is `mlflow.pydantic_ai.autolog()`, a one-directional shim
   with a `pydantic-ai <= 2.51` ceiling, and **neither DeepEval nor Pydantic AI documents
   MLflow at all**. Whether MLflow accepts generic OTLP ingest is unverified; confirm
   before assuming it, since the vanilla-OTLP route is the one that keeps us uncoupled.
4. **Skill descriptions are now tested artifacts.** In discovery mode the model sees only
   `name` + `description`, and ours are a confusable pair — "LLM model selection" vs.
   "deploys a *chosen* LLM model". If routing fails, the fix may be a frontmatter edit
   rather than a model change. Worth knowing that before reading the first results.
