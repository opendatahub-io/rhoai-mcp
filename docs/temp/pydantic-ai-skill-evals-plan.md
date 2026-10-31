# Plan: Pydantic AI + DeepEval skill evaluations (first iteration)

**Status:** proposal · **Date:** 2026-10-31 · **Scope:** first iteration, deliberately small

Complement the LightSpeed Core (LCORE) compose stack that `evals/` currently needs with a
small Pydantic AI agent that wires `rhoai-mcp` over **stdio** and exposes the two
agentskills.io skills in `.claude/skills/`. **Six matrix cells: 2 skills × 3 models**
(OpenAI, EnMaaS/GLM, EnMaaS/DeepSeek), with **2 scenarios per skill = 12 conversations**
per full sweep. Start with one model before expanding the matrix. Keep both MCP metrics
and one skill-fidelity judge per skill. Persist transcripts for debugging; OTel is
planned. EvalHub orchestration is a follow-on, not designed here.

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

agent = Agent(
    'openai:gpt-4o',
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
  rehydrate, and the validator rejects dict-shaped `mcp_tools_called`. Build the objects
  from the live run, not from serialized test cases. Since we do not re-score offline
  (§3.6), this stays a warning against a tempting shortcut rather than a constraint the
  design has to work around.

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
on failure. Fixed scripts and `temperature=0` reduce variation; model outputs still vary.

### 2.2 Layout

```
evals/
├── agent/
│   ├── models.py        # AgentModelSpec: openai | enmaas-glm | enmaas-deepseek
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
history without duplicates before running paid evaluations. The native persisted history
remains the source; the shared representation and DeepEval objects are derived views.

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
run one model per job. Set `temperature=0` where supported and record effective settings;
do not silently fall back to different settings. **Hold the judge model fixed**
across them, via the existing `RHOAI_EVAL_EVAL_*` settings and `create_judge_llm()`.
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
unreplaced `[value]` placeholders; a `<!-- navigator-handoff … -->` block is emitted
after the customer's choice, parses, and agrees with the confirmed workload and selected
recommendation. `plan_deployment`, `execute_deployment`, and deployment writes through
other tools must never be called by this skill.

**Strengthen the table checker rather than copying it verbatim.** The current helper in
`evals/scenarios/test_navigator_table_completeness.py` accepts rows with only one
recommendation cell. Validate one comparison table's exact headers, row labels, and cell
counts; do not mix it with earlier workload tables. Add malformed-table examples to
exercise those checks.

**Navigator-deploy** — `plan_deployment` is reached; the plan is shown before requesting
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

### 3.4 The matrix — 6 cells, 12 conversations per sweep

| | OpenAI gpt-4o | EnMaaS GLM | EnMaaS DeepSeek |
|---|---|---|---|
| **navigator** | run | run | run |
| **navigator-deploy** | run | run | run |

Start with **2 goldens per skill**:

| Golden | Purpose |
|---|---|
| navigator: happy path | Confirm workload, show recommendations, confirm selection, emit handoff |
| navigator: vague requirements | Ask for missing intent/scale before proceeding; then complete the flow |
| navigator-deploy: approved deployment | Full chosen-model flow through plan, approval, and execution against the default mock state |
| navigator-deploy: handoff with consent withheld | Consume a complete handoff, skip redundant workload/recommendation phases, show the plan, and honor a final "hold off" reply without writes |

**Goldens take the default mock state as given.** `--mock-cluster` builds a fixed
`create_default_cluster_state()` at startup (`server.py:216-218`); there is no seeding
hook, no env var, and no scenario file. That state already contains both a
`ServingRuntime` and a `vllm-runtime-template`
(`mock_k8s/cluster_state.py:360-399`), so the `runtime_needs_creation` branch is
probably unreachable from it. Record which branch each golden actually took and mark
the unvisited one unexercised (§3.1) rather than adding a seeding mechanism to reach
it. Adding state control is a separate decision, to be made only if the unexercised
branches turn out to matter.

The handoff case uses a fixed valid fixture supplied in the scripted conversation, so it
depends on neither a preceding navigator run nor any mock-state change.
Start with one scenario per skill and one model, then expand to all four scenarios and
three models: **2 skills × 2 goldens × 3 models = 12 conversations**. Repeated attempts
multiply this count and each gets its own record and artifact.

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
an `attempt_id`, an artifact path, and optional `trace_id` (§3.6). Keep the skill and
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
  skill/scenario hashes, rubric version, dependency versions, and failure diagnostics.
  Exclude credentials from configuration snapshots.

Use Pydantic AI's native JSON serialization and `ModelMessagesTypeAdapter` for reloading;
see [storing and loading messages](https://pydantic.dev/docs/ai/core-concepts/message-history/#storing-and-loading-messages-to-json).

**Artifacts are for human inspection, not for re-scoring.** DeepEval objects are built
from the live in-memory run, never rebuilt from a persisted artifact. When a rubric
changes, re-run the agent. This keeps the artifact a plain debugging record — no
reconstruction path to keep faithful, no scoring records to link back to earlier
attempts, no artifact database or replay engine. Reloading needs to produce something
readable, not something scoreable. An offline re-score path can be added later if rubric
iteration proves expensive enough to justify it; nothing here forecloses that.
DeepEval's JSON report is supplementary (§1.6).

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

Create a parent span around the whole multi-turn conversation so one optional
`trace_id` covers all its agent runs. Stamp `run_id`, `attempt_id`, and `golden_id` on
that span and the saved record; otherwise separate `agent.run()` calls may produce
separate traces. Scores and reasons remain in the scoring records. An unavailable
collector must not prevent transcript capture or normal evaluations. `otel-tui` is a
local viewing option; other collectors can be evaluated later.

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

**Phase 2 — both skills, one model, deterministic checks.** Build the runner, artifact
save path, corrected shared adapter, stdio fixtures, checkpoints, and the `recorder.py`
/ `reader.py` changes only (§3.5). Delete the stale `turn._mcp_interaction = True`
workaround. Start with one scenario per skill against `openai-gpt4o`, then add the
vague-requirements and handoff/withheld-consent cases. Verify that failures and partial
transcripts are recorded before pytest assertions, that branch checks actually execute
and report unexercised branches honestly, and that the reader retains new metadata.
No judge cost yet; dry-run records identify this mode explicitly.

**Phase 3 — all required metrics, then the full model roster.** On the one-model path,
add **`MultiTurnMCPUseMetric`, `MCPTaskCompletionMetric`, and the relevant
`ConversationalGEval` rubric**. Then wire EnMaaS credentials for the models that passed
the Phase 0 probe and expand to the full sweep — 6 cells and **12 conversations** if all
three models qualified. Only now extend `comparison.py` and the CLI (§3.5), against real
multi-model records. Verify that routing failures, missing scores, and errors are
reported without biasing comparisons. New targets: `make eval-skills` (one-model
default, model selection via environment) and `make eval-skills-matrix`, plus an
explicit dry-run option for deterministic-only iterations.

**Phase 4 — trace viewing.** Add native OTel provider/exporter setup and a local
collector. Acceptance: one conversation trace contains the skill-loading result, MCP
arguments and results, and correlation IDs matching its saved record. Planned work, but
sequenced last deliberately: no phase above depends on it, and an unavailable collector
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

1. **Which models can actually tool-call?** — exact `base_url` and model identifiers for
   GLM and DeepSeek, and whether each handles OpenAI-style tool calling well enough.
   Tool-calling fidelity varies a lot across hosted builds. The Phase 0 probe answers
   this and sets the roster; the matrix stays at three models unless the probe gives a
   concrete reason to drop one.
2. **Judge model** — keep `gemini-2.5-flash` on Vertex, or something stronger for the
   GEval rubrics? A weak judge on a nuanced rubric adds noise.
3. **Which trace collector** — `otel-tui` is an option documented by Pydantic and the
   cheapest place to start. Verify generic OTLP support before choosing MLflow or
   another collector; their integration versions and deployment requirements must not
   block the transcript-based phases, which is why this is Phase 4.
4. **Skill descriptions are now tested artifacts.** In discovery mode the model sees only
   `name` + `description`, and ours are a confusable pair — "LLM model selection" vs.
   "deploys a *chosen* LLM model". If routing fails, the fix may be a frontmatter edit
   rather than a model change. Worth knowing that before reading the first results.
