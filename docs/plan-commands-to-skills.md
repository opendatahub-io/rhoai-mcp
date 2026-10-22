# Plan: Navigator Commands to Skills with Claude Code Workflow Checks

## Goal and scope

1. Migrate only `navigator.md` and `navigator-deploy.md` to agentskills.io `SKILL.md` format.
2. Leave `.claude/commands/scaffold-plugin.md` in place and unchanged.
3. Build a thin Python wrapper around `ClaudeSDKClient`, then use it for simple before/after checks through Claude Code. **Deterministic workflow assertions are the primary migration gate.**
4. Preserve the before/after conversations so DeepEval can score the same interactions in a later implementation step. DeepEval remains valuable complementary validation, and the existing LCS evaluation path remains available.

This is a one-shot migration. Use the available Claude Code installation and model; no special version/model pinning, repeated-trial framework, or statistical equivalence claim is required. Passing the same explicit workflow contract before and after provides functional interchange evidence for the exercised paths, not a guarantee for every possible conversation.

Anthropic documents command files and skills as exposing the same `/name` behavior, with existing commands continuing to work. The migration changes packaging, not the navigator instructions. See [Claude Code skills](https://code.claude.com/docs/en/skills):

> Custom commands have been merged into skills. A file at .claude/commands/deploy.md and a skill at .claude/skills/deploy/SKILL.md both create /deploy and work the same way. Your existing .claude/commands/ files keep working. Skills add optional features: a directory for supporting files, frontmatter to control whether you or Claude invokes them, and the ability for Claude to load them automatically when relevant.

So a Skill is a functional equivalent of a (old) Command.

---

## Part 1: Migration — two navigator commands only

### Current state

```text
.claude/commands/
├── navigator.md
├── navigator-deploy.md
└── scaffold-plugin.md
```

### Target state

```text
.claude/
├── commands/
│   └── scaffold-plugin.md          # stays here, unchanged
└── skills/
    ├── navigator/
    │   └── SKILL.md
    └── navigator-deploy/
        └── SKILL.md
```

### Frontmatter and body

| Field | Migration |
|---|---|
| `name` | Add `navigator` or `navigator-deploy`, matching the parent directory. |
| `description` | Preserve the existing description. |
| `allowed-tools` | Replace comma separators with spaces, preserving the exact tool names and order. |
| Markdown body | Preserve byte-for-byte after the closing frontmatter delimiter. |

Both migrated skills must have valid YAML, non-empty descriptions, and names satisfying the [Agent Skills specification](https://agentskills.io/specification). Its experimental `allowed-tools` field uses a space-separated string. Claude Code supports that representation too.

Example migrated `navigator/SKILL.md` frontmatter:

```yaml
---
name: navigator
description: Guides customers through LLM model selection for Red Hat OpenShift AI
allowed-tools: mcp__rhoai-mcp__recommend_model mcp__rhoai-mcp__get_use_case_defaults mcp__rhoai-mcp__get_expected_rps mcp__rhoai-mcp__list_use_cases mcp__rhoai-mcp__list_data_science_projects mcp__rhoai-mcp__get_cluster_resources
---
```

Record baseline checks before creating either skill. Then create the two skills and remove only the corresponding navigator command files. Keep `.claude/commands/` and `scaffold-plugin.md`. Run after-checks with the old navigator command files absent, so a missing skill cannot silently fall back to its old command.

## Part 2: Shared thin wrapper around ClaudeSDKClient

New module: `evals/claude_code_client.py`.

Use the official `claude-agent-sdk` package, whose `ClaudeSDKClient` supports continuing exchanges on a single connection. Let the SDK manage Claude Code and the stdio MCP subprocess; do not implement a second subprocess/JSONL transport or restart Claude Code with `--resume` for each turn. See the [Python SDK reference](https://code.claude.com/docs/en/agent-sdk/python#claudesdkclient).

```text
simple workflow runner ─┐
                       ├─ ClaudeCodeClient → ClaudeSDKClient → Claude Code
later DeepEval path ───┘                                      ↓
                                                    rhoai-mcp over stdio
```

### Dependencies and MCP configuration

Add a separate optional dependency group, for example:

```toml
[dependency-groups]
claude-check = ["claude-agent-sdk"]
```

The simple checks require this group, the repository's runtime dependencies, and Claude authentication. They must not import DeepEval, instantiate `EvalConfig`, or require a judge API key. Keep the existing `eval` dependency group separate.

Build one shared MCP configuration in Python. Resolve the repository root from the module location, and run the MCP server with the same Python environment as the runner:

```python
mcp_servers = {
    "rhoai-mcp": {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "rhoai_mcp", "--transport", "stdio", "--mock-cluster"],
        "env": {
            "RHOAI_MCP_PLANNER_MODE": "local",
            "RHOAI_MCP_PLANNER_MODEL_CATALOG_URL": "",
            "PLANNER_DETECT_CLUSTER_GPUS": "false",
        },
    }
}
```

The explicit transport and local planner avoid inheriting a remote planner or HTTP transport from local configuration. The empty catalog URL forces bundled benchmark data, and disabling independent planner GPU discovery leaves hardware discovery to the navigator workflow. The default plugin set does not expose `get_cluster_resources`, so the script supplies manual GPU information when prompted. No planner service or container is needed.

Configure `ClaudeAgentOptions` with `cwd=repo_root`, `setting_sources=["project"]`, `system_prompt={"type": "preset", "preset": "claude_code"}`, and this MCP configuration. Include project settings so repository commands and skills are discovered. Explicitly pre-approve the MCP tools needed across all scripted turns, including deployment, so checks can complete without permission prompts. Workflow confirmation is still supplied as a separate user message and checked independently of tool permissions. See [SDK skill loading and command dispatch](https://code.claude.com/docs/en/agent-sdk/skills).

### Small result contract, independent of DeepEval

Use plain dataclasses with the attributes consumed by the current scenarios and conversion helpers:

```python
@dataclass
class ClaudeToolCall:
    name: str                       # MCP schema name, e.g. recommend_model
    arguments: dict[str, Any]
    result: Any                     # serializable tool result content
    call_id: str
    is_error: bool = False

@dataclass
class ClaudeCodeResult:
    task: str
    final_output: str
    tool_calls: list[ClaudeToolCall]
    messages: list[dict[str, Any]]    # normalized role/content/tool_calls messages
    turns: int = 0
    conversation_id: str = ""       # SDK session ID

    @property
    def tool_names_used(self) -> list[str]:
        return [call.name for call in self.tool_calls]
```

The wrapper exposes an async context manager and two operations:

```python
async with ClaudeCodeClient(repo_root=repo_root) as client:
    first = await client.query(task, skill="navigator")
    next_turn = await client.follow_up(reply, first.conversation_id)
```

- `query(task, skill=...)` sends `f"/{skill} {task}"`; invoke the same named command/skill in both phases.
- `follow_up` sends the reply through the already-connected SDK client and validates the supplied conversation ID. Keep one client alive for an entire scenario and close it afterward. A new scenario gets a fresh client and mock cluster.
- Collect typed SDK messages through the terminal `ResultMessage`. Pair MCP tool uses/results by ID, normalize the `mcp__rhoai-mcp__` prefix, preserve arguments and result/error content, and retain chronological message order. Built-in tool calls are not MCP calls; keep any diagnostic records separate from the normalized MCP conversation.
- Return only the current exchange in each result, allowing callers to concatenate exchanges without duplicating history. Use the terminal result for completion/error handling and collect the complete final assistant response. Missing results, execution errors, timeouts, or permission denials fail the check visibly.

Retain command discovery information from SDK initialization. Require the explicitly invoked navigator name to be available, combined with the on-disk source checks below. Neither advertising a name nor successfully calling MCP tools alone proves the workflow contract.

Add focused tests for tool/result pairing, name normalization, current-turn history, and terminal error handling using synthetic SDK messages. No judge or live Claude calls are needed for these adapter tests.

## Part 3: Simple before/after checks — primary migration gate

New runner: `evals/command_skill_check.py`, executed directly as a Python module. This avoids collecting the existing pytest evaluation suite or activating its HTTP fixtures.

Here, **deterministic** describes ordinary assertions over observed outputs and tool traces. Claude responses can vary; the checks do not compare exact prose, full transcripts, exact tool-call counts, or particular recommended model identities between phases.

### Static checks

- In `commands` phase, require the two navigator command files and require their skill replacements to be absent. In `skills` phase, require the two `SKILL.md` files and require their old navigator command files to be absent.
- Save each baseline command's body hash, description, and ordered tool list. After migration, assert identical bodies/descriptions/tool lists, valid `name` fields, and space-separated `allowed-tools` without commas.
- Save and compare the hash of `.claude/commands/scaffold-plugin.md`; it must remain unchanged and must not acquire a skill replacement.

### Workflow checks

Use two workflow scenarios: navigator selection through handoff in one SDK session, and deployment from that handoff in a second SDK session. The rows below are steps within those two scenarios, not separate sessions. Use a small fixed script of user messages, shared by both phases. Distinguish requirements confirmation, workload confirmation, model selection, and deployment approval explicitly; do not assume one confirmation satisfies every gate. Allow a short bounded exchange to complete the scripted steps, and report which assertion failed instead of silently retrying until it passes.

| Workflow step | Assertions |
|---|---|
| Navigator workload gate | Invoke `/navigator` with a support-chatbot use case, 50 users, and cost priority. Confirm requirements as needed. Require `get_use_case_defaults` and `get_expected_rps` with the expected use case/user count. Assert no `recommend_model` call occurs before the scripted workload confirmation. |
| Navigator recommendations | Confirm the workload. Require usable cluster inventory before the first recommendation call, or, if the tool is unavailable or returns unusable inventory, require a GPU question and wait for the scripted manual answer (`A100-80`). Verify no recommendations occur before that answer, and verify use case, user count, the discovered or manually supplied GPU filter, and any confirmed SLO overrides. Locate the comparison table and assert the four exact recommendation columns plus the row-label column, eight exact data rows, five cells per row, and non-empty recommendation cells with no unresolved placeholders. An explicit `—` is permitted for a slot with no recommendation. |
| Navigator handoff | Select an available recommendation from that run's actual tool results. Require a parseable `navigator-handoff` block with the documented keys and consistent model/category/workload values. Assert navigator never calls `plan_deployment` or `execute_deployment`. |
| Deployment approval gate | In a fresh SDK client, invoke `/navigator-deploy` with the captured handoff and the existing mock namespace `production-models`. Require `plan_deployment` with matching handoff values. Assert no `execute_deployment` or runtime creation occurs before the scripted deployment approval. Resolve a requested storage URI with a fixed mock URI before approving. |
| Deployment continuation | Send explicit `deploy` approval. Require `execute_deployment` with the agreed namespace/configuration and a successful deployment result. Ask for the new service's status in a follow-up and verify the tool targets that same service, proving mock state survives across turns. Accept the reported provisioning state; require endpoint lookup only when the service is Ready. |

Keep the table validator independent of DeepEval and test it with malformed/ragged tables. The existing assertion in `test_navigator_table_completeness.py` does not enforce exact cell counts, so copying it unchanged would leave missing columns undetected.

Use the real mock-cluster MCP server and embedded planner for these checks. If the baseline cannot complete a scenario, stop and report the baseline failure; it is not evidence of a migration regression and must not be hidden by loosening the after-checks.

### Saved artifacts and comparison

Each run writes one explicit JSON artifact containing:

- phase (`commands` or `skills`), static metadata/hashes, and named assertion outcomes;
- each scenario's ordered user/assistant messages, normalized MCP calls with arguments/results/errors, final outputs, turn counts, and conversation IDs;
- failure details and any partial conversation collected before failure.

The runner's `compare` operation pairs named checks, verifies the body/tool/scaffold invariants, and prints a compact before/after pass/fail table. It exits nonzero on any required failure or missing scenario. Both phases must pass the same workflow assertions. Save the complete normalized traces, not just scores or tool names, so later DeepEval scoring can reuse the evidence.

Planned commands, available after implementing this runner:

```bash
# BEFORE: navigator commands exist; their skill replacements do not.
uv run --group claude-check python -m evals.command_skill_check run \
    --phase commands --output evals/results/workflow-commands.json

# Migrate the two navigator files as described in Part 1, then run AFTER.
uv run --group claude-check python -m evals.command_skill_check run \
    --phase skills --output evals/results/workflow-skills.json

# Compare the same assertions and static invariants.
uv run --group claude-check python -m evals.command_skill_check compare \
    evals/results/workflow-commands.json evals/results/workflow-skills.json
```

The simple runner needs neither HTTP schema discovery nor an LLM judge. It makes real Claude calls and uses the user's configured Claude authentication.

### Implemented runner behavior

The navigator script has four or five user exchanges: initial requirements, explicit
requirements confirmation, explicit workload confirmation using default SLOs,
an optional manual GPU answer (`A100-80`) when navigator asks because inventory
is unavailable, and selection of a recommendation found in that run's tool results and table.
The manual branch requires evidence from SDK tool discovery or an unusable inventory
result; it cannot silently replace a skipped discovery step. The same hardware must
appear in recommendation filters and the handoff. SDK/tool execution errors still fail
the runner visibly rather than triggering retries.
Selection prefers the displayed Balanced slot, then other available slots. The
deployment script uses a fresh client and has three exchanges, or four when a
missing storage URI needs to be supplied. Storage input explicitly withholds
approval; a separate `deploy` message authorizes creation. There are no runner retries.

Artifacts use `schema_version: 1`. The root contains the script contract hash,
static source metadata, named `checks`, overall status, and failure details.
Each entry under `scenarios` retains SDK initialization metadata and ordered
`exchanges`. Each exchange contains its script label, user task, normalized
messages and MCP calls, final output, turn count, session ID, and separate SDK
diagnostics. Failed SDK exchanges retain their partial result. Subsequent checks
and scenarios stay `not_run` after an earlier failure.

`run --phase skills` validates the skill layout and frontmatter. `compare` checks
the saved body hashes, descriptions, ordered tools, and scaffold hash against the
command baseline, and requires every named check and both scenarios to have passed.
Recommendation identities and assistant prose may differ between phases.

The first live command baseline stopped because `get_cluster_resources` was absent
and navigator attempted alternative tools. Before establishing a passing baseline,
the command was updated to ask for manual GPU information when discovery is unavailable.
The script contract is now version 2; the failed original run is not a migration baseline.
The updated live command baseline passed all 11 checks, including the manual
`A100-80` exchange, handoff, deployment approval, execution, and follow-up status.
Its artifact is `evals/results/workflow-commands.json`; the original failed trace
is retained separately as `evals/results/workflow-commands-failed-missing-inventory.json`.
These artifacts are ignored by Git and remain available locally for the paired check.

The live skill run passed all 11 checks as well, using the same manual `A100-80`
exchange and separate deployment session. Its complete trace is saved in
`evals/results/workflow-skills.json`. The paired comparison passed all 31 checks;
the report is `evals/results/workflow-comparison.txt`. Both navigator bodies,
descriptions, and ordered tool lists match the command baseline, and the scaffold
command is unchanged. This completes step 6 and the migration gate for the exercised
workflows. DeepEval integration remains the separate step 7 follow-up.

Offline implementation checks (synthetic conversations, without Claude authentication):

```bash
uv run --group claude-check python -m pytest \
    tests/evals/test_claude_code_client.py tests/evals/test_command_skill_check.py
```

## Part 4: DeepEval before/after validation

Implement this after the simple gate works. The saved command and skill traces are the before/after samples: DeepEval can score them after migration without restoring the old files or rerunning their conversations. The shared wrapper also supports future live evaluation scenarios.

```text
saved commands/skills traces → conversion helpers → DeepEval metrics
                                    ↑
                      schemas from local MCP list_tools over stdio
```

### Fixtures and schema discovery

The existing `evals/conftest.py::tool_schemas` and `fetch_tool_schemas` use HTTP. For the Claude path, add stdio schema discovery with the MCP Python SDK, reusing the same server launch configuration as the wrapper and setting `StdioServerParameters.cwd` to the repository root. Schema discovery may use a short-lived separate MCP session because it reads tool definitions only. Keep HTTP discovery for the LCS path.

When wiring `--client=claude-code` into the broader pytest suite, route fixtures conditionally so it never requests the LCS health check or HTTP MCP endpoint. Any live SDK fixture must be function-scoped, opened and closed in the test's event loop, with one persistent connection per scenario. Adapt helper annotations/conversions to the shared result attributes rather than assuming the proposed client already returns DeepEval `MCPToolCall` objects. Convert structured results explicitly at this boundary.

### Dependency groups and Makefile

Keep two kinds of targets:

```makefile
# Simple primary gate: no DeepEval or judge required.
check-command-skills:
	uv run --group claude-check python -m evals.command_skill_check run --phase $(PHASE) --output $(OUTPUT)

# Later complementary scoring of the captured before/after traces.
eval-command-skills:
	uv run --group eval python -m evals.score_command_skill_traces --before $(BEFORE) --after $(AFTER) --output-dir $(OUTPUT_DIR)
```

Validate required Make variables with clear usage errors. Move the Makefile's current global Docker/Podman availability error into the targets that actually need a container runtime, so both targets above work without Docker/Podman installed. Direct Python commands remain available independently of Make.

The implemented simple target requires `PHASE=commands` or `PHASE=skills` and an
explicit `OUTPUT` path. For example:

```bash
make check-command-skills PHASE=commands OUTPUT=evals/results/workflow-commands.json
make check-command-skills PHASE=skills OUTPUT=evals/results/workflow-skills.json
```

The container runtime guard is a prerequisite of container-dependent targets.
Help, local development, reporting, and both workflow-check/scoring targets remain
available without Docker or Podman.

The implemented `score_command_skill_traces` module loads the saved artifacts and calls the shared DeepEval conversion/scoring machinery. Use `MCPUseMetric` for single-turn cases, and `MultiTurnMCPUseMetric` / `MCPTaskCompletionMetric` for complete workflow conversations. It requires judge configuration. Future live Claude pytest runs require both `--group claude-check --group eval`; the simple gate requires only `--group claude-check`.

### Implemented paired scoring

The scorer first requires passing, compatible primary-gate artifacts, then discovers
schemas through a short-lived local stdio MCP session using `evals/mcp_config.py`.
That shared launch configuration and the serializable result types are independent
of the Claude SDK. Scoring needs only `--group eval`, with no Claude authentication,
LCS service, HTTP MCP endpoint, or remote planner.

Each complete navigator/deployment conversation is scored once per phase with
`MultiTurnMCPUseMetric` and `MCPTaskCompletionMetric`. Conversion preserves user
confirmation turns and matches calls by ID. MCP content is retained, with the
original structured payload inside the `result` envelope required by DeepEval.
The older writable MCP-interaction flag and the newer computed property are both
supported. Existing LCS conversion helper names remain available.

For an OpenAI judge using an existing `OPENAI_API_KEY`:

```bash
RHOAI_EVAL_EVAL_PROVIDER=openai \
RHOAI_EVAL_EVAL_MODEL=gpt-5.4-mini \
make eval-command-skills \
    BEFORE=evals/results/workflow-commands.json \
    AFTER=evals/results/workflow-skills.json \
    OUTPUT_DIR=evals/results/migration-deepeval
```

The module also accepts `--judge-provider` and `--judge-model` overrides. Other
judge providers continue to use `EvalConfig` and `.env.eval` as before. The OpenAI
model used for this run supports structured outputs; see its
[official model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini).

The explicit output directory contains `commands.jsonl`, `skills.jsonl`,
`comparison.md`, `tool-schemas.json`, and `run.json`. Records carry backend
`claude-code`, the observed agent model, phase, and a shared scoring run ID. Older
JSONL records without a phase still load. The manifest records source artifact
hashes and the judge provider/model; it retains failures alongside any completed
JSONL records. Existing scoring outputs are rejected, so reruns need a new output
directory. Metric failures produce the paired report and a nonzero exit status;
they do not modify the primary-gate artifacts.

Step 7's live paired run used OpenAI `gpt-5.4-mini` via the existing environment
key. All eight scores passed the existing thresholds (tool use: 0.5; task
completion: 0.25). The report in `evals/results/migration-deepeval/comparison.md`
contains:

| Scenario | Metric | Commands | Skills | Delta |
|---|---|---|---|---|
| Navigator | Multi-turn MCP use | 0.627 | 0.517 | -0.110 |
| Navigator | MCP task completion | 0.733 | 0.767 | +0.033 |
| Deployment | Multi-turn MCP use | 0.637 | 0.750 | +0.113 |
| Deployment | MCP task completion | 0.750 | 0.833 | +0.083 |

The lower navigator tool-use score was investigated against both traces. The
recommendation tool names, GPU filters, user counts, and duplicate-removal
constraints match; the descriptive text differs. DeepEval 4.2.0 decomposes the
conversation into per-user-interaction tasks without carrying earlier planning
context into each task. Judge reasons criticize deliberate waits for GPU input
and approval, and request extra prerequisite/discovery tools outside the scripted
contract. These limitations affect interpretation of this one-shot score delta;
the deterministic confirmation and argument checks remain the primary gate.
No thresholds or workflow instructions were changed to improve judge scores.

The first scoring attempt failed before judge calls because DeepEval required a
`result` envelope. Its manifest and discovered schemas are preserved separately
in `evals/results/migration-deepeval-failed-result-envelope/`. Both the failure and
successful scoring outputs are ignored by Git. Step 8 (live pytest client routing)
remains separate.

Validation also passed 146 targeted offline tests, lint/format checks, and targeted
type checks. Real stdio discovery returned 53 schemas, both JSONL files work with
the existing summary CLI, and the 31 primary comparison checks still pass. The
scoring Make target rejects missing inputs and works without a container runtime.

```bash
uv run --group claude-check --group eval python -m pytest \
    tests/evals/test_claude_code_client.py tests/evals/test_command_skill_check.py \
    tests/evals/test_score_command_skill_traces.py tests/evals/providers/test_judge.py
```

### Result storage and comparison

Keep workflow-check artifacts separate from DeepEval JSONL. Do not assume `evals/results/latest.jsonl` exists: the current recorder defaults to `evals/results/eval_history.jsonl`.

For paired DeepEval scoring, pass explicit recorder paths under the supplied output directory, such as `commands.jsonl` and `skills.jsonl`. Record the actual backend (`claude-code`), phase, and run ID instead of the current hardcoded `lcs` label. Add a backward-compatible optional phase field to the reporting model so existing LCS history remains readable.

Use the existing single-file summary interface for each artifact:

```bash
uv run --group eval python -m evals.score_command_skill_traces \
    --before evals/results/workflow-commands.json \
    --after evals/results/workflow-skills.json \
    --output-dir evals/results/migration-deepeval

uv run --group eval python -m evals.reporting summary \
    --file evals/results/migration-deepeval/commands.jsonl
uv run --group eval python -m evals.reporting summary \
    --file evals/results/migration-deepeval/skills.jsonl
```

Have the scoring entrypoint produce a small paired scenario/metric table as well. The existing `evals.reporting compare` groups by backend/model and does not accept two positional files; extending that general-purpose CLI is unnecessary for this migration.

DeepEval results supplement the primary assertion gate with a judgment of tool use and task completion. Investigate differences as observed regression evidence, without claiming statistical equivalence or assuming every difference is a discovery failure.

## Part 5: Live pytest backend selection

The broader scenario suite accepts `--client=lcs` (the default) or
`--client=claude-code`. LCS keeps its configured agent, readiness check, and HTTP
MCP schema discovery. Selecting Claude requests stdio discovery and never resolves
the LCS fixture. Its function-scoped async fixture opens and closes a persistent
SDK connection in the test's own event loop, with a fresh mock MCP process for
each scenario.

The shared Claude client now accepts plain tasks as well as explicit skill
invocations. Its default navigator allowlist is preserved for the primary gate;
the broader pytest fixture pre-approves tool names discovered from its local MCP
server. No domain plugins are enabled implicitly. Scenarios that need tools outside
the active plugin set still need appropriate server configuration. Claude mode
currently supports mock clusters only and rejects a live-cluster setting.

The navigator table scenario invokes `/navigator` explicitly in Claude mode and
handles separate requirements, workload, and optional manual `A100-80` input.
LCS retains its original two-exchange conversation using its server-configured
instructions. Both paths use the exact table validator and merge their observed
exchanges through the backend-neutral result interface.

For selected live Claude scenarios with the existing OpenAI environment key:

```bash
RHOAI_EVAL_EVAL_PROVIDER=openai \
RHOAI_EVAL_EVAL_MODEL=gpt-5.4-mini \
uv run --group eval --group claude-check pytest \
    evals/scenarios/test_tool_discovery.py \
    evals/scenarios/test_navigator_table_completeness.py \
    --client=claude-code \
    --eval-results=evals/results/pytest-claude-code-step8.jsonl -v

# A single scenario can also be selected through Make:
make eval-scenario SCENARIO=tool_discovery CLIENT=claude-code
```

`--eval-results` supplies an explicit JSONL path; otherwise the existing recorder
history path is used. Live records carry the selected backend, observed agent model,
and packaging phase for explicitly invoked repository skills. Generic tasks have
no packaging phase. Make adds the Claude dependency group only when that backend
is selected. It continues to default to LCS.

Step 8 validation passed 155 targeted offline tests, lint/format checks, and targeted
type checks. Subprocess pytest tests verify conditional discovery, absence of LCS
requests in Claude mode, separate SDK clients per scenario, and opening/closing
within the test's event loop. All six scenario modules collect with the Claude
selector. LCS routing is tested offline; a live LCS stack was not started.

The two selected live pytest scenarios finished with **one pass and one failure**:

| Scenario | Check | Score | Threshold | Outcome |
|---|---|---|---|---|
| Navigator table completeness | Exact table validator | — | — | PASS |
| Navigator table completeness | Multi-turn MCP use | 0.750 | 0.500 | PASS |
| Navigator table completeness | MCP task completion | 0.600 | 0.250 | PASS |
| Tool discovery | MCP use | 0.200 | 0.500 | FAIL |

The initial discovery record contains no MCP calls. Its judge reason reports that the
answer named workbench tools absent from the active server: `list_notebook_images`,
`create_workbench`, and `get_workbench_url`. These tools exist in the `notebooks`
plugin, which the default server configuration does not enable. This is a broader
capability/answer-grounding finding, separate from the passing migration gate.
Review also identified a test-design mismatch: its fixed task requested a workbench
without checking whether the server exposed that capability.

The records are saved in `evals/results/pytest-claude-code-step8.jsonl`; a readable
summary and full judge explanations are in `evals/results/pytest-claude-code-step8.md`.
Both are ignored by Git and retain the original outcome.

The discovery scenario now derives both the request and expected setup tools from
the connected server's MCP `list_tools` inventory, the same inventory supplied to
DeepEval. It does not duplicate the local `RHOAI_MCP_ENABLED_PLUGINS` default: a
remote server, environment override, unhealthy plugin, or RBAC filter can expose a
different inventory. The default inventory requests project creation, storage, and
an S3 connection. A workbench is requested only when `create_workbench` is present.
Configurations without relevant setup tools or discovery tools skip with an
explicit reason before any agent query or judge evaluation.

The scenario also requires an actual enabled discovery-tool call, rejects calls to
unavailable tools and resource-creation calls, and checks that the answer covers
the requested enabled setup tools. The shared stdio launch configuration explicitly
forwards `RHOAI_MCP_ENABLED_PLUGINS` overrides; the MCP Python SDK otherwise inherits
only a small set of system environment variables.

Real stdio discovery verified three configurations: defaults (53 tools, three setup
capabilities), an explicit configuration with notebooks (22 tools, four setup
capabilities), and projects plus meta tools (7 tools, project setup only). Evidence
is saved in `evals/results/discovery-capability-inventory.json`.

The corrected live discovery case passed with MCP Use **0.500**, at the unchanged
threshold of **0.500**. It called `list_tool_categories` and `suggest_tools` and
passed the enabled-tool assertions. Its request differs from the original failed
case, so this is validation of a configuration-appropriate test, not a paired model
performance comparison. Records are in
`evals/results/pytest-tool-discovery-enabled-tools.jsonl`; the readable report is
`evals/results/pytest-tool-discovery-enabled-tools.md`. Server defaults, navigator
instructions, and judge thresholds are unchanged.

After the capability-aware follow-up, all 170 targeted offline tests pass,
including default/notebooks-enabled inventories, filtered server inventories,
unsupported-configuration skips, and explicit plugin-override forwarding. Changed
Python files pass lint and formatting checks; targeted type checks also pass.

## Implementation order

| Step | Work | Deliverable |
|---|---|---|
| 1 | Add the optional SDK dependency group and thin persistent-session wrapper. | `pyproject.toml`, normal lockfile update, `evals/claude_code_client.py`, focused adapter tests |
| 1a | Complete the existing embedded planner adapter: local reference tools and deployment metadata. | Local defaults/RPS/use-case support, preserved model/GPU metadata, offline integration tests; separate prerequisite commit |
| 2 | Implement the simple runner, workflow assertions, artifact persistence, and comparison. | `evals/command_skill_check.py`, validator tests |
| 3 | Make the simple target usable without a container runtime. | Scoped Makefile guard and `check-command-skills` |
| 4 | Run the command baseline; resolve any baseline/check defect before migration. | `evals/results/workflow-commands.json` |
| 5 | Move only the two navigator commands to skills; preserve bodies and normalize tool separators. | Two `SKILL.md` files; original scaffold command retained |
| 6 | Run the skill checks and compare the paired artifacts. | `evals/results/workflow-skills.json`, primary gate report |
| 7 | Later, add stdio schema discovery, trace-to-DeepEval conversion, phase-aware recording, and paired scoring. | DeepEval before/after report from the saved traces |
| 8 | Extend the broader live pytest suite to select the shared Claude wrapper when useful. | Conditional fixtures and `--client=claude-code`; existing LCS path retained |

Complete the migration when the static invariants and both workflow runs pass. DeepEval integration is retained as an explicit follow-up deliverable; it does not block the simple migration gate.

Step 1a addresses gaps found before implementing the workflow runner: the reference
tools originally rejected local mode, and local config generation omitted deployment
metadata. Use the embedded engine's SLO defaults and token profiles. Its specification
field `expected_qps` includes peak headroom and a small-workload capacity floor;
`get_expected_rps` reports this as `peak_rps` and reports average demand separately as
`expected_rps`. Offline tests disable optional planner GPU discovery and block network
connections. When running the later mock-cluster workflow checks, also disable the
planner's independent cluster discovery with `PLANNER_DETECT_CLUSTER_GPUS=false` so
recommendations use the navigator's discovered or manually supplied GPU filter.
