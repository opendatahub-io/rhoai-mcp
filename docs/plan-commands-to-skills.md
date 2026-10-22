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

Offline implementation checks (synthetic conversations, without Claude authentication):

```bash
uv run --group claude-check python -m pytest \
    tests/evals/test_claude_code_client.py tests/evals/test_command_skill_check.py
```

## Part 4: Keep DeepEval before/after validation as a later step

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
Help, local development, reporting, and the simple workflow-check target remain
available without Docker or Podman. The DeepEval scoring target is still a later step.

The planned `score_command_skill_traces` module loads the saved artifacts and calls the existing DeepEval conversion/scoring machinery. Use `MCPUseMetric` for single-turn cases, and `MultiTurnMCPUseMetric` / `MCPTaskCompletionMetric` for complete workflow conversations. It requires the existing judge configuration. Future live Claude pytest runs require both `--group claude-check --group eval`; the simple gate requires only `--group claude-check`.

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
the mock cluster remains the source of GPU inventory.
