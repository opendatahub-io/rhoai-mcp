# Plan: Convert Commands to Skills + DeepEval Validation via Claude Code

## Goal

1. Migrate 3 `.claude/commands/` to agentskills.io `SKILL.md` format
2. Build a `ClaudeCodeClient` so DeepEval evals run through Claude Code directly (no docker-compose)
3. Run DeepEval before (commands) and after (skills) to verify behavioral equivalence

---

Notice:
> Custom commands have been merged into skills. A file at .claude/commands/deploy.md and a skill at .claude/skills/deploy/SKILL.md both create /deploy and work the same way. Your existing .claude/commands/ files keep working. Skills add optional features: a directory for supporting files, frontmatter to control whether you or Claude invokes them, and the ability for Claude to load them automatically when relevant.
from https://code.claude.com/docs/en/skills

So a Skill is a functional equivalent of a (old) Command.

---

## Part 1: Migration — Commands to Skills

### Current state

```
.claude/commands/
├── navigator.md          (289 lines, 6 allowed-tools)
├── navigator-deploy.md   (372 lines, 14 allowed-tools)
└── scaffold-plugin.md    (793 lines, no allowed-tools)
```

### Target state (agentskills.io format)

```
.claude/skills/
├── navigator/
│   └── SKILL.md
├── navigator-deploy/
│   └── SKILL.md
└── scaffold-plugin/
    └── SKILL.md
```

### Frontmatter mapping

| Command field | agentskills.io field | Change needed |
|---|---|---|
| (filename) | `name` | Add `name:` matching directory name |
| `description` | `description` | Keep as-is |
| `allowed-tools` | `allowed-tools` | Keep as-is (in spec, experimental) |

Body content (markdown instructions) requires **no changes** — both formats use free-form markdown.

### Example: navigator.md → navigator/SKILL.md

**Before** (`.claude/commands/navigator.md`):
```yaml
---
allowed-tools: mcp__rhoai-mcp__recommend_model ...
description: Guides customers through LLM model selection for Red Hat OpenShift AI
---
```

**After** (`.claude/skills/navigator/SKILL.md`):
```yaml
---
name: navigator
description: Guides customers through LLM model selection for Red Hat OpenShift AI
allowed-tools: mcp__rhoai-mcp__recommend_model ...
---
```

### Migration steps

1. Create `.claude/skills/{navigator,navigator-deploy,scaffold-plugin}/` directories
2. For each command: copy to `SKILL.md`, add `name:` field to frontmatter
3. Verify skills appear in Claude Code (`/navigator` etc. still work)
4. Remove `.claude/commands/` after validation passes

---

## Part 2: DeepEval via Claude Code (no docker-compose)

### Why

The current eval suite flows through lightspeed-stack → llama-stack → rhoai-mcp. This:
- Requires docker-compose with 3 containers
- Does **not** test Claude Code commands/skills (Claude Code is not in the loop)
- Cannot detect behavioral differences from the commands→skills migration

### New architecture

```
pytest → ClaudeCodeClient → claude -p (subprocess) → Claude Code + skill
                                                          ↓
                                              rhoai-mcp (stdio, auto-managed)
                                                          ↓
                                              parse stream-json output
                                                          ↓
                                              DeepEval MCPToolCall / MCPServer
                                                          ↓
                                              MCPUseMetric / MCPTaskCompletionMetric
```

Claude Code launches rhoai-mcp as a stdio subprocess via MCP config — no separate server process to manage.

**Dependencies**: only a judge LLM API key (Gemini/OpenAI/Anthropic). No containers, no background processes.

### ClaudeCodeClient design

New file: `evals/claude_code_client.py` (~100-150 lines)

Analogous to the existing `LCSClient`, but runs `claude -p` instead of calling LCS HTTP:

```python
@dataclass
class ClaudeCodeResult:
    """Result of a Claude Code invocation, parallel to LCSResult."""
    task: str
    final_output: str
    tool_calls: list[MCPToolCall]     # from deepeval.test_case
    messages: list[dict[str, Any]]    # raw stream-json events
    turns: int = 0
    session_id: str = ""

class ClaudeCodeClient:
    """Runs claude -p and parses stream-json output into DeepEval-compatible results."""

    def __init__(self, mcp_config_path: str, working_dir: str, ...):
        ...

    async def query(self, task: str, skill: str | None = None) -> ClaudeCodeResult:
        """Run a prompt (optionally prefixed with /skill) through Claude Code."""
        prompt = f"/skill {skill} {task}" if skill else task
        # Runs: claude -p <prompt> --output-format stream-json --verbose
        #       --mcp-config <path> --permission-mode <mode>
        # Parses JSONL stream:
        #   - type=assistant, content[].type=tool_use → tool call
        #   - type=user, content[].type=tool_result → tool result
        #   - last assistant text block → final_output
        # Pairs tool_use → tool_result by id to build MCPToolCall(name, args, result)
        ...

    async def follow_up(self, message: str, session_id: str) -> ClaudeCodeResult:
        """Continue a conversation (for multi-turn scenarios)."""
        # Runs: claude -p <message> --resume <session_id> --output-format stream-json --verbose
        ...
```

### conftest.py changes

Add a `claude_code_client` fixture alongside the existing `lcs_client`:

```python
@pytest.fixture(scope="session")
def claude_code_client(eval_config):
    """ClaudeCodeClient — Claude Code manages rhoai-mcp via stdio internally."""
    return ClaudeCodeClient(
        mcp_config_path="evals/config/claude-code-mcp.json",
        working_dir=str(Path.cwd()),
    )
```

New MCP config file (`evals/config/claude-code-mcp.json`):
```json
{
  "mcpServers": {
    "rhoai-mcp": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "rhoai-mcp", "--mock-cluster"]
    }
  }
}
```

Claude Code manages the rhoai-mcp process lifecycle via stdio — it starts automatically when the session begins and stops when it ends. No background process management needed.

### Reusing existing test scenarios

The existing scenario tests can be adapted with minimal changes. The key substitution is `lcs_client` → `claude_code_client`:

```python
# Before (LCS path):
result = await lcs_client.query(self.TASK)

# After (Claude Code path):
result = await claude_code_client.query(self.TASK)
# or, to test with a specific skill:
result = await claude_code_client.query(self.TASK, skill="navigator")
```

The DeepEval metric scoring is identical — both paths produce `MCPToolCall` objects and feed them into the same `MCPUseMetric` / `MCPTaskCompletionMetric`.

### Makefile targets

```makefile
# Run evals through Claude Code (no containers, no background processes)
# Claude Code starts/stops rhoai-mcp automatically via stdio MCP config
eval-claude-code:
	uv run pytest evals/ -v -m "eval and not live" --client=claude-code

eval-claude-code-scenario:
	uv run pytest evals/scenarios/test_$(SCENARIO).py -v -m eval --client=claude-code
```

---

## Part 3: Before/After Evaluation Protocol

### Prerequisites

- rhoai-mcp runs locally with `--mock-cluster`
- A judge LLM API key configured (e.g., `GEMINI_API_KEY` or `OPENAI_API_KEY`)
- Claude Code installed and authenticated

### Step 1: Baseline — Run with Commands

Commands are still in `.claude/commands/`. Run the eval suite through Claude Code:

```bash
# Run all scenarios (rhoai-mcp starts/stops automatically via stdio)
uv run pytest evals/ -v -m "eval and not live" --client=claude-code

# Save results
cp evals/results/latest.jsonl evals/results/baseline-commands.jsonl
```

### Step 2: Migrate Commands to Skills

Execute the migration (Part 1 steps).

### Step 3: Validation — Run with Skills

Skills are now in `.claude/skills/`. Same eval suite, same metrics:

```bash
# Run all scenarios (same command — only .claude/ layout changed)
uv run pytest evals/ -v -m "eval and not live" --client=claude-code

# Save results
cp evals/results/latest.jsonl evals/results/baseline-skills.jsonl
```

### Step 4: Compare

```bash
uv run -m evals.reporting compare \
    evals/results/baseline-commands.jsonl \
    evals/results/baseline-skills.jsonl
```

**Expected outcome**: scores should be statistically equivalent. The migration is structural (frontmatter + directory layout), so skill behavior should be identical to command behavior. Any regression indicates an issue with skill discovery or loading, not content.

### Scenarios to prioritize

| Scenario | Tests | Why it matters for this migration |
|---|---|---|
| `navigator_table_completeness` | Multi-turn navigator flow | Directly exercises the navigator command/skill |
| `model_deployment` | Deployment tools | Exercises navigator-deploy adjacent tools |
| `tool_discovery` | Tool listing/description | Verifies MCP tools are still discoverable |
| `cluster_exploration` | Cluster overview | General regression check |

`scaffold-plugin` is interactive/generative (creates files) — harder to score with DeepEval metrics. Validate manually.

---

## Implementation Order

| # | Task | Effort | Deliverable |
|---|---|---|---|
| 1 | Build `ClaudeCodeClient` | ~150 lines | `evals/claude_code_client.py` |
| 2 | Add conftest fixture + MCP config | ~30 lines | `evals/conftest.py`, `evals/config/claude-code-mcp.json` |
| 3 | Add `--client` pytest option to switch between LCS and Claude Code | ~20 lines | `evals/conftest.py` |
| 4 | Add Makefile targets | ~10 lines | `Makefile` |
| 5 | Run baseline eval (commands) | ~10 min | `evals/results/baseline-commands.jsonl` |
| 6 | Migrate commands → skills | ~30 min | `.claude/skills/*/SKILL.md` |
| 7 | Run validation eval (skills) | ~10 min | `evals/results/baseline-skills.jsonl` |
| 8 | Compare and document results | ~5 min | comparison report |

### What this does NOT replace

The existing LCS-based eval path (`make eval`) remains for testing the full stack (LCS + llama-stack + rhoai-mcp). The new Claude Code path is complementary — it tests the Claude Code agent experience directly.
