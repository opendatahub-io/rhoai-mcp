# MCP Evaluation Framework

This guide covers the DeepEval-based evaluation framework for testing how well LLM agents use the RHOAI MCP server's tools to accomplish real-world tasks.

## Overview

The evaluation framework measures whether an LLM agent can effectively use the MCP tools provided by the RHOAI server. Instead of checking tool implementations directly (that's what unit tests do), evals answer the question: **"Given a natural-language task, does the agent call the right tools in the right order and produce a useful result?"**

The framework uses:

- **A real LLM agent** (OpenAI, Anthropic, or Google Gemini) that receives tasks and calls MCP tools
- **The real RHOAI MCP server** running in-process with all plugins loaded
- **A mock K8s cluster** (or optionally a live cluster) providing realistic data
- **DeepEval metrics** with a judge LLM that scores the agent's tool usage and task completion

This replaces the earlier self-instrumentation approach (`ENABLE_EVALUATION` hooks) with an external, LLM-judged evaluation that better reflects real-world agent behavior.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- An API key for at least one supported LLM provider (for the agent LLM and the DeepEval judge LLM)
- (Optional) A live OpenShift cluster with RHOAI installed, for live-cluster evals

## Setup

1. Copy the example environment file and fill in your API keys:

   ```bash
   cp .env.eval.example .env.eval
   ```

   At minimum, set the provider and API key for both agent and judge:

   ```bash
   RHOAI_EVAL_LLM_API_KEY=sk-...
   RHOAI_EVAL_EVAL_API_KEY=sk-...
   ```

2. Install the eval dependency group:

   ```bash
   uv sync --group eval
   ```

## Running Evaluations

### Make targets

```bash
# Run all mock-cluster scenarios
make eval

# Run all scenarios including live-cluster tests
make eval-live

# Run a single scenario by name
make eval-scenario SCENARIO=cluster_exploration
make eval-scenario SCENARIO=training_workflow
make eval-scenario SCENARIO=model_deployment
make eval-scenario SCENARIO=troubleshooting
make eval-scenario SCENARIO=tool_discovery
```

### Direct pytest

```bash
# Mock-cluster scenarios only
uv run --group eval pytest evals/ -v -m "eval and not live" --tb=short

# All scenarios
uv run --group eval pytest evals/ -v -m "eval" --tb=short

# Single scenario file
uv run --group eval pytest evals/scenarios/test_cluster_exploration.py -v --tb=short
```

## Configuration Reference

All variables use the `RHOAI_EVAL_` prefix and can be set in `.env.eval` or as environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Agent LLM provider (see [Supported Providers](#supported-providers)) |
| `LLM_MODEL` | `gpt-4o` | Model name for the agent LLM |
| `LLM_API_KEY` | (none) | API key for the agent LLM |
| `LLM_BASE_URL` | (none) | Base URL for vLLM or Azure endpoints |
| `EVAL_PROVIDER` | `openai` | Judge LLM provider (see [Supported Providers](#supported-providers)) |
| `EVAL_MODEL` | `gpt-4o` | Model name for the DeepEval judge LLM |
| `EVAL_API_KEY` | (none) | API key for the judge LLM |
| `EVAL_MODEL_BASE_URL` | (none) | Base URL for a custom judge endpoint |
| `VERTEX_PROJECT_ID` | (none) | Google Cloud project ID (for `anthropic-vertex` and `google-vertex`) |
| `VERTEX_LOCATION` | `us-central1` | Google Cloud region (for `anthropic-vertex` and `google-vertex`) |
| `CLUSTER_MODE` | `mock` | `mock` (no cluster needed) or `live` (real cluster) |
| `MCP_USE_THRESHOLD` | `0.5` | Minimum score for MCP tool usage metrics (0.0-1.0) |
| `TASK_COMPLETION_THRESHOLD` | `0.6` | Minimum score for task completion metrics (0.0-1.0) |
| `MAX_AGENT_TURNS` | `20` | Maximum LLM turns per scenario (1-100) |

### Supported Providers

| Provider | Value | SDK | Notes |
|----------|-------|-----|-------|
| OpenAI | `openai` | `openai` | Default. Uses OpenAI API directly. |
| vLLM | `vllm` | `openai` | OpenAI-compatible. Requires `LLM_BASE_URL`. |
| Azure OpenAI | `azure` | `openai` | OpenAI-compatible. Requires `LLM_BASE_URL`. |
| Anthropic | `anthropic` | `anthropic` | Claude models via direct API. |
| Anthropic on Vertex AI | `anthropic-vertex` | `anthropic` | Claude models via Google Vertex AI. Requires `VERTEX_PROJECT_ID`. |
| Google Gemini | `google-genai` | `google-genai` | Gemini models via API key. |
| Google Gemini on Vertex AI | `google-vertex` | `google-genai` | Gemini models via Vertex AI. Requires `VERTEX_PROJECT_ID`. |

## Architecture

```text
evals/
├── config.py                         # EvalConfig (pydantic-settings)
├── conftest.py                       # Shared pytest fixtures
├── mcp_harness.py                    # In-process MCP server lifecycle
├── agent.py                          # Provider-agnostic agent loop
├── deepeval_helpers.py               # AgentResult -> DeepEval test case conversion
├── providers/
│   ├── __init__.py                   # Exports factory functions
│   ├── base.py                       # AgentLLMProvider ABC + dataclasses
│   ├── openai_provider.py            # OpenAI/Azure/vLLM provider
│   ├── anthropic_provider.py         # Anthropic/Anthropic-Vertex provider
│   ├── google_provider.py            # Google GenAI/Vertex provider
│   ├── judge.py                      # DeepEvalBaseLLM subclasses per provider
│   └── factory.py                    # create_agent_provider(), create_judge_llm()
├── reporting/
│   ├── __init__.py                   # Exports EvalRecorder, evaluate_and_record
│   ├── models.py                     # Dataclasses for JSONL schema
│   ├── recorder.py                   # EvalRecorder + evaluate_and_record wrapper
│   ├── reader.py                     # JSONL loading
│   ├── formatting.py                 # Table rendering (terminal + markdown)
│   ├── comparison.py                 # Provider comparison report
│   ├── trending.py                   # Score trend report
│   ├── cli.py                        # CLI (summary, compare, trend)
│   └── __main__.py                   # python -m entry point
├── results/
│   └── eval_history.jsonl            # Persisted eval results (gitignored)
├── mock_k8s/
│   ├── cluster_state.py              # ClusterState dataclass + default data
│   └── mock_client.py                # MockK8sClient (subclasses K8sClient)
├── metrics/
│   └── config.py                     # Metric factory functions
└── scenarios/
    ├── test_cluster_exploration.py    # Cluster discovery scenario
    ├── test_training_workflow.py      # Training job creation scenario
    ├── test_model_deployment.py       # Model serving scenario
    ├── test_troubleshooting.py        # Failed job diagnosis scenario
    └── test_tool_discovery.py         # Meta tool usage scenario
```

### How the pieces fit together

1. **`EvalConfig`** (`config.py`) loads settings from `RHOAI_EVAL_*` env vars or `.env.eval` using pydantic-settings.

2. **`MCPHarness`** (`mcp_harness.py`) starts the real RHOAI MCP server in-process. In mock mode, it injects a `MockK8sClient` before the server lifespan begins, so all domain logic, plugin loading, and tool registration execute for real — only the K8s API calls are faked. In live mode, it uses the server's normal lifespan with a real cluster connection.

3. **Provider abstraction** (`providers/`) decouples the agent loop from any specific LLM SDK. Each provider implements `AgentLLMProvider` — handling tool schema conversion, message formatting, API communication, and conversion back to OpenAI-style dicts for DeepEval. The factory function `create_agent_provider()` dispatches on the configured provider.

4. **`MCPAgent`** (`agent.py`) implements a provider-agnostic agent loop: it calls provider methods to format tools, build messages, send completions, and append results. It records all tool calls and messages in an `AgentResult`, converting messages to OpenAI format via `messages_for_deepeval()` at the end.

5. **`deepeval_helpers.py`** converts the `AgentResult` into DeepEval test case objects (`ConversationalTestCase` for multi-turn scenarios, `LLMTestCase` for single-turn), attaching the `MCPServer` tool definitions and `MCPToolCall` records.

6. **Metrics** (`metrics/config.py`) wrap DeepEval's built-in MCP metrics with configured thresholds. The `create_judge_llm()` factory creates the appropriate judge LLM based on the configured `eval_provider`.

7. **Scenarios** (`scenarios/`) are pytest test classes marked with `@pytest.mark.eval`. Each defines a natural-language `TASK`, runs the agent, builds a DeepEval test case, and asserts that all metrics pass.

### Data flow

```text
Scenario TASK ──> MCPAgent.run()
                    │
                    ├──> AgentLLMProvider.send() ──> tool_calls
                    │         (OpenAI / Anthropic / Google)
                    │                                  │
                    ├──< MCPHarness.call_tool() <──────┘
                    │       │
                    │       └──> RHOAI MCP Server ──> MockK8sClient
                    │
                    └──> AgentResult
                            │
                            ├──> deepeval_helpers ──> ConversationalTestCase
                            │                              │
                            └──> DeepEval evaluate() <─────┘
                                    │
                                    └──> Judge LLM scores metrics
```

## Available Scenarios

| Scenario | File | Task | Metrics |
|----------|------|------|---------|
| Cluster Exploration | `test_cluster_exploration.py` | Discover projects, running workbenches, and GPU availability | `MultiTurnMCPUseMetric`, `MCPTaskCompletionMetric` |
| Training Workflow | `test_training_workflow.py` | Fine-tune Llama 3.1-8B with LoRA: check prerequisites, plan resources, create the job | `MultiTurnMCPUseMetric`, `MCPTaskCompletionMetric` |
| Model Deployment | `test_model_deployment.py` | Deploy granite model via vLLM runtime and verify status | `MultiTurnMCPUseMetric`, `MCPTaskCompletionMetric` |
| Troubleshooting | `test_troubleshooting.py` | Diagnose why `failed-training-001` failed (OOMKilled) | `MultiTurnMCPUseMetric`, `MCPTaskCompletionMetric` |
| Tool Discovery | `test_tool_discovery.py` | Discover which tools to use for project setup with storage and workbench | `MCPUseMetric` (single-turn) |

## Mock Cluster State

When `CLUSTER_MODE=mock`, the `create_default_cluster_state()` function in `evals/mock_k8s/cluster_state.py` pre-populates a realistic RHOAI cluster:

| Resource Type | Name | Namespace | Details |
|---------------|------|-----------|---------|
| Namespace/Project | `ml-experiments` | — | "ML Experiments" |
| Namespace/Project | `production-models` | — | "Production Models" |
| DataScienceCluster | `default-dsc` | — | All components ready |
| AcceleratorProfile | `nvidia-a100` | — | NVIDIA A100 80GB GPU |
| Notebook (Workbench) | `my-workbench` | `ml-experiments` | Running, Minimal Python image |
| TrainJob (completed) | `llama-finetune-001` | `ml-experiments` | Llama 3.1-8B fine-tune, completed |
| TrainJob (failed) | `failed-training-001` | `ml-experiments` | OOMKilled: GPU out of memory |
| ClusterTrainingRuntime | `torchtune-llama` | — | TorchTune LLaMA runtime |
| TrainingRuntime | `custom-training-runtime` | `ml-experiments` | Custom runtime |
| InferenceService | `granite-serving` | `production-models` | Granite 3B via vLLM, ready |
| ServingRuntime | `vllm-runtime` | `production-models` | vLLM serving runtime |
| DSPA | `dspa-default` | `ml-experiments` | Pipeline server, ready |
| Secret | `aws-connection-models` | `ml-experiments` | S3 data connection |
| PVC | `workbench-storage` | `ml-experiments` | 20Gi, bound |

The `MockK8sClient` subclasses the real `K8sClient` and overrides all methods to return data from this state. This means the MCP server's domain logic runs unmodified — only the underlying K8s API calls are replaced.

## Adding a New Scenario

1. Create a new file `evals/scenarios/test_<name>.py`:

```python
"""Scenario: <Description>.

<What this scenario tests>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from evals.agent import MCPAgent
from evals.config import EvalConfig
from evals.deepeval_helpers import build_mcp_server, result_to_conversational_test_case
from evals.mcp_harness import MCPHarness
from evals.metrics.config import create_multi_turn_mcp_use_metric, create_task_completion_metric

if TYPE_CHECKING:
    from collections.abc import Callable

    from evals.agent import AgentResult


@pytest.mark.eval
class TestMyScenario:
    """Evaluate agent's ability to <do something>."""

    TASK = (
        "Natural language description of what the agent should accomplish. "
        "Be specific about resource names, namespaces, and expected actions."
    )

    @pytest.mark.eval
    async def test_my_scenario(
        self,
        eval_config: EvalConfig,
        harness: MCPHarness,
        agent: MCPAgent,
        evaluate_and_record: Callable[[str, AgentResult, list[Any], list[Any]], Any],
    ) -> None:
        """Agent should <expected behavior>."""
        result = await agent.run(self.TASK)

        # Basic sanity checks
        tool_names = result.tool_names_used
        assert len(tool_names) > 0, "Agent should call at least one tool"

        # Build DeepEval test case and evaluate
        mcp_server = build_mcp_server(harness)
        test_case = result_to_conversational_test_case(result, mcp_server)

        metrics = [
            create_multi_turn_mcp_use_metric(eval_config),
            create_task_completion_metric(eval_config),
        ]

        eval_result = evaluate_and_record(
            scenario="my_scenario",
            agent_result=result,
            test_cases=[test_case],
            metrics=metrics,
        )

        for metric_result in eval_result.test_results[0].metrics_data:
            assert metric_result.success, (
                f"Metric {metric_result.metric_name} failed: {metric_result.reason}"
            )
```

2. If the scenario needs mock data that doesn't exist yet, add resources to `create_default_cluster_state()` in `evals/mock_k8s/cluster_state.py`.

3. Run the new scenario:

```bash
make eval-scenario SCENARIO=my_scenario
```

## DeepEval Metrics

The framework uses three DeepEval metrics, created via factory functions in `evals/metrics/config.py`:

### `MCPUseMetric`

Evaluates whether the agent selected and called appropriate MCP tools for a **single-turn** interaction. The judge LLM scores tool selection against the available tool set. Used by the tool discovery scenario.

### `MultiTurnMCPUseMetric`

Like `MCPUseMetric`, but evaluates the full multi-turn conversation. It considers the sequence and combination of tool calls across turns. Used by most scenarios.

### `MCPTaskCompletionMetric`

Evaluates whether the agent actually accomplished the task based on the tool call results and final output. Checks not just that the right tools were called, but that the overall task goal was met.

All metrics accept a `threshold` (0.0-1.0) configurable via `RHOAI_EVAL_MCP_USE_THRESHOLD` and `RHOAI_EVAL_TASK_COMPLETION_THRESHOLD`.

## Using Custom LLM Providers

### OpenAI (default)

```bash
RHOAI_EVAL_LLM_PROVIDER=openai
RHOAI_EVAL_LLM_MODEL=gpt-4o
RHOAI_EVAL_LLM_API_KEY=sk-...
```

### vLLM

Set the provider to `vllm` and provide the endpoint URL:

```bash
RHOAI_EVAL_LLM_PROVIDER=vllm
RHOAI_EVAL_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
RHOAI_EVAL_LLM_API_KEY=token-placeholder
RHOAI_EVAL_LLM_BASE_URL=http://localhost:8000/v1
```

### Azure OpenAI

```bash
RHOAI_EVAL_LLM_PROVIDER=azure
RHOAI_EVAL_LLM_MODEL=gpt-4o
RHOAI_EVAL_LLM_API_KEY=your-azure-key
RHOAI_EVAL_LLM_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/gpt-4o
```

### Anthropic

```bash
RHOAI_EVAL_LLM_PROVIDER=anthropic
RHOAI_EVAL_LLM_MODEL=claude-sonnet-4-20250514
RHOAI_EVAL_LLM_API_KEY=sk-ant-...
```

### Anthropic on Vertex AI

```bash
RHOAI_EVAL_LLM_PROVIDER=anthropic-vertex
RHOAI_EVAL_LLM_MODEL=claude-sonnet-4@20250514
RHOAI_EVAL_VERTEX_PROJECT_ID=my-gcp-project
RHOAI_EVAL_VERTEX_LOCATION=us-east5
```

Authentication uses Application Default Credentials (ADC). Ensure `gcloud auth application-default login` has been run or a service account key is configured.

### Google Gemini

```bash
RHOAI_EVAL_LLM_PROVIDER=google-genai
RHOAI_EVAL_LLM_MODEL=gemini-2.0-flash
RHOAI_EVAL_LLM_API_KEY=AIza...
```

### Google Gemini on Vertex AI

```bash
RHOAI_EVAL_LLM_PROVIDER=google-vertex
RHOAI_EVAL_LLM_MODEL=gemini-2.0-flash
RHOAI_EVAL_VERTEX_PROJECT_ID=my-gcp-project
RHOAI_EVAL_VERTEX_LOCATION=us-central1
```

Authentication uses Application Default Credentials (ADC).

### Custom judge endpoint

The judge provider can be different from the agent provider. Set `EVAL_PROVIDER` to control which LLM evaluates the agent:

```bash
# Use Anthropic as the agent, OpenAI as the judge
RHOAI_EVAL_LLM_PROVIDER=anthropic
RHOAI_EVAL_LLM_MODEL=claude-sonnet-4-20250514
RHOAI_EVAL_LLM_API_KEY=sk-ant-...
RHOAI_EVAL_EVAL_PROVIDER=openai
RHOAI_EVAL_EVAL_MODEL=gpt-4o
RHOAI_EVAL_EVAL_API_KEY=sk-...
```

For self-hosted judge endpoints (vLLM, Ollama), set the base URL:

```bash
RHOAI_EVAL_EVAL_PROVIDER=vllm
RHOAI_EVAL_EVAL_MODEL=my-judge-model
RHOAI_EVAL_EVAL_API_KEY=token
RHOAI_EVAL_EVAL_MODEL_BASE_URL=http://localhost:8001/v1
```

## CI/CD

The GitHub Actions workflow (`.github/workflows/eval.yml`) runs mock-cluster evals on manual dispatch:

- **Trigger:** `workflow_dispatch` with `agent_provider`, `agent_model`, `judge_provider`, and `judge_model` inputs
- **Defaults:** `openai` provider, `gpt-4o-mini` for the agent, `gpt-4o` for the judge
- **Required secrets:** Depends on provider selection:
  - OpenAI/vLLM/Azure: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - Google: `GOOGLE_API_KEY`
  - Vertex AI: `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`
- **Output:** JUnit XML results, JSONL history, and markdown summary uploaded as artifacts

To trigger manually from the GitHub UI or CLI:

```bash
# Default (OpenAI)
gh workflow run eval.yml --field agent_model=gpt-4o --field judge_model=gpt-4o

# Anthropic agent, OpenAI judge
gh workflow run eval.yml \
  --field agent_provider=anthropic \
  --field agent_model=claude-sonnet-4-20250514 \
  --field judge_provider=openai \
  --field judge_model=gpt-4o
```

### PR Comment Trigger (`@run_evals`)

Maintainers can trigger evals directly from a pull request by commenting `@run_evals` on the PR. This provides a quick way to validate changes without navigating to the Actions tab.

**Who can trigger:** Repository owners, organization members, and collaborators (based on `author_association`).

**What happens:**

1. The workflow adds an `eyes` reaction to acknowledge the comment
2. The PR's head branch is checked out (not the default branch)
3. Evals run using `google-genai` / `gemini-2.0-flash` for both the agent and judge LLMs
4. Results are posted as a PR comment with the markdown summary table
5. A `rocket` reaction is added on success, or `thumbsdown` on failure

**Required secret:** `GOOGLE_API_KEY` (Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)).

**Default configuration for `@run_evals`:**

| Setting | Value |
|---------|-------|
| Agent provider | `google-genai` |
| Agent model | `gemini-2.0-flash` |
| Judge provider | `google-genai` |
| Judge model | `gemini-2.0-flash` |
| Cluster mode | `mock` |

The `workflow_dispatch` trigger remains available for full provider/model flexibility.

### CI Result Persistence

Eval results are persisted across CI runs using GitHub Actions cache:

1. **Before evals run**, the workflow restores `evals/results/eval_history.jsonl` from cache using the key pattern `eval-results-{branch}-{run_id}`, falling back to `eval-results-{branch}-` (latest from the same branch), then `eval-results-main-` (baseline from main).

2. **During evals**, each scenario appends a JSONL record to the history file automatically via the `evaluate_and_record` fixture.

3. **After evals**, the workflow generates a summary report and score trend table, posting both to the GitHub Actions step summary. The updated JSONL file is saved back to cache via the `actions/cache` post-action.

This means PR runs can see and compare against results from previous `main` branch runs, making regressions immediately visible in the step summary.

## Result Reporting

Eval results are automatically recorded to `evals/results/eval_history.jsonl` during each run. Each scenario produces one JSONL line containing the run ID, git metadata, environment config, metric scores, and timing data.

### Viewing Results

Three Make targets provide terminal reports:

```bash
# Summary of the latest eval run (all scenarios in a table)
make eval-report

# Compare scores across different providers/models
make eval-compare

# Show score trends over time
make eval-trend
```

### CLI Options

The reporting CLI supports additional filtering and formatting:

```bash
# Summary for a specific run ID
uv run --group eval python -m evals.reporting.cli summary --run-id a1b2c3d4e5f6

# Compare a specific scenario across providers
uv run --group eval python -m evals.reporting.cli compare --scenario cluster_exploration

# Trend for a specific provider, last 5 records, markdown output
uv run --group eval python -m evals.reporting.cli trend --provider openai/gpt-4o --last 5 --format markdown
```

### JSONL Record Format

Each line in `eval_history.jsonl` is a self-contained JSON object:

```json
{
  "run_id": "a1b2c3d4e5f6",
  "timestamp": "2026-02-18T14:30:00+00:00",
  "scenario": "cluster_exploration",
  "git": {"commit": "b9d6777", "branch": "main"},
  "environment": {
    "llm_provider": "openai", "llm_model": "gpt-4o",
    "eval_provider": "openai", "eval_model": "gpt-4o",
    "cluster_mode": "mock",
    "mcp_use_threshold": 0.5, "task_completion_threshold": 0.6,
    "max_agent_turns": 20
  },
  "metrics": [
    {"name": "MultiTurnMCPUseMetric", "score": 0.85, "success": true, "threshold": 0.5, "reason": "..."}
  ],
  "turns": 5,
  "tool_names_used": ["list_projects", "list_workbenches"],
  "passed": true,
  "duration_seconds": 12.3
}
```

The file is append-only and gitignored locally. No external dependencies are required for reporting — all formatting uses stdlib only.

## Troubleshooting

### Missing API key

```
openai.AuthenticationError: Error code: 401
```

Ensure `RHOAI_EVAL_LLM_API_KEY` and `RHOAI_EVAL_EVAL_API_KEY` are set in `.env.eval` or the environment. For Anthropic, the key should start with `sk-ant-`. For Google, use a Gemini API key.

### Mock client errors

```
NotFoundError: <resource type> '<name>' not found
```

The agent asked for a resource that doesn't exist in the mock cluster state. If this is expected for your scenario, add the resource to `create_default_cluster_state()` in `evals/mock_k8s/cluster_state.py`.

### Agent reaches max turns

```
Agent reached maximum turns (20) without completing the task.
```

The agent couldn't finish within the turn limit. Try increasing `RHOAI_EVAL_MAX_AGENT_TURNS` or simplifying the task. This may also indicate the agent is stuck in a loop calling the same tools repeatedly.

### Metric failures

```
Metric MCPTaskCompletionMetric failed: <reason>
```

The judge LLM determined the agent didn't complete the task successfully. Check the `reason` field for details. You can lower the threshold temporarily to see partial scores:

```bash
RHOAI_EVAL_TASK_COMPLETION_THRESHOLD=0.3 make eval
```

### vLLM connection errors

```
openai.APIConnectionError: Connection error.
```

Verify your vLLM endpoint is running and accessible at the URL specified in `RHOAI_EVAL_LLM_BASE_URL`. The URL should include `/v1` (e.g., `http://localhost:8000/v1`).

### Vertex AI authentication

For `anthropic-vertex` and `google-vertex` providers, authentication uses Google Cloud Application Default Credentials. Ensure you have authenticated:

```bash
gcloud auth application-default login
```

Or set a service account key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Verbose output

For more detail on what the agent is doing, enable debug logging:

```bash
RHOAI_EVAL_LLM_MODEL=gpt-4o uv run --group eval pytest evals/ -v -m "eval and not live" --tb=long -s --log-cli-level=DEBUG
```
