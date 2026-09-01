# RHOAI MCP - fpj-ai-skills Branch Overview

## Summary

The `fpj-ai-skills` branch implements a comprehensive **Recommendation-to-Deployment Workflow** for the RHOAI MCP Server, enabling AI agents to get model recommendations from the llm-d-planner backend and deploy them end-to-end with cluster-aware GPU validation.

**Branch Statistics:**
- Files changed: 10
- Lines added: 1,035
- Lines removed: 17
- Net change: +1,018 lines

## What Was Built

### Core Objective
Enable a seamless three-tool workflow that bridges the llm-d-planner's model recommendation engine with RHOAI's inference deployment capabilities, including intelligent GPU availability checking and deployment planning.

### Architecture Overview

```
User Intent → recommend_model() → plan_deployment() → execute_deployment()
                   ↓                      ↓                     ↓
          Planner API + GPU Cross-ref   Pre-flight Checks   Create ISVC
```

## Key Components Implemented

### 1. Planner Models (`src/rhoai_mcp/composites/planner/models.py`)
**Purpose:** Comprehensive Pydantic models for the recommendation-to-deployment workflow.

**Added 340 lines of model definitions:**

#### Request Models
- `DeploymentIntent` - Natural language intent parsing (use case, scale, GPU preferences, quality/cost/latency priorities)
- `GpuPreference` - GPU type and optional count constraints
- `DeploymentSpecification` - Complete spec with SLO targets, workload profile, and priority weights

#### Planner Response Models
- `ModelRecommendation` - Individual recommendation with GPU config, latency predictions, costs, SLO compliance
- `RecommendationResult` - Complete planner output with top 4 recommendations (performance, cost, balanced, quality)
- `GPUConfig` - GPU configuration including type, count, tensor parallelism, replicas
- `SLOTargets` - TTFT, ITL, E2E latency targets with percentile ranges

#### Cluster-Aware Models
- `ClusterGPU` - GPU types available on the cluster (product, total, available, node count)
- `ClusterFitResult` - Result of checking a recommendation against cluster GPU availability
  - Status: `available` | `partial` | `unavailable`
  - Shows needed vs. available GPUs with human-readable explanation

#### Deployment Planning Models
- `DeploymentPlanIssue` - Issues found during planning (category, blocking status, suggestion)
- `ResolvedDeployParams` - Fully resolved parameters for `deploy_model` (name, namespace, runtime, resources)
- `DeploymentPlan` - Reviewable plan with steps, issues, and warnings

#### Deployment Execution Models
- `EndpointValidation` - Endpoint health check result (reachability, response time, status)
- `DeploymentResult` - Final result with success status, endpoint URL, SLO comparison

### 2. Planner Tools (`src/rhoai_mcp/composites/planner/tools.py`)
**Purpose:** Three MCP tools that implement the recommendation-to-deployment workflow.

**Added 280 lines of tool implementations:**

#### Tool 1: `recommend_model()`
Gets model recommendations from llm-d-planner with optional cluster GPU cross-reference.

**Parameters:**
- `use_case` - One of 9 predefined use cases (chatbot, code completion, translation, etc.)
- `expected_qps` - Expected queries per second
- `prompt_tokens`, `output_tokens` - Expected token lengths
- `category` - Model category (llama, mistral, etc.)
- `optimization_profile` - Quality/cost/latency balance (balanced, optimize_latency, etc.)
- `preferred_gpu_types` - GPU preference list
- `check_cluster_fit` - Whether to cross-reference cluster GPU availability

**Implementation:**
- Calls planner backend API with specification
- Extracts top 4 recommendations (performance, cost, balanced, quality)
- When `check_cluster_fit=true`, performs GPU availability checking:
  - Queries cluster for available GPUs via `TrainingClient`
  - Matches recommendation GPU types against cluster products
  - Includes fuzzy matching for GPU aliases (e.g., "A100-80" matches "NVIDIA-A100-SXM4-80GB")
  - Returns fit status for each recommendation

**Helper Functions:**
- `_check_cluster_gpu_fit()` - Cross-reference recommendations vs. cluster GPUs
- `_extract_gpu_type()` - Parse GPU type from formatted string (e.g., "4x H100")
- `_extract_gpu_count()` - Parse GPU count from formatted string
- `_match_gpu_on_cluster()` - Fuzzy matching between planner GPU types and cluster products

#### Tool 2: `plan_deployment()`
Resolves deployment parameters from a recommendation and generates a reviewable plan.

**Parameters:**
- `recommendation_json` - Recommendation from `recommend_model()` (or custom JSON)
- `namespace` - Target Kubernetes namespace
- `override_storage_uri` - Optional custom model storage location
- `override_replicas` - Optional custom replica count

**Implementation:**
- Validates namespace exists in cluster
- Resolves serving runtime (auto-selects vLLM/TGIS for LLMs)
- Validates GPU availability for the recommendation
- Generates DNS-safe deployment name from model ID
- Constructs full deployment parameters with computed resources
- Identifies issues and warnings (blocking and non-blocking)
- Returns a `DeploymentPlan` ready for review

**Helper Functions:**
- `_estimate_model_resources()` - Estimate GPU/memory needs from model size
- `_generate_deployment_name()` - Convert HuggingFace model ID to DNS-1123 name
- `_find_serving_runtime()` - Select appropriate runtime (vLLM, TGIS, OVMS, etc.)
- `_resolve_storage_uri()` - Validate and normalize model storage path
- `_validate_namespace()` - Check namespace exists and user has permission

#### Tool 3: `execute_deployment()`
Creates the InferenceService, waits for readiness, and validates the endpoint.

**Parameters:**
- `plan_json` - Deployment plan from `plan_deployment()`
- `wait_for_ready` - Whether to wait for InferenceService to become ready (default: true)
- `validation_timeout_seconds` - Timeout for endpoint validation (default: 120)

**Implementation:**
- Executes each step in the plan
- Creates InferenceService with resolved parameters
- Polls for Ready condition (with configurable timeout)
- Tests the endpoint with a sample request
- Compares actual performance against SLO predictions
- Returns `DeploymentResult` with endpoint URL and validation results

### 3. Supporting Infrastructure

#### Planner Client (`src/rhoai_mcp/composites/planner/client.py`)
HTTP client for the llm-d-planner backend. Handles:
- API communication with error handling
- Response parsing and validation
- Timeout management

**Configuration via environment:**
```
RHOAI_MCP_PLANNER_URL=http://planner-backend.example.com
RHOAI_MCP_PLANNER_TIMEOUT=30
```

#### Planner Registry (`src/rhoai_mcp/composites/registry.py`)
Updates to register the planner composite as a plugin.

#### Documentation Updates
- **ARCHITECTURE.md** - Added planner composite section (220 lines total, 37+ new)
- **README.md** - Updated feature list and docs
- **EVALS.md** - Updated evaluation documentation

### 4. Testing Infrastructure

#### Unit Tests (`tests/composites/planner/test_models.py`)
**Added 230 lines of tests:**
- Model serialization/deserialization
- Pydantic validation
- Edge cases (None values, empty lists, boundary values)
- Tests for all 15+ model types

#### Integration Tests (`tests/composites/planner/test_tools.py`)
**Added 317 lines of tests:**
- `recommend_model()` - API call, response parsing, GPU matching
- `plan_deployment()` - Plan generation, validation, issue detection
- `execute_deployment()` - ISVC creation, readiness polling, endpoint validation
- Mock cluster scenarios (GPUs available, partial, unavailable)
- Error handling (network errors, invalid parameters, cluster issues)

#### Tool Registration Tests (`tests/composites/planner/test_plugin.py`)
**Added 16 lines of tests:**
- Plugin metadata validation
- Tool registration verification
- Prompt registration verification

## Key Features Delivered

### Cluster-Aware GPU Validation
- Queries actual cluster GPU availability
- Fuzzy matching between planner GPU types (e.g., "A100-80") and cluster product labels (e.g., "NVIDIA-A100-SXM4-80GB")
- Three-state feedback: `available` | `partial` | `unavailable`
- Returns specific reasons (e.g., "Need 4x H100 but only 2 available")

### Deployment Planning & Pre-flight Checks
- Validates namespace exists
- Checks GPU availability
- Verifies runtime compatibility
- Checks storage accessibility
- Identifies issues vs. warnings (blocking vs. non-blocking)
- Generates ordered execution steps

### End-to-End Deployment
- Creates InferenceService with optimized parameters
- Polls for Ready condition with timeout
- Tests endpoint with sample request
- Compares actual vs. predicted SLO performance
- Returns endpoint URL for immediate use

### Domain Expertise
- **Use Case Support:** 9 predefined use cases (chatbot, code completion, RAG, summarization, translation, etc.)
- **Optimization Profiles:** 4 profiles (balanced, optimize_latency, optimize_cost, optimize_quality)
- **GPU Support:** 6 GPU types (L4, A100-40GB, A100-80GB, H100, H200, B200)
- **Latency Metrics:** TTFT, ITL, E2E targets with percentile ranges

### AI Agent Workflow Integration
The workflow is designed for multi-agent orchestration:

```
User Request
    ↓
suggest_tools() or recommend-and-deploy prompt
    ↓
recommend_model(use_case=..., qps=..., check_cluster_fit=true)
    ↓
[Agent reviews recommendations with GPU fit status]
    ↓
plan_deployment(recommendation_json=..., namespace=...)
    ↓
[Agent reviews deployment plan and issues]
    ↓
execute_deployment(plan_json=...)
    ↓
[Agent gets endpoint URL and SLO comparison]
```

## Files Modified

| File | Lines | Type | Purpose |
|------|-------|------|---------|
| [src/rhoai_mcp/composites/planner/models.py](src/rhoai_mcp/composites/planner/models.py) | +340, -0 | New | Pydantic models for recommendation/deployment workflow |
| [src/rhoai_mcp/composites/planner/tools.py](src/rhoai_mcp/composites/planner/tools.py) | +280, -7 | Modified | Three MCP tools with GPU validation and planning |
| [src/rhoai_mcp/composites/meta/tools.py](src/rhoai_mcp/composites/meta/tools.py) | +24, -8 | Modified | Updated tool suggestion patterns for planner |
| [src/rhoai_mcp/composites/registry.py](src/rhoai_mcp/composites/registry.py) | +17, -2 | Modified | Register planner composite plugin |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | +37, -0 | Modified | Added planner composite documentation |
| [tests/composites/planner/test_models.py](tests/composites/planner/test_models.py) | +230, -0 | New | Model validation tests |
| [tests/composites/planner/test_tools.py](tests/composites/planner/test_tools.py) | +317, -0 | Modified | Tool and workflow integration tests |
| [tests/composites/planner/test_plugin.py](tests/composites/planner/test_plugin.py) | +16, -0 | Modified | Plugin registration tests |
| [docs/EVALS.md](docs/EVALS.md) | +9, -1 | Modified | Evaluation documentation |
| [README.md](README.md) | +3, -1 | Modified | Updated feature list |

## Design Patterns & Best Practices

### 1. Stateless Token Flow
Each tool output is a complete, self-contained result. No server state is required between tool calls.

### 2. Lazy Imports
Domain clients are imported inside functions to avoid circular dependencies:
```python
def _check_cluster_gpu_fit(...):
    from rhoai_mcp.domains.training.client import TrainingClient
    training_client = TrainingClient(server.k8s)
```

### 3. Error Returns
Tools return error dictionaries instead of raising exceptions, keeping the workflow transparent to agents:
```python
if not issues:
    return {"error": "Issue found", "message": "...", "blocking": True}
```

### 4. GPU Matching with Aliases
Fuzzy matching handles variations in GPU naming:
```python
_GPU_NAME_ALIASES = {
    "a100-80": ["a100", "80gb", "a100-sxm"],
    "h100": ["h100"],
    ...
}
```

### 5. Comprehensive Validation
Pre-flight checks catch issues before deployment:
- Namespace existence
- GPU availability
- Runtime compatibility
- Storage accessibility

## Configuration

The planner workflow is configured via environment variables:

```bash
# Planner backend URL
export RHOAI_MCP_PLANNER_URL=http://planner-backend.example.com

# Planner API timeout
export RHOAI_MCP_PLANNER_TIMEOUT=30

# Deployment timeouts
export RHOAI_MCP_DEPLOYMENT_WAIT_TIMEOUT=600  # 10 minutes
export RHOAI_MCP_ENDPOINT_VALIDATION_TIMEOUT=120  # 2 minutes
```

## Testing the Workflow

### Unit Tests
```bash
cd rhoai-mcp
uv run pytest tests/composites/planner/test_models.py -v
uv run pytest tests/composites/planner/test_plugin.py -v
```

### Integration Tests
```bash
uv run pytest tests/composites/planner/test_tools.py -v
```

### Full Test Suite
```bash
make test
```

## Next Steps / Potential Enhancements

1. **Advanced GPU Allocation:** Support tensor parallelism configuration and multi-GPU strategies
2. **Cost Optimization:** Implement cost prediction and budget constraints
3. **A/B Testing Support:** Deploy multiple models for comparison
4. **Auto-scaling:** Configure HPA-based scaling based on predicted workload
5. **Model Registry Integration:** Fetch latest models from Model Registry at deployment time
6. **Prompt Workflow:** `recommend-and-deploy` prompt for guided agent workflows

## Related Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Planner composite section
- [CLAUDE.md](CLAUDE.md) - Development principles and commands
- [README.md](README.md) - Feature overview and configuration

---

**Branch:** `fpj-ai-skills`  
**Status:** Feature complete with comprehensive testing  
**Commits:** Includes dependency updates and CI/CD improvements (79 total on branch since base)
