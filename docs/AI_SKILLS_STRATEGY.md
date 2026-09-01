# AI Skills Strategy: Reusable Workflows for Model Recommendation & Deployment

## Overview

This document outlines a recommended approach for building reusable AI skills that guide customers through the process of model recommendation and deployment using the RHOAI MCP Server.

## Problem Statement

Customers need to:
1. Get recommendations for which LLM model to deploy
2. Validate that their cluster has sufficient GPU resources
3. Plan the deployment with appropriate configuration
4. Execute the deployment safely
5. Validate the endpoint is working correctly

Currently, these are separate MCP tools. Customers may not know the correct sequence or how to handle errors/decisions at each step.

**Goal:** Provide comprehensive, reusable guidance through this entire workflow that customers can leverage and customize for their specific clusters.

---

## Recommended Strategy: MCP Prompts + Guided Workflows

### Core Principle
Build **MCP Prompts** that orchestrate the workflow, paired with **discoverable guidance tools** and **well-documented templates**. This is the most portable and customer-friendly approach.

---

## 1. Prompt-Driven Orchestration (Primary)

Create two interconnected prompts in `src/rhoai_mcp/domains/prompts/deployment_prompts.py`:

### Prompt 1: `recommend-and-deploy`

**Purpose:** End-to-end workflow for getting recommendations and deploying to cluster.

**Signature:**
```python
@mcp.prompt(
    name="recommend-and-deploy",
    description="End-to-end workflow: get model recommendations and deploy to your cluster",
)
def recommend_and_deploy(
    use_case: str,
    namespace: str,
    qps: float,
    user_count: int = 10,
    preferred_gpu_types: list[str] | None = None,
    optimization_profile: str = "balanced"
) -> str:
```

**Template:**
```markdown
I need to recommend and deploy an LLM for {use_case}.

**Deployment Requirements:**
- Namespace: {namespace}
- Expected Load: {qps} queries/second, ~{user_count} users
- Use Case: {use_case}
- GPU Preference: {preferred_gpu_types or "any"}
- Optimization Profile: {optimization_profile}

**Please help me complete these steps:**

1. **Get Recommendations** (`recommend_model`)
   - Call with:
     - use_case: "{use_case}"
     - expected_qps: {qps}
     - preferred_gpu_types: {preferred_gpu_types}
     - check_cluster_fit: true (validates GPU availability)
   - Review the 4 options (performance, cost, balanced, quality)
   - Note which show "available" GPU status

2. **Plan Deployment** (`plan_deployment`)
   - Select a recommendation
   - Call with recommendation_json + namespace: "{namespace}"
   - Review the plan:
     - **Blocking issues** = stop and resolve
     - **Warnings** = non-critical but important
     - GPU/compute resources and estimated costs

3. **Execute Deployment** (`execute_deployment`)
   - Call with plan_json from step 2
   - Wait for InferenceService to become Ready
   - Receive endpoint URL

4. **Validate & Monitor**
   - Use `test_model_endpoint` to verify it works
   - Compare predicted vs. actual SLO metrics
   - Scale with `scale_deployment` if needed

**Decision Points:**
- If "gpu_unavailable": Consider a smaller model or request GPU allocation
- If "deployment_failed": Check the issues list for specific blockers
- If "endpoint_unreachable": Check namespace, RBAC, and networking

**For Production:**
- Set up monitoring on the endpoint URL
- Configure autoscaling based on expected load
- Test with representative workload samples
```

### Prompt 2: `diagnose-deployment-issues`

**Purpose:** Troubleshooting guide when things go wrong.

**Signature:**
```python
@mcp.prompt(
    name="diagnose-deployment-issues",
    description="Troubleshoot and fix common deployment problems",
)
def diagnose_deployment_issues(
    issue_type: str,  # "gpu_unavailable", "endpoint_unreachable", "slow_performance", "oom", etc.
    namespace: str,
    deployment_name: str | None = None
) -> str:
```

---

## 2. Guided Decision Trees (Supporting)

Enhance the `suggest_tools()` composite to help customers **choose the right workflow**:

**Location:** `src/rhoai_mcp/composites/meta/tools.py`

**Implementation:**

```python
DECISION_TREES = {
    "deploy_an_llm": {
        "intent": "I want to deploy an LLM for production",
        "workflow": "recommend-and-deploy",
        "tools": ["recommend_model", "plan_deployment", "execute_deployment"],
        "estimated_time": "10-15 minutes",
        "prerequisites": [
            "Know your use case (chatbot, RAG, code completion, etc.)",
            "Know your expected load (queries/second, user count)",
            "Have a Kubernetes namespace ready",
        ],
        "steps": [
            "Define use case and load profile",
            "Get recommendations (checks GPU availability)",
            "Review recommendations with GPU fit status",
            "Plan deployment with pre-flight checks",
            "Execute deployment and validate endpoint",
        ],
        "prompts": ["recommend-and-deploy"]
    },
    
    "deploy_known_model": {
        "intent": "I have a specific model, just deploy it",
        "workflow": "direct-deployment",
        "tools": ["prepare_model_deployment", "deploy_model"],
        "estimated_time": "5 minutes",
        "prerequisites": [
            "Model ID or storage URI",
            "Serving runtime choice",
        ],
        "steps": [
            "Prepare deployment (pre-flight checks)",
            "Deploy model",
            "Validate endpoint",
        ],
    },
    
    "compare_models": {
        "intent": "Compare multiple models before deciding",
        "workflow": "model-comparison",
        "tools": ["recommend_model", "plan_deployment"],
        "estimated_time": "15-20 minutes",
        "description": "Get recommendations for multiple configurations and compare",
    },
    
    "fine_tune_model": {
        "intent": "Fine-tune a model on my data",
        "workflow": "training-workflow",
        "tools": ["check_training_prerequisites", "prepare_training", "training"],
        "estimated_time": "varies",
        "description": "Prepare environment and fine-tune with LoRA/QLoRA",
    },
    
    "troubleshoot_deployment": {
        "intent": "Something went wrong with my deployment",
        "workflow": "troubleshooting",
        "tools": ["diagnose_resource", "get_resource_logs"],
        "estimated_time": "5-10 minutes",
        "prompts": ["diagnose-deployment-issues"],
        "description": "Identify and fix deployment issues",
    },
}
```

---

## 3. Templated Workflows (Reusable Artifacts)

Create example JSON/YAML files customers can adapt:

**Directory structure:**
```
src/rhoai_mcp/skills/
├── workflows/
│   ├── README.md                           # How to use templates
│   ├── recommend-and-deploy-template.json  # Parameterized workflow
│   └── examples/
│       ├── chatbot-deployment.json         # Pre-filled for chatbots
│       ├── rag-deployment.json             # For RAG pipelines
│       ├── code-completion.json            # For code generation
│       ├── content-generation.json         # For content creation
│       └── summarization.json              # For summarization tasks
```

**Example template:** `recommend-and-deploy-template.json`

```json
{
  "workflow": "recommend-and-deploy",
  "parameters": {
    "use_case": "chatbot_conversational",
    "expected_qps": 10,
    "user_count": 100,
    "namespace": "llm-deployments",
    "preferred_gpu_types": ["H100", "A100-80"],
    "optimization_profile": "balanced"
  },
  "overrides": {
    "prompt_tokens": 2048,
    "output_tokens": 512,
    "min_replicas": 1,
    "max_replicas": 3
  },
  "validation": {
    "require_gpu_available": true,
    "min_available_gpus": 1,
    "max_deployment_time_seconds": 600
  }
}
```

**Example:** `chatbot-deployment.json`

```json
{
  "workflow": "recommend-and-deploy",
  "name": "customer-chatbot",
  "description": "Deploy a chatbot model for customer service",
  "parameters": {
    "use_case": "chatbot_conversational",
    "expected_qps": 20,
    "user_count": 500,
    "namespace": "chatbot-prod",
    "preferred_gpu_types": ["H100", "A100-80"],
    "optimization_profile": "balanced",
    "cost_priority": "medium",
    "quality_priority": "high"
  },
  "deployment_config": {
    "min_replicas": 2,
    "max_replicas": 5,
    "hpa_threshold": 70,
    "timeout_seconds": 120
  },
  "monitoring": {
    "track_latency": true,
    "track_cost": true,
    "alert_on_slo_miss": true
  }
}
```

---

## 4. Documentation & Decision Guides

Create user-facing guides in `docs/skills/`:

```
docs/skills/
├── README.md                          # Overview of available skills
├── quick-start.md                     # 5-minute getting started
├── workflows/
│   ├── recommend-and-deploy.md        # Detailed workflow guide
│   ├── direct-deployment.md           # Deploy a known model
│   └── troubleshooting.md             # Common issues and fixes
├── reference/
│   ├── use-cases.md                   # Detailed use case descriptions
│   ├── gpu-types.md                   # GPU characteristics and sizing
│   ├── optimization-profiles.md       # Understanding each profile
│   └── slo-metrics.md                 # Latency metrics explained
├── examples/
│   ├── chatbot-walkthrough.md         # Step-by-step for chatbots
│   ├── rag-walkthrough.md             # Step-by-step for RAG
│   └── code-completion-walkthrough.md # Step-by-step for code models
└── troubleshooting/
    ├── gpu-unavailable.md             # GPU not found on cluster
    ├── deployment-failed.md           # InferenceService won't start
    ├── endpoint-unreachable.md        # Can't reach the model
    ├── slow-performance.md            # Model slower than expected
    └── out-of-memory.md               # OOM errors during deployment
```

### Sample: `docs/skills/workflows/recommend-and-deploy.md`

```markdown
# Recommend and Deploy Workflow

## Overview

This workflow helps you:
1. Get AI-powered recommendations for which model to deploy
2. Validate your cluster has the necessary GPU resources
3. Plan the deployment with appropriate configuration
4. Deploy safely with validation

## Time Required

10-15 minutes for a typical deployment.

## Prerequisites

- [ ] Know your use case (chatbot, RAG, code completion, etc.)
- [ ] Know your expected load (queries/second, number of users)
- [ ] Have a Kubernetes namespace ready
- [ ] Have GPU resources available (or plan to request)

## Step-by-Step Guide

### Step 1: Get Recommendations

Use the `recommend_model` tool to get recommendations:

```
Use Case: chatbot_conversational
Expected QPS: 10
User Count: 100
Check Cluster Fit: YES (this validates GPU availability)
```

You'll receive 4 recommendations:
- **Top Performance** - Best latency (usually largest/most expensive)
- **Top Cost** - Most efficient (smallest model that meets SLOs)
- **Top Balanced** - Sweet spot between quality and cost
- **Top Quality** - Best model quality/capability

### Step 2: Review GPU Fit Status

Each recommendation shows GPU fit status:
- ✅ **Available** - GPUs are ready
- ⚠️ **Partial** - Some GPUs available but might need more
- ❌ **Unavailable** - GPU type not found on cluster

### Step 3: Plan Deployment

Select a recommendation and call `plan_deployment`:

The plan shows:
- **Resolved Parameters** - Exact deployment config that will be created
- **Blocking Issues** - Stop here if any (must be resolved)
- **Warnings** - Non-critical but worth knowing about
- **Execution Steps** - What will happen

### Step 4: Execute Deployment

Run `execute_deployment` with the plan from Step 3.

The process:
1. Creates InferenceService in Kubernetes
2. Waits for it to become Ready (timeout: 10 minutes)
3. Tests the endpoint with a sample request
4. Returns the endpoint URL + SLO comparison

### Step 5: Validate & Monitor

Test with `test_model_endpoint` to confirm it's working.

## Decision Points

### "GPU Unavailable"

**Options:**
1. Use a smaller model (CPU only)
2. Request GPU allocation from cluster admin
3. Use a different GPU type if available

**Action:**
- Run `recommend_model` again with `preferred_gpu_types: []` to see CPU-only options
- Or adjust the `optimization_profile` to favor cost over quality

### "Deployment Failed"

**Check the issues list:**
- Namespace doesn't exist → Create it
- Storage URI invalid → Verify model location
- Runtime not found → Choose different runtime
- Insufficient resources → Scale up cluster or use smaller model

### "Endpoint Slow"

**Compare predicted vs. actual metrics:**
- Network overhead
- Queueing delay
- Model inference time
- Token generation speed

**Fixes:**
- Increase replicas with `scale_deployment`
- Adjust batch size in runtime config
- Move to faster GPU type

## Common Scenarios

### Scenario 1: Production Chatbot

```json
{
  "use_case": "chatbot_conversational",
  "expected_qps": 50,
  "user_count": 1000,
  "optimization_profile": "balanced"
}
```

### Scenario 2: Cost-Sensitive RAG

```json
{
  "use_case": "document_analysis_rag",
  "expected_qps": 10,
  "user_count": 100,
  "optimization_profile": "optimize_cost"
}
```

### Scenario 3: Low-Latency Code Completion

```json
{
  "use_case": "code_completion",
  "expected_qps": 100,
  "user_count": 500,
  "optimization_profile": "optimize_latency"
}
```

## Troubleshooting

See [troubleshooting guide](../troubleshooting/) for detailed help.
```

---

## 5. Distribution Options

### Option A: Package with MCP Server (Recommended for Phase 1)
- Prompts and tools are built-in
- Customers use via Claude Desktop/Code with `.mcp.json`
- No extra configuration needed
- Evolves with each server release

**Pros:** Simplest, built-in, always in sync  
**Cons:** Tightly coupled to server version

### Option B: Separate Skill Library (Phase 2+)
- Create separate package `rhoai-ai-skills`
- Distribute via PyPI, GitHub, or internal package manager
- Customers import and customize independently
- Can evolve on different cadence than server

**Pros:** Decoupled, flexible versioning, customers can fork  
**Cons:** Requires separate distribution, maintenance

### Option C: Claude Code Skills Integration (Phase 2+)
- Create `.claude/skills/` in the repo
- Define skill metadata and hooks
- Available as `/recommend-and-deploy` commands
- Integrates with Claude Code slash commands

**Pros:** Native IDE integration, discoverable  
**Cons:** Only works with Claude Code, requires skill registration

---

## Implementation Roadmap

### Phase 1: Core Prompts & Documentation (1-2 weeks)

**Deliverables:**
- [ ] Create `recommend-and-deploy` prompt
- [ ] Create `diagnose-deployment-issues` prompt
- [ ] Write `docs/skills/workflows/recommend-and-deploy.md`
- [ ] Create 3-4 example templates (chatbot, RAG, code-completion)
- [ ] Write quick-start guide

**Effort:** ~3-4 days  
**Impact:** High - customers have guided workflows immediately

### Phase 2: Decision Trees & Discovery (1 week)

**Deliverables:**
- [ ] Enhance `suggest_tools()` with decision trees
- [ ] Create `rhoai://skills/decision-trees` MCP resource
- [ ] Write workflow decision guide
- [ ] Create tool suggestion prompts

**Effort:** ~2-3 days  
**Impact:** Medium - helps customers find right workflow

### Phase 3: Comprehensive Documentation (1-2 weeks)

**Deliverables:**
- [ ] Complete `docs/skills/` structure
- [ ] Write reference guides (use cases, GPU types, optimization profiles)
- [ ] Create step-by-step walkthroughs for 5+ scenarios
- [ ] Write troubleshooting guides for all common issues

**Effort:** ~4-5 days  
**Impact:** High - self-service resolution of common problems

### Phase 4: Advanced Scenarios (2-3 weeks)

**Deliverables:**
- [ ] Workflow templates for advanced use cases (multi-model, A/B testing, cost optimization)
- [ ] Integration with Model Registry for auto-discovery
- [ ] Cost optimization prompts
- [ ] Performance tuning guides

**Effort:** ~5-7 days  
**Impact:** Medium-High - enables advanced customers

### Phase 5: Customer Validation & Iteration (Ongoing)

**Deliverables:**
- [ ] Gather feedback from early customers
- [ ] Refine prompts based on real interactions
- [ ] Add new decision trees based on common patterns
- [ ] Update docs with customer scenarios

**Effort:** ~3-5 days per iteration  
**Impact:** High - ensures skills solve real problems

---

## Success Metrics

### Phase 1-2 (Core Workflows)
- [ ] Customers can deploy an LLM in <15 minutes with prompts
- [ ] 80%+ of customers find the right workflow
- [ ] No manual support for "how do I get started" questions

### Phase 3 (Documentation)
- [ ] Self-service troubleshooting resolves 70%+ of common issues
- [ ] Docs get <10% incorrect/out-of-date ratings
- [ ] Customers report workflows are easy to understand and follow

### Phase 4+ (Scale)
- [ ] Customers can customize workflows for their use cases
- [ ] Templates reduce deployment setup time by 50%+
- [ ] Community contributes examples and improvements

---

## Design Principles

### 1. **Customer-First Language**
- Use customer terminology (not just technical)
- Clear decision points with explanations
- No unexplained jargon or acronyms

### 2. **Fail-Safe Defaults**
- Workflows validate before executing
- Provide clear error messages with fixes
- Never silently ignore issues

### 3. **Transparency**
- Show what's happening at each step
- Explain GPU matching logic
- Reveal cost/latency trade-offs

### 4. **Customizability**
- Template parameters are easy to override
- Prompts accept configuration inputs
- Examples show how to adapt

### 5. **Discoverability**
- Workflows appear in tool suggestions
- Prompts are listed in MCP resources
- Templates are in well-documented location

---

## Technical Considerations

### Prompt Parameters
Make prompts accept key parameters so agents can adapt:

```python
def recommend_and_deploy(
    use_case: str,                           # Required
    namespace: str,                          # Required
    qps: float,                              # Required
    user_count: int = 10,                    # Optional
    preferred_gpu_types: list[str] | None = None,  # Optional
    optimization_profile: str = "balanced",  # Optional with defaults
) -> str:
```

### Error Handling
Prompts should guide on common errors:

```markdown
**If you see "gpu_unavailable":**
- This means the GPU type isn't on your cluster
- Options:
  1. Use CPU-only model (slower but no GPU needed)
  2. Choose a different GPU type: {available_gpu_types}
  3. Request GPU allocation from your admin
```

### Decision Guidance
Be explicit about trade-offs:

```markdown
**Recommendations are ranked by:**
- Performance: Lowest latency (fastest responses)
- Cost: Lowest hourly cost (best for budget)
- Balanced: Best overall score (recommended for most)
- Quality: Highest model quality (best accuracy)

Choose based on your priority.
```

---

## Related Resources

- [ARCHITECTURE.md](../ARCHITECTURE.md) - Planner composite technical details
- [README.md](../README.md) - Feature overview and tool listing
- [CLAUDE.md](../CLAUDE.md) - Development principles and commands
- [OVERVIEW_fpj-ai-skills.md](../OVERVIEW_fpj-ai-skills.md) - Implementation details

---

## Questions for Early Adopters

When gathering feedback, ask:

1. **Workflow:** Did you know which tools to use? Was the order clear?
2. **Decisions:** Were the decision points clear? Did you know which option to choose?
3. **Issues:** When something went wrong, could you figure out why?
4. **Time:** How long did deployment take? Was that acceptable?
5. **Customization:** Could you adapt the workflow to your use case?
6. **Next:** What would make this even more useful?

---

**Last Updated:** 2026-09-01  
**Status:** Strategy Document - Ready for Implementation  
**Next Step:** Begin Phase 1 implementation
