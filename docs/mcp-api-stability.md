# MCP API Stability and GA Readiness

**Date:** September 2026

**Purpose:** Define the API tier classification and stability guarantees for the `rhoai-mcp` MCP server as it moves from Technology Preview to General Availability.
This document applies the [RHAI Release Stages and API Tiers](https://github.com/red-hat-data-services/rhai-process-docs/blob/main/docs/planning/release-stages-and-api-tiers.md) framework to the specific characteristics of an MCP (Model Context Protocol) server consumed by AI agents.

**Audience:** RHAI engineering, QE, PM, and architecture teams working on `rhoai-mcp` or integrating with it.

## Context: How `rhoai-mcp` Differs from Operator-Managed Components

`rhoai-mcp` is not a typical RHAI component. Several characteristics set it apart from the Operator-managed services the release-stages-and-api-tiers framework was primarily designed for:

| Characteristic | Typical RHOAI Component | `rhoai-mcp` |
|---|---|---|
| **Lifecycle management** | RHOAI Operator | MCP Lifecycle Operator (MCPLO) via MCP Catalog |
| **K8s APIs / CRDs defined** | Yes (often core to the API surface) | None — consumes others' CRDs, defines none |
| **REST APIs exposed** | Often (with SLA/versioning) | None |
| **Primary API surface** | K8s CRDs and/or REST endpoints | MCP Tools, Resources, and Prompts |
| **API consumers** | Humans, scripts, controllers | AI agents (LLMs, Agentic frameworks/loops) |
| **Consumer discovers API via** | Documentation, OpenAPI specs | MCP protocol (`tools/list`, `resources/list`) |

### Deployment paths

At Architects' request, `rhoai-mcp` is surfaced through the **MCP Catalog**; the `rhoai-mcp` can be deployed through:

1. **MCP Catalog + MCPLO**: this the standard path for environments with the MCP Lifecycle Operator installed, provided required/dependent resources are defined prior the definition of the `MCPServer` resource; e.g. SA, ConfigMaps, etc.
2. **GitOps (via kustomize)**: using the `openshift-oidc-mcpserver` overlay (for MCPLO-managed environments) or the `openshift-oidc` overlay (for environments without MCPLO or MCPLO "`Removed`").

### Boundary with MCPLO

The MCPLO is the lifecycle operator for MCP servers on the RHOAI platform.
The operational concerns it governs (e.g.: deployment, upgrades, scaling, health monitoring, network exposure, TLS termination, rollout strategy, version pinning, ...) are beyond the scope of this document, analogous to the way the RHOAI Operator governs these concerns for Operator-managed Components.

What `rhoai-mcp` owns is the **MCP API surface and its compatibility contract between versions**: the Tier 2 goal-oriented stability guarantee defined in this document.

When a new version changes the MCP tool surface, the rules in [Change and Deprecation Process](#change-and-deprecation-process) below determine whether that change is non-breaking (e.g. tool renames, parameter changes) or requires a deprecation window (e.g. use-case removal, capability removal).

### Operator integration row in the Quick Reference Card

The GA quick reference card in the release-stages-and-api-tiers framework currently requires "Integrated into the RHOAI operator: Yes" for GA.
`rhoai-mcp` is not managed by the RHOAI Operator, as it is managed by the **MCPLO**.
The reference card pre-dates the MCPLO.

We derive the MCPLO integration satisfies this requirement for MCP-Catalog-surfaced components, as the MCPLO is the designated lifecycle operator for MCP servers in the RHOAI platform.

## API Tier Classification

### Why MCP Tools Are Not Traditional APIs

The release-stages-and-api-tiers framework defines tiers for K8s CRDs, REST endpoints, CLI commands, and SDK interfaces: these are all consumed **programmatically** by humans or automation scripts.
A renamed field, a changed parameter type, or a removed endpoint is a breaking change because consumers hardcode these identifiers.

MCP Tools are fundamentally different:

- **Dynamic discovery**: AI agents discover tools at runtime via MCP's `tools/list` protocol. They do not necessarily hardcode tool names.
- **Natural language interpretation**: AI agents read tool descriptions and parameter schemas to understand what a tool does. They adapt to changes in naming, parameter structure, and response format.
- **Goal-oriented consumption**: An AI agent's objective is to achieve a goal (e.g., "deploy a model for serving"), not to call a specific function with a specific signature.

This means that renaming `deploy_model` to `deploy_inference_service`, adding or removing a parameter, or consolidating three tools into one is **not a breaking change** in the traditional sense; for as long as, an AI agent can still discover and use the tools to achieve the original goal.

### Tier Designation: Tier 2 — Goal-Oriented Stability

rhoai-mcp's MCP API surface is designed for a **Tier 2** classification, with a **goal-oriented stability definition** (more below).

**Why Tier 2:**

- **Not Tier 1**: Tier 1 is for APIs customers hardcode in automation (CRDs, REST endpoints).
MCP tools are dynamically discovered by AI agents, and the surface (currently 76+ tools, 5 resources, 18 prompts) is still being refined based on real-world agent usage.
Locking individual tool signatures for 18 months would prevent improving this `rhoai-mcp` for better agent ergonomics without providing meaningful additional value to the AI Agent consumers.

- **Not Tier 4**: Tier 4 is for internal-only APIs.
MCP tools are explicitly the external interface, as they are what customers' AI agents interact with. Hence, not really applicable.

- **Tier 2**: Appropriate for an API surface that is maturing and may evolve based on customer feedback, while still providing meaningful stability guarantees.
The deprecation window gives consumers time to adapt their workflows.

### What "Tier 2 Stability" Means for MCP

Because MCP tools are consumed by AI agents rather than hardcoded in scripts, the stability guarantee applies to the **capability set** (the "use cases"/goals that an AI Agent can achieve), not to individual tool names or parameter signatures.

**Stable (Tier 2 guarantee):**

- The set of achievable goals (see [Capability Catalog](#capability-catalog))
- MCP Resource URI scheme (`rhoai://`)
- The availability of MCP Prompts for each documented workflow category

**Can change without deprecation:**

- Individual tool names
- Tool parameter names and types
- Tool response structure (when semantically equivalent)
- Consolidating or splitting tools
- Adding new tools, parameters, resources, or prompts
- Tool descriptions and prompt text

***Most importantly:***

- **Deprecation of a capability/use-case** (removing the ability to achieve a documented goal entirely) requires a deprecation notice, consistent with Tier 2.
- **Modification of tools** that implement a capability (renaming, restructuring, consolidating) does _*not*_ require deprecation, as long as an MCP-compliant AI agent can still achieve the same goal using the modified tool(s).

### Rationale for Goal-Oriented Stability

The "Python and SDK libraries" clause in the release-stages-and-api-tiers framework states that tiers apply to the "public interface (the classes, functions, and parameters) documented for external use."

For an MCP server consumed by AI agents, the "public interface" is the **capability set** (the goals an AI Agent can achieve) because:

1. AI agents do not import function names or reference parameter types at compile time.
2. The MCP protocol's tool discovery mechanism (`tools/list`) means agents re-discover the tool surface on every session.
3. Customer value comes from "I can use my AI agent to deploy a model on RHOAI leveraging the `rhoai-mcp` MCP server", not from "there is a tool named `deploy_model` with a parameter named `storage_uri`".

This is analogous to how a REST API's stability guarantee is about the *endpoints and their behavior*, not about the internal function names that implement those endpoints.

## Capability Catalog

The following table defines the **stable capability set**: the use cases that Tier 2 guarantees an AI agent can achieve through this `rhoai-mcp` MCP server.

Each Capability maps to (one or more) MCP Tools, Resources, or Prompts that implement it.
The Tools may change; the Capability must remain achievable.

### [RHAIRFE-1705](https://redhat.atlassian.net/browse/RHAIRFE-1705): Intent-Based Model Recommendation

**Goal:** An AI agent can guide a user from a natural-language description of their use case to a set of justified, ranked model recommendations — without requiring the user to manually research models, benchmarks, or hardware compatibility.

**What the capability covers (in-scope):**

1. The agent can accept user intent (use case description, scale, priority) and map it to a workload profile with SLO targets (TTFT, ITL, E2E latency).
2. The agent can discover the cluster's available GPU inventory and use it as a hardware constraint.
3. The agent can query a recommendation engine that cross-references model capabilities, benchmark performance data, and cluster hardware capacity.
4. The agent can present multiple ranked model options (by cost, performance, quality, and balanced trade-offs), each with: a justification linking model strengths to the user's stated intent, benchmark-backed latency and throughput predictions, and a cluster hardware fit assessment.
5. Recommended models are deployable on the user's available cluster hardware without manual resource verification.

**Implementing MCP tools (may change):**

| Tool | Role in capability |
|---|---|
| `list_use_cases` | Enumerate supported use case categories |
| `get_use_case_defaults` | Retrieve workload profile and SLO targets for a use case |
| `get_expected_rps` | Estimate request throughput from concurrent user count |
| `get_cluster_resources` | Discover cluster GPU inventory for hardware-aware filtering |
| `recommend_model` | Return ranked model recommendations across optimization dimensions |
| `list_data_science_projects` | Confirm target namespace exists (supporting) |

**Boundary:** This capability covers model *selection* only. Deployment configuration and execution are covered by RHAIRFE-1706.

### [RHAIRFE-1706](https://redhat.atlassian.net/browse/RHAIRFE-1706): Deployment Optimization and Execution

**Goal:** An AI agent can take a chosen model and guide a user through finding the optimal GPU configuration for their workload, then deploy the model to a specified namespace on the cluster — all backed by empirical benchmark data rather than manual trial-and-error.

**What the capability covers (in-scope):**

1. For a selected model, the agent can generate multiple distinct deployment configurations varying GPU type, GPU count, and tensor parallelism.
2. The agent can provide empirical performance evidence (TTFT, tail latency, throughput, GPU utilization, cost) for each configuration, grounded in pre-validated benchmark data.
3. The agent can recommend a deployment configuration based on the user's stated priority (cost vs. latency vs. throughput vs. quality), with a justification linking the recommendation to reported metrics.
4. The agent can resolve all deployment prerequisites (serving runtime availability, storage URI, namespace existence) and present a complete deployment plan for user approval.
5. With explicit user confirmation, the agent can execute the deployment (creating serving runtimes and inference services as needed) and report the resulting endpoint URL or provisioning status.

**Implementing MCP tools (may change):**

| Tool | Role in capability |
|---|---|
| `get_use_case_defaults` | Retrieve workload profile and SLO targets |
| `get_expected_rps` | Estimate request throughput from concurrent user count |
| `get_cluster_resources` | Discover cluster GPU inventory for hardware-aware filtering |
| `recommend_model` | Return ranked deployment configurations across optimization dimensions |
| `plan_deployment` | Resolve runtime, storage, and pre-conditions; generate deployment parameters |
| `execute_deployment` | Create the InferenceService and poll for initial status |
| `list_data_science_projects` | Enumerate existing namespaces |
| `create_data_science_project` | Create a namespace if needed |
| `list_serving_runtimes` | Inspect available runtimes in the target namespace |
| `create_serving_runtime` | Instantiate a serving runtime from a template |
| `list_inference_services` | Check existing deployments |
| `get_inference_service` | Monitor deployment status |
| `get_model_endpoint` | Retrieve the inference endpoint URL once the service is ready |

**Boundary:** This capability assumes a model has already been selected (by the user or via RHAIRFE-1705). It does not cover model selection or ongoing production monitoring/auto-tuning.

## Change and Deprecation Process

### Changes that do NOT require a deprecation notice

The following changes can be made in any release and documented in release notes:

- Renaming an MCP tool (e.g., `deploy_model` → `deploy_inference_service`)
- Adding, removing, or renaming tool parameters
- Changing tool response structure (when the new structure conveys equivalent information)
- Consolidating multiple tools into one, or splitting one tool into multiple
- Adding new tools, resources, or prompts
- Changing tool descriptions or prompt template text
- Reordering or restructuring the internal plugin/domain layout

### Changes that DO require a deprecation notice

The following changes require a deprecation announcement in release notes, with the capability remaining available for at least 9 months or 3 minor releases (whichever is longer) after the announcement:

- **Removing a capability entirely** — making it impossible for an AI agent to achieve a documented goal from the capability catalog.
- **Removing the MCP Resource URI scheme** (`rhoai://`) or changing it incompatibly.
- **Removing an entire prompt category** (training, deployment, troubleshooting, exploration, project setup) without replacement.

### How to deprecate a Capability

1. **Announce** in release notes: "Capability X is deprecated and will be removed no earlier than [date, at least 9 months out]."
2. **Document alternatives** if the capability is being replaced by a different approach.
3. **Keep the capability functional** during the deprecation window.
4. **Remove** after the window has passed, with a final release note confirming removal.

## Relationship to the MCP Specification

The MCP protocol itself provides mechanisms that support the goal-oriented stability model:

- **`tools/list`**: Agents discover available tools and their schemas at the start of every session. Tool renames are transparent to agents.
- **`resources/list`**: Agents discover available resources dynamically.
- **`prompts/list`**: Agents discover available prompts dynamically.
- **Tool descriptions and JSON Schema**: Agents interpret tool purpose from descriptions and validate input via schemas. Well-written descriptions ensure agents can use tools correctly even after structural changes.

This dynamic discovery model is why goal-oriented stability is appropriate: the protocol was designed for an environment where tools can evolve, and consumers (AI agents) adapt automatically.

## No CRD or REST API considerations

For completeness: at the time of writing this document, `rhoai-mcp` defines no Kubernetes CRDs and exposes no REST APIs of its own.
It consumes CRDs and APIs defined by other RHOAI components (KServe, Kubeflow Training Operator, Data Science Pipelines, Model Registry, etc.).
The stability of those upstream APIs is governed by their respective component owners and is outside the scope of this document.

The MCP server's HTTP transport endpoints (`/sse`, `/mcp`) are MCP protocol transport: they are not a REST API surface.
Their stability is governed by the MCP specification, not by RHAI API tiers.
