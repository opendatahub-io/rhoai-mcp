# RHOAI MCP Server

[![CI](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/ci.yml)
[![Container Build](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/container-build.yml/badge.svg)](https://github.com/opendatahub-io/rhoai-mcp/actions/workflows/container-build.yml)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/opendatahub-io/rhoai-mcp)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/opendatahub-io/rhoai-mcp)
[![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

An MCP (Model Context Protocol) server that enables AI agents to interact with Red Hat OpenShift AI (RHOAI) environments. This server replicates the capabilities of the OpenShift AI Dashboard through programmatic tools.

## Features

- **Project Management**: Create, list, and manage Data Science Projects
- **Workbench Operations**: Create, start, stop, and delete Jupyter workbenches
- **Model Serving**: Deploy and manage InferenceServices with KServe
- **Data Connections**: Manage S3 credentials for data access
- **Pipelines**: Configure Data Science Pipelines infrastructure
- **Storage**: Create and manage persistent volume claims
- **Training**: Fine-tune models with Kubeflow Training Operator
- **Quickstarts**: Discover and deploy Red Hat AI quickstarts from an OCI registry
- **MCP Prompts**: Workflow guidance for multi-step operations (19 prompts)

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python 3.10+ | Core language |
| **MCP Framework** | FastMCP 1.0+ | Model Context Protocol server |
| **Kubernetes Client** | kubernetes-python 28.1+ | Cluster API interactions |
| **Data Validation** | Pydantic 2.0+ | Type-safe models and settings |
| **HTTP Client** | httpx 0.27+ | Async HTTP requests |
| **Container Base** | Red Hat UBI 9 | Production container image |
| **Package Manager** | uv | Fast Python dependency management |

## Extending

Want to add your team's capabilities to rhoai-mcp? See the [contributor guide](CONTRIBUTING.md#adding-plugins) for templates, documentation, and a working example domain.

## Installation

For OpenShift installation, refer to [OpenShift deployment](#openshift-deployment).

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/opendatahub-io/rhoai-mcp.git
cd rhoai-mcp

# Install dependencies
uv sync

# Run the server
uv run rhoai-mcp
```

### Using pip

```bash
pip install -e .
rhoai-mcp
```

### Using Container (Podman/Docker)

```bash
# Build the image
make build

# Run with HTTP transport
make run-http

# Run with STDIO transport (interactive)
make run-stdio

# Run with debug logging
make run-dev
```

Or run directly without Make:

```bash
# Build
podman build -f Containerfile -t rhoai-mcp:latest .

# Run with HTTP transport
podman run -p 8000:8000 \
  -v ~/.kube/config:/opt/app-root/src/kubeconfig/config:ro \
  -e RHOAI_MCP_AUTH_MODE=kubeconfig \
  -e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
  rhoai-mcp:latest --transport sse

# Run with STDIO transport
podman run -it \
  -v ~/.kube/config:/opt/app-root/src/kubeconfig/config:ro \
  -e RHOAI_MCP_AUTH_MODE=kubeconfig \
  -e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
  rhoai-mcp:latest --transport stdio
```

Available Make targets:

| Target | Description |
|--------|-------------|
| `make build` | Build the container image |
| `make run-http` | Run with SSE transport on port 8000 |
| `make run-streamable` | Run with streamable-http transport |
| `make run-stdio` | Run with STDIO transport (interactive) |
| `make run-dev` | Run with debug logging |
| `make run-token` | Run with token auth (requires TOKEN and API_SERVER) |
| `make stop` | Stop the running container |
| `make logs` | View container logs |
| `make clean` | Remove container and image |

### Kubernetes Deployment

Deploy using Kustomize with environment-specific overlays:

```bash
# KIND / local development
kustomize build deploy/kustomize/overlays/kind/ | kubectl apply -f -

# OpenShift — see "OpenShift deployment" section below
```

The Kustomize structure uses a shared base with per-environment overlays:

```text
deploy/kustomize/
├── base/                    # Shared resources (all environments)
│   ├── clusterrole.yaml     # RBAC for RHOAI resources
│   ├── deployment.yaml      # Hardened pod spec (non-root, read-only rootfs, probes)
│   ├── configmap.yaml       # Default config (SSE transport, INFO logging)
│   └── ...
└── overlays/
    ├── kind/                # NodePort, DEBUG logging, imagePullPolicy: Never
    ...                      # other platform's overlays
```

The **KIND overlay** enables debug logging, dangerous operations, `NodePort` service type, and `imagePullPolicy: Never` (for `kind load docker-image`).

### OpenShift deployment

Two overlays are available, both configured with multi-user OIDC authentication:

**1. MCP Lifecycle Operator (recommended)**

For environments with the MCP Lifecycle Operator (MCPLO) installed.
The MCPLO operator manages the Deployment, Service, and NetworkPolicy from an `MCPServer` resource:

```bash
kustomize build deploy/kustomize/overlays/openshift-oidc-mcpserver | oc apply -f -
```

**2. GitOps / standalone**

For environments without MCPLO, or where MCPLO is in `Removed` state:

```bash
kustomize build deploy/kustomize/overlays/openshift-oidc | oc apply -f -
```

See [`deploy/kustomize/overlays/openshift-oidc/README.md`](deploy/kustomize/overlays/openshift-oidc/README.md) for RBAC model, configuration details, and verification steps.

**Model Catalog NetworkPolicy**

When [required](deploy/kustomize/overlays/openshift-oidc/README.md), apply the NetworkPolicy to allow `rhoai-mcp` to connect to the RHOAI Model Catalog:

```bash
oc apply -f deploy/kustomize/overlays/openshift/networkpolicy.yaml
```

## Configuration

The server can be configured via environment variables (with `RHOAI_MCP_` prefix) or a `.env` file.

### Authentication

The server supports three authentication modes for connecting to the Kubernetes API:

1. **Auto (default)**: Tries in-cluster authentication first, falls back to kubeconfig
2. **Kubeconfig**: Uses a kubeconfig file
3. **Token**: Uses explicit API server URL and token

```bash
# Auto mode (default)
export RHOAI_MCP_AUTH_MODE=auto

# Kubeconfig mode
export RHOAI_MCP_AUTH_MODE=kubeconfig
export RHOAI_MCP_KUBECONFIG_PATH=/path/to/kubeconfig
export RHOAI_MCP_KUBECONFIG_CONTEXT=my-context

# Token mode
export RHOAI_MCP_AUTH_MODE=token
export RHOAI_MCP_API_SERVER=https://api.cluster.example.com:6443
export RHOAI_MCP_API_TOKEN=sha256~xxxxx
```

For multi-user deployments, the OpenShift OIDC overlays add caller-level authentication: each request's bearer token is validated via K8s TokenReview, a SubjectAccessReview pre-flight filters tools based on the caller's RBAC, and resource access is ultimately enforced by the K8s API against the caller's own token.
See [`deploy/kustomize/overlays/openshift-oidc/README.md`](deploy/kustomize/overlays/openshift-oidc/README.md) for OIDC configuration details.

### Transport

```bash
# stdio (default) - for Claude Desktop and similar tools
export RHOAI_MCP_TRANSPORT=stdio

# HTTP transports
export RHOAI_MCP_TRANSPORT=streamable-http
export RHOAI_MCP_HOST=127.0.0.1
export RHOAI_MCP_PORT=8000
```

### Safety Settings

```bash
# Enable delete of resources NOT created by rhoai-mcp (disabled by default)
export RHOAI_MCP_ENABLE_DANGEROUS_OPERATIONS=true

# Read-only mode (disable all write operations)
export RHOAI_MCP_READ_ONLY_MODE=true
```

By default, delete operations are ONLY scoped to resources created by this server (labeled `app.kubernetes.io/managed-by: rhoai-mcp`).
Deleting resources created outside rhoai-mcp requires `RHOAI_MCP_ENABLE_DANGEROUS_OPERATIONS=true`.

### Safety Features Summary

| Feature | Description | Default |
|---------|-------------|---------|
| **Read-Only Mode** | Disables all create/update/delete operations | Disabled |
| **Dangerous Operations Gate** | Deletes restricted to resources created by rhoai-mcp; enable to allow deleting unmanaged resources | Disabled |
| **Confirmation Pattern** | Delete tools require `confirm=True` parameter | Required |
| **Credential Masking** | S3 secret keys are masked in all responses | Always |
| **RBAC-Aware** | Uses OpenShift Projects API to respect user permissions | Always |
| **Auth Validation** | Validates authentication configuration at startup | Always |

### Workflow Tokens

Workflow tokens enforce ordering of multi-step MCP tool calls. When tools are decorated with `@workflow_step`, each step signs its output with an HMAC token that the next step must present and verify before executing. This prevents agents from skipping prerequisite steps.

```bash
# Set an explicit HMAC secret (recommended for production — tokens survive restarts)
export RHOAI_MCP_WORKFLOW_HMAC_SECRET=my-secret-key

# Adjust token time-to-live (default: 3600 seconds / 1 hour)
export RHOAI_MCP_WORKFLOW_TOKEN_TTL=1800
```

| Variable | Description | Default |
|----------|-------------|---------|
| `RHOAI_MCP_WORKFLOW_HMAC_SECRET` | HMAC secret for signing workflow tokens | Random per process |
| `RHOAI_MCP_WORKFLOW_TOKEN_TTL` | Token time-to-live in seconds | `3600` |

If no secret is configured, a random one is generated at process startup. This means tokens are not portable across server restarts — suitable for development but not production deployments where long-running workflows may span restarts.

### Model Registry

The MCP server integrates with the RHOAI Model Registry to list and query registered models. By default, it auto-discovers the Model Registry service in the cluster.

#### Discovery Modes

```bash
# Auto-discovery (default) - finds Model Registry in the cluster
export RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE=auto

# Manual - use a specific URL
export RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE=manual
export RHOAI_MCP_MODEL_REGISTRY_URL=https://model-registry.example.com
```

#### Authentication

When accessing the Model Registry via an external route (outside the cluster), authentication is typically required (OAuth for OAuth-proxied routes; explicit token auth is also supported):

```bash
# No authentication (default) - for in-cluster access
export RHOAI_MCP_MODEL_REGISTRY_AUTH_MODE=none

# OAuth authentication - uses your oc login token
export RHOAI_MCP_MODEL_REGISTRY_AUTH_MODE=oauth

# Explicit token authentication
export RHOAI_MCP_MODEL_REGISTRY_AUTH_MODE=token
export RHOAI_MCP_MODEL_REGISTRY_TOKEN=sha256~xxxxx
```

| Auth Mode | Description | Use Case |
|-----------|-------------|----------|
| `none` | No authentication headers | In-cluster access via port 8080 |
| `oauth` | Uses OAuth token from kubeconfig | External route with OAuth proxy |
| `token` | Uses explicit bearer token | Service accounts, CI/CD |

#### External Route Access

To access the Model Registry from outside the cluster via an OpenShift Route:

```bash
# 1. Log in to OpenShift (this stores the OAuth token in kubeconfig)
oc login --server=https://api.cluster.example.com:6443

# 2. Configure the MCP server to use the external route with OAuth
export RHOAI_MCP_MODEL_REGISTRY_URL=https://model-catalog.apps.cluster.example.com
export RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE=manual
export RHOAI_MCP_MODEL_REGISTRY_AUTH_MODE=oauth

# 3. Optional: Skip TLS verification for self-signed certificates (not recommended)
# export RHOAI_MCP_MODEL_REGISTRY_SKIP_TLS_VERIFY=true
```

#### Port-Forwarding Alternative

If no external route is available, you can use port-forwarding:

```bash
# Set up port-forwarding to the Model Registry service
kubectl port-forward -n rhoai-model-registries svc/model-catalog 8080:8443

# Configure the MCP server to use localhost
export RHOAI_MCP_MODEL_REGISTRY_URL=http://localhost:8080
export RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE=manual
```

#### All Model Registry Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `RHOAI_MCP_MODEL_REGISTRY_ENABLED` | Enable Model Registry integration | `true` |
| `RHOAI_MCP_MODEL_REGISTRY_URL` | Model Registry service URL | Auto-discovered |
| `RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE` | `auto` or `manual` | `auto` |
| `RHOAI_MCP_MODEL_REGISTRY_AUTH_MODE` | `none`, `oauth`, or `token` | `none` |
| `RHOAI_MCP_MODEL_REGISTRY_TOKEN` | Explicit bearer token (when auth_mode=token) | None |
| `RHOAI_MCP_MODEL_REGISTRY_TIMEOUT` | Request timeout in seconds | `30` |
| `RHOAI_MCP_MODEL_REGISTRY_SKIP_TLS_VERIFY` | Skip TLS certificate verification | `false` |

### Quickstarts

The MCP server can discover Red Hat AI quickstarts from an OCI registry and
deploy them by running an installer Job in the cluster. Discovery works without
a cluster; deployment actions require the installer RBAC prerequisite below.

#### Quickstart Installer RBAC

Deployment actions (`run_quickstart_action`) create an installer Job — and its
parameters Secret — in a dedicated namespace (`RHOAI_MCP_QUICKSTART_JOB_NAMESPACE`,
default `openshift-quickstarts`), running as a dedicated ServiceAccount
(`RHOAI_MCP_QUICKSTART_INSTALLER_SERVICE_ACCOUNT`, default `quickstart-installer`).

**The MCP server does not create this namespace or ServiceAccount.** Because the
installer runs with elevated privileges, a cluster admin must provision them
once, out-of-band:

```bash
oc apply -f deploy/quickstarts/installer-rbac.yaml
```

The quickstarts plugin health check verifies the namespace and ServiceAccount
exist and points back at this manifest when they are missing. (The **target**
application namespace is created by the installer Job itself at INSTALL time.)

#### All Quickstart Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `RHOAI_MCP_QUICKSTART_REGISTRY_REF` | OCI reference of the quickstart registry index artifact | `quay.io/rh-ai-quickstart/quickstart-registry:latest` |
| `RHOAI_MCP_QUICKSTART_JOB_NAMESPACE` | Namespace where installer Jobs (and their param Secrets) run | `openshift-quickstarts` |
| `RHOAI_MCP_QUICKSTART_INSTALLER_SERVICE_ACCOUNT` | Pre-provisioned ServiceAccount the installer Jobs run as | `quickstart-installer` |
| `RHOAI_MCP_QUICKSTART_INSTALLER_CPU_REQUEST` | CPU request for the installer container | `100m` |
| `RHOAI_MCP_QUICKSTART_INSTALLER_CPU_LIMIT` | CPU limit for the installer container | `1` |
| `RHOAI_MCP_QUICKSTART_INSTALLER_MEMORY_REQUEST` | Memory request for the installer container | `256Mi` |
| `RHOAI_MCP_QUICKSTART_INSTALLER_MEMORY_LIMIT` | Memory limit for the installer container | `2Gi` |
| `RHOAI_MCP_QUICKSTART_JOB_TTL_SECONDS` | `ttlSecondsAfterFinished` for installer Jobs (0 = delete immediately) | `3600` |
| `RHOAI_MCP_QUICKSTART_JOB_ACTIVE_DEADLINE_SECONDS` | Hard wall-clock cap on installer Job runtime | `7200` |
| `RHOAI_MCP_QUICKSTART_OCI_TIMEOUT` | Timeout in seconds for quickstart OCI registry requests | `30` |
| `RHOAI_MCP_QUICKSTART_OCI_SKIP_TLS_VERIFY` | Skip TLS verification when pulling quickstart OCI artifacts | `false` |
| `RHOAI_MCP_QUICKSTART_ALLOWED_REPOS` | Comma-separated allowlist of OCI repository prefixes for quickstart artifacts | `quay.io/rh-ai-quickstart/` |

## Usage with agent harness

These examples connect an agent harness to a deployed rhoai-mcp server over HTTP.
Replace `<RHOAI_MCP_ROUTE>` with the Route URL of your deployment (e.g. `oc get route rhoai-mcp -n rhoai-mcp -o jsonpath='{.spec.host}'`).

### Claude Code

```sh
claude mcp add rhoai-mcp \
  --scope project \
  --transport http \
  --url "https://<RHOAI_MCP_ROUTE>/mcp" \
  --header "Authorization: Bearer \${OCP_TOKEN}"
```

Then start Claude Code with the token:

```sh
OCP_TOKEN=$(oc whoami -t) claude
```

### Codex

`.codex/config.toml`:

```toml
[mcp_servers.rhoai-mcp]
url = "https://<RHOAI_MCP_ROUTE>/mcp"
http_headers_helper = ".codex/rhoai-mcp-headers"
```

`.codex/rhoai-mcp-headers` (make executable with `chmod +x`):

```sh
#!/bin/sh
set -eu
token="$(oc whoami -t)"
printf '{"Authorization":"Bearer %s"}\n' "$token"
```

Note: Codex [caches `http_headers_helper`](https://learn.chatgpt.com/docs/extend/mcp?surface=cli#cli-streamable-http-servers), then re-runs the helper after a same-origin 401 or 403.
Alternatively, do similar setup of exporting OCP_TOKEN, and setup config.toml to use the Env variable instead of `http_headers_helper`.

### OpenCode

`opencode.json`:

```json
{
  "mcp": {
    "servers": {
      "rhoai-mcp": {
        "type": "remote",
        "url": "https://<RHOAI_MCP_ROUTE>/mcp",
        "oauth": false,
        "headers": {
          "Authorization": "Bearer {env:OCP_TOKEN}"
        }
      }
    }
  }
}
```

```sh
export OCP_TOKEN="$(oc whoami -t)"
opencode service restart
opencode reload
opencode mcp list
opencode
```

### Pi

Install the MCP adapter first:

```sh
pi install npm:pi-mcp-adapter
```

`.mcp.json` (project root):

```json
{
  "mcpServers": {
    "rhoai-mcp": {
      "url": "https://<RHOAI_MCP_ROUTE>/mcp"
    }
  }
}
```

`.pi/mcp.json` (Pi-specific override for auth headers):

```json
{
  "mcpServers": {
    "rhoai-mcp": {
      "headers": {
        "Authorization": "!.pi/oc-whoami-t-bearer"
      }
    }
  }
}
```

`.pi/oc-whoami-t-bearer` (make executable with `chmod +x`):

```sh
#!/bin/sh
exec printf 'Bearer %s\n' "$(oc whoami -t)"
```

Note: the [`!` prefix in headers](https://github.com/nicobailon/pi-mcp-adapter#server-options) tells Pi to execute the script and use its stdout as the header value.

## Local usage with agent harness

### Local usage with Claude Code

Add to your project's `.mcp.json` file:

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/opendatahub-io/rhoai-mcp", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_KUBECONFIG_PATH": "/home/user/.kube/config"
      }
    }
  }
}
```

### Local usage with Claude Desktop

Add to your Claude Desktop configuration (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/opendatahub-io/rhoai-mcp", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_KUBECONFIG_PATH": "/home/user/.kube/config"
      }
    }
  }
}
```

### Local Development

For contributors working with a local clone:

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rhoai-mcp", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_KUBECONFIG_PATH": "/home/user/.kube/config"
      }
    }
  }
}
```

### Local usage with Container Image (Podman/Docker)

First, build the container image:

```bash
make build
```

Then configure Claude Desktop with the container:

**Podman:**

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "podman",
      "args": [
        "run", "-i", "--rm",
        "--userns=keep-id",
        "-v", "${HOME}/.kube/config:/opt/app-root/src/kubeconfig/config:ro",
        "-e", "RHOAI_MCP_AUTH_MODE=kubeconfig",
        "-e", "RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config",
        "rhoai-mcp:latest"
      ]
    }
  }
}
```

**Docker:**

```json
{
  "mcpServers": {
    "rhoai": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "${HOME}/.kube/config:/opt/app-root/src/kubeconfig/config:ro",
        "-e", "RHOAI_MCP_AUTH_MODE=kubeconfig",
        "-e", "RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config",
        "rhoai-mcp:latest"
      ]
    }
  }
}
```

Note: The container uses `stdio` transport by default, which is required for Claude Desktop integration.

## Available Tools

### Project Management (6 tools)

| Tool | Description |
|------|-------------|
| `list_data_science_projects` | List all RHOAI projects |
| `get_project_details` | Get project with resource summary |
| `create_data_science_project` | Create new project |
| `delete_data_science_project` | Delete project (requires confirmation) |
| `get_project_status` | Get comprehensive project status |
| `set_model_serving_mode` | Set single vs multi-model serving |

### Workbench Management (8 tools)

| Tool | Description |
|------|-------------|
| `list_workbenches` | List workbenches in project |
| `get_workbench` | Get workbench details |
| `create_workbench` | Create new workbench |
| `start_workbench` | Start a stopped workbench |
| `stop_workbench` | Stop a running workbench |
| `delete_workbench` | Delete workbench |
| `list_notebook_images` | List available images |
| `get_workbench_url` | Get OAuth-protected URL |

### Model Serving (6 tools)

| Tool | Description |
|------|-------------|
| `list_inference_services` | List deployed models |
| `get_inference_service` | Get model details |
| `deploy_model` | Create InferenceService |
| `delete_inference_service` | Delete deployed model |
| `list_serving_runtimes` | List available runtimes |
| `get_model_endpoint` | Get inference endpoint URL |

### Data Connections (4 tools)

| Tool | Description |
|------|-------------|
| `list_data_connections` | List connections in project |
| `get_data_connection` | Get connection details (masked) |
| `create_s3_data_connection` | Create S3 connection |
| `delete_data_connection` | Delete connection |

### Pipelines (3 tools)

| Tool | Description |
|------|-------------|
| `get_pipeline_server` | Get DSPA status |
| `create_pipeline_server` | Create DSPA |
| `delete_pipeline_server` | Delete DSPA |

### Storage (3 tools)

| Tool | Description |
|------|-------------|
| `list_storage` | List PVCs in project |
| `create_storage` | Create PVC |
| `delete_storage` | Delete PVC (requires confirmation) |

### Quickstarts (5 tools)

| Tool | Description |
|------|-------------|
| `list_quickstarts` | List available quickstarts from the Red Hat AI quickstart registry |
| `get_quickstart_manifest` | Get a quickstart's full manifest, including its parameters |
| `run_quickstart_action` | Run a quickstart action (e.g. `INSTALL`, `UNINSTALL_*`) as an installer Job |
| `get_quickstart_action_status` | Get the status of a quickstart action (installer Job) |
| `get_quickstart_action_logs` | Get installer logs for a quickstart action (installer Job) |

> **Prerequisite:** Deployment actions run an installer Job under a dedicated
> ServiceAccount in a dedicated namespace. A cluster admin must provision these
> once by applying [`deploy/quickstarts/installer-rbac.yaml`](deploy/quickstarts/installer-rbac.yaml)
> — the MCP server does not create them automatically. See
> [Quickstart installer RBAC](#quickstart-installer-rbac).

## MCP Resources

The server also exposes read-only resources:

| Resource URI | Description |
|--------------|-------------|
| `rhoai://cluster/status` | Cluster health and RHOAI status |
| `rhoai://cluster/components` | DataScienceCluster component status |
| `rhoai://cluster/accelerators` | Available GPU profiles |
| `rhoai://projects/{name}/status` | Project resource summary |
| `rhoai://projects/{name}/workbenches` | Workbench list with status |
| `rhoai://projects/{name}/models` | Deployed models with status |

## MCP Prompts

The server provides 19 prompts that guide AI agents through multi-step workflows. Prompts are templates that provide step-by-step instructions and reference the appropriate tools for each workflow stage.

### Training Workflow (3 prompts)

| Prompt | Description |
|--------|-------------|
| `train-model` | Guide through fine-tuning a model with LoRA/QLoRA |
| `monitor-training` | Monitor an active training job and diagnose issues |
| `resume-training` | Resume a suspended or failed training job from checkpoint |

### Cluster Exploration (4 prompts)

| Prompt | Description |
|--------|-------------|
| `explore-cluster` | Discover what's available in the RHOAI cluster |
| `explore-project` | Explore resources within a specific Data Science Project |
| `find-gpus` | Find available GPU resources for training or inference |
| `whats-running` | Quick status check of all active workloads |

### Troubleshooting (4 prompts)

| Prompt | Description |
|--------|-------------|
| `troubleshoot-training` | Diagnose and fix issues with a training job |
| `troubleshoot-workbench` | Diagnose and fix issues with a workbench |
| `troubleshoot-model` | Diagnose and fix issues with a deployed model |
| `analyze-oom` | Analyze and resolve Out-of-Memory issues in training |

### Project Setup (3 prompts)

| Prompt | Description |
|--------|-------------|
| `setup-training-project` | Set up a new project for model training |
| `setup-inference-project` | Set up a new project for model serving |
| `add-data-connection` | Add an S3 data connection to an existing project |

### Model Deployment (4 prompts)

| Prompt | Description |
|--------|-------------|
| `deploy-model` | Deploy a model for inference serving |
| `deploy-llm` | Deploy a Large Language Model with vLLM or TGIS |
| `test-endpoint` | Test a deployed model endpoint |
| `scale-model` | Scale a model deployment up or down |

### Quickstart Deployment (1 prompt)

| Prompt | Description |
|--------|-------------|
| `deploy-quickstart` | Discover, configure and deploy a Red Hat AI quickstart (discover → manifest → elicit parameters → run → poll) |

## Example Interactions

### Create a Data Science Project

```
User: Create a new data science project called "fraud-detection" for my ML experiments
Agent: I'll create that project for you with a descriptive display name.
[Calls create_data_science_project with name="fraud-detection", 
 display_name="Fraud Detection ML", description="Machine learning experiments for fraud detection"]

Result: Project 'fraud-detection' created successfully
```

### Deploy a Model

```
User: Deploy my trained fraud model from S3 to the fraud-detection project

Agent: I'll deploy your model. First, let me check available serving runtimes.
[Calls list_serving_runtimes with namespace="fraud-detection"]
[Calls deploy_model with name="fraud-model", namespace="fraud-detection",
 runtime="ovms", model_format="onnx", 
 storage_uri="s3://models/fraud-detection/model.onnx"]

Result: Model 'fraud-model' deployment initiated. It may take a few minutes to become ready.
```

### Check Project Status

```
User: What's the status of my fraud-detection project?

Agent: [Calls get_project_status with namespace="fraud-detection"]

Result:
- Project: fraud-detection (Fraud Detection ML)
- Workbenches: 2 total, 1 running
- Models: 1 deployed, 1 ready
- Pipeline Server: Ready
- Data Connections: 1
- Storage: 3 PVCs
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=rhoai_mcp
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint
uv run ruff check

# Type check
uv run mypy src/rhoai_mcp
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                MCP Transport Layer (stdio/SSE/HTTP)             │
├─────────────────────────────────────────────────────────────────┤
│  FastMCP Server (server.py)                                     │
│  - Tool registration     - Resource registration                │
│  - Prompt registration   - Lifecycle management                 │
├───────────────────┬─────────────────────┬───────────────────────┤
│  Tools Layer      │  Resources Layer    │  Prompts Layer        │
│  - projects       │  - cluster.py       │  - training (3)       │
│  - notebooks      │  - projects.py      │  - exploration (4)    │
│  - inference      │                     │  - troubleshooting (4)│
│  - connections    │                     │  - project setup (3)  │
│  - storage        │                     │  - deployment (4)     │
│  - pipelines      │                     │                       │
│  - training       │                     │                       │
├───────────────────┴─────────────────────┴───────────────────────┤
│  Clients Layer (clients/) - Business Logic                      │
│  - base.py (K8sClient)   - projects.py    - notebooks.py        │
│  - inference.py          - connections.py - storage.py          │
│  - pipelines.py          - training.py                          │
├─────────────────────────────────────────────────────────────────┤
│  Models Layer (models/) - Pydantic Data Structures              │
│  - common.py (shared)    - Domain-specific models per resource  │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                           │
│  - K8sClient: Kubernetes API abstraction (Core + CRDs)          │
│  - Configuration: Environment-based settings                    │
│  - Plugin Manager: Pluggy-based plugin system                   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| **clients/** | Kubernetes client abstractions for each resource type |
| **models/** | Pydantic models for type-safe resource handling |
| **tools/** | MCP tool definitions that wrap client operations |
| **resources/** | MCP resource definitions for read-only data access |
| **utils/** | Helper functions for annotations, labels, and errors |

### Request Flow

```
AI Agent Request → MCP Transport → Tool Handler → Domain Client
                                                       ↓
AI Agent Response ← Pydantic Model ← K8s Response ← K8sClient → Kubernetes API
```

### Key CRDs Supported

| Resource | API Group | Purpose |
|----------|-----------|---------|
| Namespace | core/v1 | Data Science Projects |
| Notebook | kubeflow.org/v1 | Workbenches |
| InferenceService | serving.kserve.io/v1beta1 | Model serving |
| ServingRuntime | serving.kserve.io/v1alpha1 | Model server configs |
| DataSciencePipelinesApplication | datasciencepipelinesapplications.opendatahub.io/v1alpha1 | Pipeline infrastructure |
| AcceleratorProfile | dashboard.opendatahub.io/v1 | GPU profiles |

## License

MIT License - see [LICENSE](LICENSE) for details.
