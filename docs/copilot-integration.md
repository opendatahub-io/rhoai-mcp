# Integrating rhoai-mcp with Microsoft Copilot

This guide explains how to connect Microsoft Copilot to a rhoai-mcp server so that
Copilot can manage Red Hat OpenShift AI (RHOAI) resources through natural language:
data science projects, workbenches, model serving, pipelines, data connections, and
model registries.

The repository documentation already covers Claude Desktop and Claude Code. Microsoft's
Copilot ecosystem is broader and each surface integrates with MCP differently, so this
guide starts with a short analysis of which surface to use, then gives step-by-step
instructions for the recommended paths.

---

## Choosing a Copilot surface

Microsoft ships several products named "Copilot". Their MCP support differs in
transport, authentication, and capability:

| Surface | Transports | Caller authentication | Tools | Resources | Prompts | Execution model |
|---------|-----------|----------------------|-------|-----------|---------|-----------------|
| **VS Code (GitHub Copilot Chat)** | stdio, HTTP (streamable HTTP and SSE) | Environment variables, custom headers, OAuth | Yes | Yes | Yes | User confirms tool calls in the editor |
| **GitHub.com Copilot (cloud agent, coding agent, code review)** | stdio (runs on the agent's runner), HTTP, SSE | Static headers or `COPILOT_MCP_*` secrets (no OAuth) | Yes | No | No | Autonomous: tools run without per-call confirmation |
| **Copilot Studio** | Remote HTTP (MCP connector) | Connector authentication | Yes | Yes | No | Agent-driven, depends on agent configuration |

**Recommendation:**

1. **VS Code with GitHub Copilot Chat is the primary integration path.** It supports
   all rhoai-mcp transports (including stdio for local use), exposes tools, resources,
   and prompts, and asks for confirmation before each tool call. It is the closest
   equivalent to the Claude Desktop and Claude Code experience.
2. **GitHub.com Copilot (cloud agent) can use rhoai-mcp** either as a deployed remote
   server or as a stdio server started on the agent's runner, with important
   limitations: it does not support OAuth-authenticated MCP servers, it ignores
   resources and prompts, and it executes tools autonomously.
   See [GitHub.com Copilot (cloud agent)](#githubcom-copilot-cloud-agent-and-code-review).
3. **Copilot Studio** is a viable future path for Microsoft 365-facing scenarios. It
   requires a Copilot Studio tenant with generative orchestration enabled and is out
   of scope for this guide.
4. **Other Microsoft Copilot surfaces** exist but are out of scope for this guide and
   have not been validated with rhoai-mcp.

---

## Prerequisites

- A current VS Code release with **GitHub Copilot** (chat is built in), signed in with
  a GitHub account that has Copilot access. Agent mode is required for tool use (ask/edits modes can
  chat but cannot call tools).
- Access to an OpenShift cluster with RHOAI installed, and `oc` or `kubectl` configured.
- macOS or Linux. The steps in this guide were validated on macOS; running rhoai-mcp
  locally on Windows has not been tested.
- One of:
  - a local checkout of this repository with [`uv`](https://docs.astral.sh/uv/)
    installed, or
  - `uvx` (comes with `uv`) to run the server directly from the repository, or
  - a deployed rhoai-mcp server reachable over HTTP (see the
    [OpenShift deployment](../README.md#openshift-deployment) section of the README).

---

## VS Code (GitHub Copilot Chat)

VS Code discovers MCP servers from a `mcp.json` file. The three options below differ
in how the server is launched and reached; pick one.

> **Format note:** VS Code uses a top-level `servers` object in `mcp.json`. This is
> different from the portable `mcpServers` format used by Claude Desktop and some
> other clients. The examples in this guide use the VS Code format.

### Option A: stdio with a locally running server (simplest)

VS Code launches and manages the rhoai-mcp process as a subprocess. No port is
occupied and no server terminal is needed.

Create `.vscode/mcp.json` in your workspace (or add the entry to your user profile
`mcp.json` to share it across workspaces):

```json
{
  "servers": {
    "rhoai-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/opendatahub-io/rhoai-mcp", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

For development against a local clone, replace the `command` and `args` with:

```json
      "command": "uv",
      "args": ["run", "rhoai-mcp"],
      "cwd": "/path/to/rhoai-mcp"
```

> **Note:** the `git+https://...` source tracks the default branch. For reproducible
> setups, pin a tag or commit, for example
> `git+https://github.com/opendatahub-io/rhoai-mcp@v0.1.1`.

Then:

1. Log in to your cluster so the server can pick up credentials:
   ```sh
   oc login --server=https://api.cluster.example.com:6443
   ```
2. Open Copilot Chat, switch to **Agent** mode, and ask:
   *"List my data science projects."*

The server starts on demand with the first chat message; there is no need to start
it manually. (You can check its state with **MCP: List Servers** from the Command
Palette. Starting it manually from that view can fail with "Cannot start an MCP
server before the session runtime has been created" if no chat session exists yet;
sending a message first avoids the error.)

With the default `RHOAI_MCP_AUTH_MODE=auto`, the server authenticates to the cluster
using your current kubeconfig (the `oc login` from step 1). No token is written to
any configuration file.

### Option B: local streamable HTTP

Run the server yourself and point VS Code at its HTTP endpoint. This exercises the
same code path as remote deployments, which makes it useful for testing before you
deploy.

1. Start the server with HTTP transport:
   ```sh
   uvx --from git+https://github.com/opendatahub-io/rhoai-mcp \
       rhoai-mcp --transport streamable-http
   ```
   The server listens on `127.0.0.1:8000` by default (configure with
   `RHOAI_MCP_HOST` and `RHOAI_MCP_PORT`).
2. Add the server to `.vscode/mcp.json`:
   ```json
   {
     "servers": {
       "rhoai-mcp": {
         "type": "http",
         "url": "http://127.0.0.1:8000/mcp"
       }
     }
   }
   ```
3. VS Code picks up the new entry automatically. Use it from Agent mode as above.

### Option C: deployed server (remote HTTP)

Connect to a rhoai-mcp server deployed on OpenShift. Obtain the route URL:

```sh
oc get route rhoai-mcp -n rhoai-mcp -o jsonpath='{.spec.host}'
```

`.vscode/mcp.json`:

```json
{
  "servers": {
    "rhoai-mcp": {
      "type": "http",
      "url": "https://<RHOAI_MCP_ROUTE>/mcp",
      "headers": {
        "Authorization": "Bearer ${env:OCP_TOKEN}"
      }
    }
  }
}
```

Start VS Code with the token in the environment:

```sh
OCP_TOKEN=$(oc whoami -t) code .
```

The `Authorization` header is only required when the deployment enforces caller
authentication (the OpenShift OIDC overlay). Without such a deployment, omit the
`headers` block. Never paste a token literal into `mcp.json` or commit it to a
repository; the `${env:...}` substitution reads it from your shell environment.

> **Note:** VS Code evaluates `${env:...}` when the server starts, so VS Code must be
> launched with the variable set. Tokens expire, so when you need a fresh one,
> quit VS Code and relaunch it with the new `OCP_TOKEN`.
>
> On macOS, launch VS Code through the `code` command so it inherits your shell
> environment. Launching through `open -a "Visual Studio Code"` or from the Dock
> does not pass `OCP_TOKEN` to VS Code.

### Talking to your cluster

Once connected, Copilot Agent mode can call any rhoai-mcp tool. Examples:

- *"What data science projects do I have and which are running workbenches?"*
- *"Create a data science project called llm-demo with a workbench using the PyTorch
  image."*
- *"List the registered models in the model registry."*
- *"Deploy an InferenceService in project llm-demo using this YAML: ..."*

Copilot shows each tool call and asks for confirmation before executing it (Agent
mode default).

### Model Registry from a local server

When running the server locally (Options A and B), Model Registry auto-discovery
cannot reach the in-cluster service. Port-forward it and configure the server:

```sh
oc port-forward -n rhoai-model-registries svc/model-catalog 18443:8443
```

Add to the `env` block of the stdio server entry (or export before starting the
HTTP server):

```json
{
  "RHOAI_MCP_MODEL_REGISTRY_URL": "https://localhost:18443",
  "RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE": "manual",
  "RHOAI_MCP_MODEL_REGISTRY_AUTH_MODE": "oauth",
  "RHOAI_MCP_MODEL_REGISTRY_SKIP_TLS_VERIFY": "true"
}
```

`SKIP_TLS_VERIFY` is needed because the port-forward presents the service's
self-signed certificate. See the
[Model Registry](../README.md#model-registry) section of the README for the full
list of settings.

---

## GitHub.com Copilot (cloud agent and code review)

The Copilot experiences on github.com (the cloud agent, the coding agent, and code
review) use MCP servers configured in **repository settings**
(**Settings → Copilot → MCP servers**, repository administrators only). The same
configuration is shared by the cloud agent and code review. Organization owners must
allow MCP servers in Copilot for the repository (the policy is disabled by default
for Copilot Business and Enterprise).

Unlike VS Code, the configuration uses a top-level `mcpServers` object, and servers
can be local (`stdio`) or remote (`http`, `sse`). Credentials are supplied as
**Agents secrets or variables** whose names start with `COPILOT_MCP_`; reference
them in the configuration as `$COPILOT_MCP_NAME`. Use the `tools` array to allowlist
the tools Copilot may call (`"*"` enables all of them).

There are two ways to connect rhoai-mcp. Neither has been tested end to end yet.

**Remote server.** Point Copilot at a deployed rhoai-mcp. The URL must be reachable
from GitHub's infrastructure, for example an OpenShift Route:

```json
{
  "mcpServers": {
    "rhoai-mcp": {
      "type": "http",
      "url": "https://<RHOAI_MCP_ROUTE>/mcp",
      "headers": {
        "Authorization": "Bearer $COPILOT_MCP_RHOAI_TOKEN"
      },
      "tools": ["list_data_science_projects", "get_project_details", "project_summary"]
    }
  }
}
```

**Local (stdio) server.** Copilot starts rhoai-mcp on the agent's runner, which then
calls your cluster's Kubernetes API directly. This needs a network path from the
runner to the API server and cluster credentials supplied as secrets, with the
server in `token` auth mode:

```json
{
  "mcpServers": {
    "rhoai-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/opendatahub-io/rhoai-mcp@v0.1.1", "rhoai-mcp"],
      "env": {
        "RHOAI_MCP_AUTH_MODE": "token",
        "RHOAI_MCP_API_SERVER": "$COPILOT_MCP_RHOAI_API_SERVER",
        "RHOAI_MCP_API_TOKEN": "$COPILOT_MCP_RHOAI_API_TOKEN",
        "RHOAI_MCP_READ_ONLY_MODE": "true"
      },
      "tools": ["list_data_science_projects", "get_project_details", "project_summary"]
    }
  }
}
```

`uv` is not preinstalled on the agent's runner, so this option may also need a
`copilot-setup-steps.yml` workflow that installs it.

To verify either setup, assign an issue to Copilot, open the session logs, and expand
the **Start MCP Servers** step to see the tools it discovered.

**Limitations to plan around:**

- **No OAuth flow.** These surfaces do not support MCP servers that require an
  interactive OAuth sign-in, so authentication must be a static header. The OpenShift
  OIDC overlay validates each request's bearer token through Kubernetes TokenReview,
  so a dedicated ServiceAccount token supplied in a static `Authorization` header
  should be accepted (this combination has not been tested). Treat that token as a
  long-lived credential: scope its RBAC narrowly and rotate it.
- **Tools only.** Resources and prompts exposed by rhoai-mcp are ignored; only
  tools are available.
- **Autonomous execution.** The cloud agent runs tools without asking for
  confirmation. Combined with a static token, this makes the `tools` allowlist and
  read-only mode strongly recommended (see below).
- **Code review requires `readOnlyHint`-annotated tools.** Copilot code review only
  invokes tools whose definitions declare `annotations.readOnlyHint: true`; mutating
  tools are excluded automatically. Note that current rhoai-mcp releases do not yet
  annotate their tools, so code review sees no rhoai-mcp tools at all until
  annotations are added (tracked as a follow-up). The cloud agent and coding agent
  are not affected by this and can use all tools.

---

## Authentication and transport reference

rhoai-mcp involves two independent authentication layers. Keeping them apart
avoids the most common configuration mistakes:

1. **Server to cluster** (`RHOAI_MCP_AUTH_MODE`): how the rhoai-mcp process
   authenticates to the Kubernetes API. Modes: `auto` (default, in-cluster first,
   then kubeconfig), `kubeconfig`, and `token` (explicit API server URL and token).
2. **Client to server**: how Copilot authenticates to rhoai-mcp. Locally over
   stdio there is none (the subprocess is trusted). Over HTTP, a deployed server
   may enforce caller authentication (the OpenShift OIDC overlay validates each
   request's bearer token); VS Code supplies it via the `Authorization` header.

| Scenario | Transport | `RHOAI_MCP_AUTH_MODE` | Caller auth |
|----------|-----------|----------------------|-------------|
| VS Code, local server | stdio | `auto` or `kubeconfig` (after `oc login`) | None |
| VS Code, local server | streamable HTTP | `auto` or `kubeconfig` | None |
| VS Code, deployed server | streamable HTTP | `auto` (in-cluster) | `Authorization` header if OIDC overlay is deployed |
| GitHub.com Copilot, remote server | remote HTTP | `auto` (in-cluster) | Static bearer token in configured headers (for example a ServiceAccount token when the OIDC overlay is deployed; untested) |
| GitHub.com Copilot, local server on the runner | stdio | `token` (API server URL and token from `COPILOT_MCP_*` secrets) | None (the runner subprocess is trusted) |
| Copilot Studio | remote HTTP | `auto` (in-cluster) | Connector authentication |

Transport selection for the server itself:

```sh
# stdio (default)
rhoai-mcp

# HTTP transports
rhoai-mcp --transport streamable-http   # recommended for HTTP
rhoai-mcp --transport sse               # legacy HTTP transport
```

Note that OIDC caller authentication (the OpenShift overlay) requires an HTTP
transport and is rejected with stdio.

---

## Security recommendations

- **Start read-only.** For initial rollouts and any connection used by an
  autonomous surface, run the server with `RHOAI_MCP_READ_ONLY_MODE=true`. This
  disables all create, update, and delete operations at the tool level while
  still allowing listing and inspection.
- **Restrict tools where the platform supports it.** The GitHub.com repository
  settings for Copilot support allowlisting individual tools; expose only what the
  use case needs. VS Code has no config-level tool allowlist (`mcp.json` has no
  supported `tools` field), and its per-session tool selection in the chat UI is a
  client convenience, not an enforcement boundary. For reliable, client-independent
  tool restriction, use the server-side controls: `RHOAI_MCP_READ_ONLY_MODE=true`
  and Kubernetes RBAC on the identity the server (or, with the OIDC overlay, each
  caller) uses.
- **Never embed tokens in configuration files.** Use `${env:...}` substitution in
  VS Code `mcp.json`, keep `.env` files out of version control, and do not paste
  credentials into chat.
- **Scope static tokens used with GitHub.com Copilot.** They are long-lived
  credentials executed against autonomously. Prefer a service account or user
  with minimal RBAC, and prefer read-only tool sets.
- **Rely on rhoai-mcp's built-in safety gates.** Deletes are scoped to resources
  created by rhoai-mcp unless `RHOAI_MCP_ENABLE_DANGEROUS_OPERATIONS=true` is set,
  delete tools require an explicit `confirm` parameter, and access is ultimately
  bounded by the RBAC of the identity the server uses (or, with the OIDC overlay,
  of each caller).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `rhoai-mcp` shows Stopped in **MCP: List Servers** | VS Code did not detect `mcp.json` | Open VS Code from the directory containing `.vscode/mcp.json`, then reload the window |
| Manual server start fails with "Cannot start an MCP server before the session runtime has been created" | Start was clicked before any chat session existed | Send an Agent-mode chat message first; the server starts on demand |
| Tools/list works but cluster calls fail with auth errors | Server has no cluster credentials | Run `oc login`, then reload the VS Code window so the server restarts |
| Remote route returns 401 in chat after relaunching VS Code | `${env:OCP_TOKEN}` was not set in VS Code's environment (common on macOS when launched via `open -a` or the Dock) | Launch VS Code with the `code` CLI from a shell where `OCP_TOKEN` is exported |
| "TLS certificate is self-signed" on Model Registry calls | Port-forward presents the service certificate | Set `RHOAI_MCP_MODEL_REGISTRY_SKIP_TLS_VERIFY=true` |
| Model Registry tools report wrong endpoint or fail discovery | Auto-discovery cannot reach the in-cluster service | Set `RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE=manual` and `RHOAI_MCP_MODEL_REGISTRY_URL` to the port-forward or route URL |
| Cloud agent cannot reach the server | Route not publicly reachable, or org policy disables MCP | Check route exposure and the organization's "MCP servers in Copilot" policy |
| Tools missing in code review results | Tool lacks `readOnlyHint: true` | Expected for mutating tools; code review only uses read-only tools |

---

## Other Copilot surfaces

- **Copilot Studio** connects to remote MCP servers through its MCP onboarding
  wizard and makes the server's tools and resources available to agents with
  generative orchestration enabled. It suits Microsoft 365-facing, business-user
  scenarios. Like the remote option for GitHub.com Copilot, it needs an HTTPS
  endpoint reachable from the service, plus a Copilot Studio tenant. It has not
  been validated with rhoai-mcp.

---

## References

- [VS Code documentation: MCP servers](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [GitHub documentation: Use MCP with Copilot](https://docs.github.com/en/copilot/concepts/about-mcp)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [rhoai-mcp README](../README.md)
