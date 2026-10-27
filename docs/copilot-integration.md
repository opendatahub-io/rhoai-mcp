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
| **GitHub.com Copilot (cloud agent, coding agent, code review)** | Remote HTTP only | Custom headers only (no OAuth) | Yes | No | No | Autonomous: tools run without per-call confirmation |
| **Copilot Studio** | Remote HTTP (MCP connector) | Connector authentication | Yes | Yes | No | Agent-driven, depends on agent configuration |
| Windows Copilot | Local stdio (connected apps) | Local | Yes | Limited | No | User confirms |
| Copilot Extensions / Semantic Kernel | You build or embed the client | Code | n/a | n/a | n/a | Programmatic |

**Recommendation:**

1. **VS Code with GitHub Copilot Chat is the primary integration path.** It supports
   all rhoai-mcp transports (including stdio for local use), exposes tools, resources,
   and prompts, and asks for confirmation before each tool call. It is the closest
   equivalent to the Claude Desktop and Claude Code experience.
2. **GitHub.com Copilot (cloud agent) works against a deployed rhoai-mcp server** over
   HTTP, with important limitations: it does not support OAuth-authenticated MCP
   servers, it ignores resources and prompts, and it executes tools autonomously.
   See [GitHub.com Copilot (cloud agent)](#githubcom-copilot-cloud-agent-and-code-review).
3. **Copilot Studio** is a viable future path for Microsoft 365-facing scenarios. It
   requires a Copilot Studio tenant with generative orchestration enabled and is out
   of scope for this guide.
4. **Copilot Extensions and Semantic Kernel** are development platforms for building
   your own agent or publishing an app on the GitHub Marketplace. They are not needed
   to consume rhoai-mcp from Copilot and are not covered here.

---

## Prerequisites

- VS Code with the **GitHub Copilot Chat** extension, signed in with a GitHub account
  that has Copilot access. Agent mode is required for tool use (ask/edits modes can
  chat but cannot call tools).
- Access to an OpenShift cluster with RHOAI installed, and `oc` or `kubectl` configured.
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

Then:

1. Log in to your cluster so the server can pick up credentials:
   ```sh
   oc login --server=https://api.cluster.example.com:6443
   ```
2. Open the Command Palette (**Cmd+Shift+P** / **Ctrl+Shift+P**) and run
   **MCP: List Servers**. `rhoai-mcp` should appear; if it shows **Stopped**, start it
   from the `...` menu.
3. Open Copilot Chat, switch to **Agent** mode, and verify the connection by asking:
   *"List my data science projects."*

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
3. Start or restart the server from **MCP: List Servers** and use it from Agent mode
   as above.

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
> launched with the variable set. If you change the token later, restart the MCP
> server (**MCP: Restart Server**).

### Talking to your cluster

Once connected, Copilot Agent mode can call any rhoai-mcp tool. Examples:

- *"What data science projects do I have and which are running workbenches?"*
- *"Create a data science project called llm-demo with a workbench using the PyTorch
  image."*
- *"List the registered models in the model registry."*
- *"Deploy an InferenceService in project llm-demo using this YAML: ..."*

Copilot shows each tool call and asks for confirmation before executing it (Agent
mode default). Use the `#rhoai-mcp` reference in chat to force a specific server's
tools.

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
review) can call remote MCP servers configured in **repository settings**
(**Settings → Copilot → MCP servers**). These surfaces run in GitHub's
infrastructure, so they can only reach servers that are publicly reachable over
HTTPS: a deployed rhoai-mcp behind an OpenShift Route, or an equivalent internal
deployment exposed through your network policy.

To add the server, provide its URL (for example
`https://<RHOAI_MCP_ROUTE>/mcp`) and, if the deployment requires it, a custom
`Authorization` header with a static bearer token. Organization owners must allow
MCP servers in Copilot for the repository (the policy is disabled by default for
Copilot Business and Enterprise).

**Limitations to plan around:**

- **No OAuth.** These surfaces do not support OAuth-authenticated MCP servers.
  A rhoai-mcp deployment using the OpenShift OIDC overlay (which validates each
  caller's token) cannot be used here. The server must be deployed without caller
  OIDC, or fronted by a gateway that accepts a static token you configure in the
  repository settings. Treat that token as a long-lived credential: scope it
  narrowly and rotate it.
- **Tools only.** Resources and prompts exposed by rhoai-mcp are ignored; only
  tools are available.
- **Autonomous execution.** The cloud agent runs tools without asking for
  confirmation. Combined with a static token, this makes tool allowlisting and
  read-only mode strongly recommended (see below).
- **Code review requires read-only tools.** Copilot code review only invokes
  tools annotated with `readOnlyHint: true`; mutating tools are excluded
  automatically.

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
| GitHub.com Copilot | remote HTTP | `auto` (in-cluster) | Static token in configured headers; OIDC overlay not supported |
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
- **Allowlist tools.** Both VS Code and the GitHub.com repository settings let you
  restrict which tools a server exposes to Copilot. Expose only what the use case
  needs.
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
| Tools/list works but cluster calls fail with auth errors | Server has no cluster credentials | Run `oc login`, then restart the MCP server (**MCP: Restart Server**) |
| "TLS certificate is self-signed" on Model Registry calls | Port-forward presents the service certificate | Set `RHOAI_MCP_MODEL_REGISTRY_SKIP_TLS_VERIFY=true` |
| Model Registry tools report wrong endpoint or fail discovery | Auto-discovery cannot reach the in-cluster service | Set `RHOAI_MCP_MODEL_REGISTRY_DISCOVERY_MODE=manual` and `RHOAI_MCP_MODEL_REGISTRY_URL` to the port-forward or route URL |
| VS Code asks for sign-in on the MCP server | Server requires OAuth and VS Code attempts the flow | Deployments using static bearer auth: cancel and use the `Authorization` header instead |
| Cloud agent cannot reach the server | Route not publicly reachable, or org policy disables MCP | Check route exposure and the organization's "MCP servers in Copilot" policy |
| Tools missing in code review results | Tool lacks `readOnlyHint: true` | Expected for mutating tools; code review only uses read-only tools |

---

## Other Copilot surfaces

- **Copilot Studio** connects to remote MCP servers through its MCP onboarding
  wizard and makes the server's tools and resources available to agents with
  generative orchestration enabled. It suits Microsoft 365-facing, business-user
  scenarios. The same server requirements as GitHub.com Copilot apply (publicly
  reachable HTTPS endpoint), plus a Copilot Studio tenant.
- **Windows Copilot** supports local MCP servers via connected apps in agent mode;
  the stdio configuration is analogous to the VS Code Option A setup.
- **Copilot Extensions** (GitHub Apps that implement MCP over HTTPS) and
  **Semantic Kernel / Agent Framework** (programmatic MCP clients) are for
  building your own products on top of MCP rather than consuming rhoai-mcp as an
  end user.

---

## References

- [VS Code documentation: MCP servers](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)
- [GitHub documentation: Use MCP with Copilot](https://docs.github.com/en/copilot/concepts/about-mcp)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [rhoai-mcp README](../README.md)
