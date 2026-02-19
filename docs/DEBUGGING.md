# Debugging the RHOAI MCP Server

This guide covers how to diagnose and debug errors reported by users of the RHOAI MCP server.

## Gathering Information from the User

Before diving in, collect these details from the user:

- The exact error message or traceback
- Which tool or operation they were calling (and the parameters used)
- Their transport type (`stdio`, `sse`, or `streamable-http`)
- How they are running the server (container, local `uv run`, Claude Desktop, etc.)
- Whether the error is reproducible or intermittent

## Enable DEBUG Logging

All server logs are written to **stderr** in the format:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

Enable verbose logging to see the full picture:

**Local:**
```bash
RHOAI_MCP_LOG_LEVEL=DEBUG uv run rhoai-mcp
```

**CLI flag:**
```bash
uv run rhoai-mcp --log-level DEBUG
```

**Container (using the dev target):**
```bash
make run-dev   # Sets LOG_LEVEL=DEBUG and ENABLE_DANGEROUS_OPERATIONS=true
```

Each domain module uses its own named logger (`logging.getLogger(__name__)`), so the `%(name)s` field identifies which module produced each message.

## Exception Hierarchy

The server uses structured exceptions defined in `src/rhoai_mcp/utils/errors.py`. All inherit from `RHOAIError` and carry a `message` string plus a `details` dict with structured metadata (resource type, name, namespace, etc.).

| Exception | HTTP Status | Meaning | Typical Cause |
|-----------|------------|---------|---------------|
| `AuthenticationError` | 401/403 | K8s auth failed | Expired `oc login` session or kubeconfig token |
| `NotFoundError` | 404 | Resource does not exist | Wrong namespace, typo in name, resource was deleted |
| `ResourceExistsError` | 409 | Resource already exists | Trying to create a duplicate |
| `OperationNotAllowedError` | n/a | Safety check blocked the call | `READ_ONLY_MODE=true` or `ENABLE_DANGEROUS_OPERATIONS=false` |
| `ConfigurationError` | n/a | Server config is invalid | Missing or conflicting environment variables |
| `ValidationError` | n/a | Bad input to a tool | Invalid parameters from the calling agent |

The K8s base client (`src/rhoai_mcp/clients/base.py`) translates raw `ApiException` status codes into these domain exceptions automatically.

## Common Failure Scenarios

### Server won't start

**Symptoms:** Server exits immediately with a non-zero exit code.

**Check:**
1. **Auth config** — Run `oc whoami` or `kubectl cluster-info` to verify the cluster is reachable.
2. **Kubeconfig path** — If using `AUTH_MODE=kubeconfig`, verify `RHOAI_MCP_KUBECONFIG_PATH` points to an existing file.
3. **Startup logs** — The server validates auth config early (`config.py: validate_auth_config()`). Look for `ConfigurationError` or `AuthenticationError` in the output.

The server has special handling for expired credentials at startup (`__main__.py`). It catches `AuthenticationError` (including inside `ExceptionGroup`) and prints:

```
Kubernetes authentication failed. Your credentials may be expired.
Try re-authenticating with: oc login / kubectl config set-credentials
```

### Tool returns "not found"

**Check:**
1. Verify the namespace is correct — the user may be targeting the wrong project.
2. Verify the resource name — check for typos with `kubectl get <resource> -n <namespace>`.
3. Verify the CRD is installed — some resource types (training jobs, pipelines) require operator CRDs. If the CRD is missing, the plugin will have failed its health check at startup.

### Tool returns "operation not allowed"

**Check these config settings:**
- `RHOAI_MCP_READ_ONLY_MODE` — If `true`, all create/update/delete operations are blocked.
- `RHOAI_MCP_ENABLE_DANGEROUS_OPERATIONS` — If `false` (the default), delete operations are blocked.

The `is_operation_allowed()` method on the config object controls this. Each tool checks it before performing writes.

### Tool hangs or times out

**Check:**
- K8s API reachability — Try `kubectl get nodes` from the same environment.
- Network/VPN — The server may be unable to reach the cluster API endpoint.
- For Model Registry tools specifically, check whether the registry service is accessible (see the Model Registry section in `ARCHITECTURE.md`).

### Intermittent failures

Enable DEBUG logging to capture detailed request/response patterns:

```bash
RHOAI_MCP_LOG_LEVEL=DEBUG uv run rhoai-mcp
```

Each domain module uses its own named logger, so the `%(name)s` field in the log output identifies which module produced each message. Look for patterns in timing, specific namespaces, or resource types that correlate with the failures.

## Health Check Endpoint

When running with HTTP transport (`sse` or `streamable-http`), the server exposes a health endpoint:

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "connected": true,
  "plugins": {
    "total": 12,
    "healthy": 12
  }
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "connected": false,
  "plugins": {
    "total": 12,
    "healthy": 0
  }
}
```

A 503 means the server cannot reach the Kubernetes API.

## Plugin Health Checks

At startup, each plugin runs a health check via the `rhoai_health_check` hook (`plugin_manager.py`). Plugins that fail are **skipped** rather than blocking the entire server (graceful degradation).

Look for these log messages:

```
Plugin {name} health check failed with error: {e}
```

And the summary line:

```
X/Y plugins active
```

If a plugin is missing, its required CRDs are likely not installed on the cluster. You can also run:

```bash
make test-plugins
```

to list all discovered plugins outside of a running server.

## Built-in Diagnostic Tools

The server provides several composite tools for runtime diagnostics. These can be called by the AI agent or invoked manually through an MCP client.

| Tool | Purpose |
|------|---------|
| `cluster_summary()` | Quick cluster health overview |
| `project_summary(namespace)` | Summary of a single project |
| `explore_cluster(include_health=True)` | Full cluster scan with issue detection |
| `diagnose_resource(resource_type, name, namespace)` | Deep dive on a specific resource — checks status, events, logs, and suggests fixes |
| `resource_status(resource_type, name, namespace)` | Minimal status check for any resource |
| `analyze_training_failure(namespace, job_name)` | Training-specific failure analysis |
| `get_training_logs(namespace, job_name, previous=True)` | Container logs (use `previous=True` for crashed pods) |
| `get_job_events(namespace, job_name)` | Raw Kubernetes events for a training job |
| `get_training_progress(namespace, job_name)` | Real-time training metrics |

### Troubleshooting Prompts

The server also includes guided troubleshooting prompts (`src/rhoai_mcp/domains/prompts/troubleshooting_prompts.py`) that walk AI agents through diagnostic workflows:

- **`troubleshoot-training`** — Guides through checking job status, events, logs, and common failure patterns (OOMKilled, ImagePullBackOff, FailedScheduling).
- **`troubleshoot-workbench`** — Guides through checking workbench pods, events, and container logs.

## Configuration Reference

Key environment variables relevant to debugging (all use the `RHOAI_MCP_` prefix):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AUTH_MODE` | `auto` | Authentication mode: `auto`, `kubeconfig`, `token` |
| `KUBECONFIG_PATH` | `~/.kube/config` | Path to kubeconfig file |
| `KUBECONFIG_CONTEXT` | (current) | Kubeconfig context to use |
| `READ_ONLY_MODE` | `false` | Disable all write operations |
| `ENABLE_DANGEROUS_OPERATIONS` | `false` | Enable delete operations |
| `DEFAULT_VERBOSITY` | `standard` | Response detail level: `minimal`, `standard`, `full` |
| `ENABLE_RESPONSE_CACHING` | `false` | Cache list responses |
| `MODEL_REGISTRY_SKIP_TLS_VERIFY` | `false` | Skip TLS for Model Registry |

## Quick Triage Checklist

1. **Server won't start** — Check auth (`oc whoami`), check kubeconfig path, read startup logs.
2. **Tool returns "not found"** — Verify namespace, resource name, and that the CRD is installed.
3. **Tool returns "operation not allowed"** — Check `READ_ONLY_MODE` and `ENABLE_DANGEROUS_OPERATIONS`.
4. **Tool hangs or times out** — Check K8s API reachability (`kubectl get nodes`), check network/VPN.
5. **Plugin missing** — Check health endpoint or startup logs for failed health checks; CRD may not be installed.
6. **Intermittent failures** — Enable DEBUG logging to capture patterns.
