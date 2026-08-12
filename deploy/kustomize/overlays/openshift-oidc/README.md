# OpenShift OIDC Overlay

Deploys the RHOAI MCP server with multi-user OIDC authentication on OpenShift.

> [!WARNING]
> This is not a supported overlay and it's being currently used for Development purposes.

## RBAC model

This overlay **replaces** the base ClusterRole entirely. Instead of granting the ServiceAccount direct access to resources (notebooks, inference services, secrets, etc.), it grants only:

| Permission | Purpose |
|---|---|
| `create` tokenreviews | Validate opaque bearer tokens via the K8s TokenReview API |
| `create` subjectaccessreviews | Filter MCP tools based on user RBAC |
| `get` users (user.openshift.io) | Fetch OCP group memberships for authenticated users |

All resource-level access is enforced by the Kubernetes API against the **user's own bearer token**. The ServiceAccount only needs permissions for token validation and RBAC filtering — it never accesses user resources directly.

## Deployment

```bash
# 1. Apply the overlay
kubectl kustomize deploy/kustomize/overlays/openshift-oidc | oc apply -f -

# 2. (Optional) Apply the NetworkPolicy to allow traffic to the Model Catalog
oc apply -f deploy/kustomize/overlays/openshift/networkpolicy.yaml

# 3. Wait for the pod to start
oc get pods -n rhoai-mcp -w
```

> **Note:** The NetworkPolicy targets the `rhoai-model-registries` namespace and must be applied separately because it lives outside this MCP app namespace.

## Configuration

The overlay enables OIDC with `token-review` mode via ConfigMap patches:

| Variable | Value | Description |
|---|---|---|
| `RHOAI_MCP_OIDC_ENABLED` | `true` | Enables OIDC authentication |
| `RHOAI_MCP_OIDC_TOKEN_MODE` | `token-review` | Validates tokens via K8s TokenReview API (suited for OpenShift opaque tokens) |
| `RHOAI_MCP_OIDC_KUBE_AUTH_STRATEGY` | `user-token` | Forwards the caller's bearer token to the K8s API (default). Set to `impersonation` to use SA credentials with Impersonate-* headers instead. |

> **Note:** The default `user-token` strategy works with OpenShift opaque OAuth tokens validated via `token-review` mode. For environments using JWT tokens from an external IdP, the `user-token` strategy is not supported — the server rejects the `user-token` + `jwt` combination at startup. Switch to `impersonation` strategy and add the impersonation ClusterRole rule manually.

Other OIDC settings (`oidc_audience`, `oidc_username_claim`, `oidc_groups_claim`, etc.) use sensible defaults and can be overridden via environment variables or a downstream overlay.

## Verifying the deployment

### 1. Health check

```bash
ROUTE=$(oc get route rhoai-mcp -n rhoai-mcp -o jsonpath='{.spec.host}')
curl -k https://$ROUTE/health
```

### 2. OIDC metadata is advertised

```bash
curl -k https://$ROUTE/.well-known/oauth-protected-resource
```

Should return a JSON document with `resource`, `authorization_servers`, etc.

### 3. Unauthenticated requests are rejected

```bash
curl -k -i -X POST https://$ROUTE/mcp
```

Should return `401` with a `WWW-Authenticate: Bearer` header.

### 4. Authenticated requests succeed

```bash
TOKEN=$(oc whoami -t)
curl -k -i -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}' \
  https://$ROUTE/mcp
```

Should return `200` with a JSON-RPC response containing server capabilities and an `mcp-session-id` header.

### 5. Tool execution and RBAC

Use, for example, the `fastmcp` CLI to list available tools and call them with the intended bearer token:

```bash
ROUTE=$(oc get route rhoai-mcp -n rhoai-mcp -o jsonpath='{.spec.host}')
TOKEN=$(oc whoami -t)

# List all tools
uvx fastmcp list "https://${ROUTE}/mcp" --transport http --auth "$TOKEN"

# Call a tool
uvx fastmcp call "https://${ROUTE}/mcp" --transport http --auth "$TOKEN" \
  --target list_data_science_projects --json

# Cluster overview
uvx fastmcp call "https://${ROUTE}/mcp" --transport http --auth "$TOKEN" \
  --target cluster_summary --json
```

The tool list should be filtered by your OCP permissions. Tools targeting namespaces you don't have access to should return 403.
