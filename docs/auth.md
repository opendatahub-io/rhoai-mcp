# Authentication Architecture

This note explains how rhoai-mcp validates bearer tokens relative to the
[MCP authorization model](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
— in particular why **TokenReview mode** is the OpenShift-native equivalent of the
spec's RFC 7662 opaque-token pathway, and why no standard introspection endpoint
exists to point at instead. It exists so that "we use TokenReview instead of
RFC 7662 Token Introspection" does not have to be re-derived from first principles
in every interoperability review, gateway certification, or contribution discussion.

## Overview: one middleware, two validation modes

A deployment picks exactly one of two mutually exclusive modes via
`RHOAI_MCP_OIDC_TOKEN_MODE` (`jwt` | `token-review`). Both feed the same
pure-ASGI `OIDCAuthMiddleware`, which extracts the Bearer token, validates it
through the configured validator, populates the request's `UserContext`, and
answers `401 + WWW-Authenticate: Bearer resource_metadata="<url>"` on failure
(the RFC 9728 reference to the metadata document served at
`/.well-known/oauth-protected-resource`).

| | JWT mode (`jwt`, default) | TokenReview mode (`token-review`) |
|---|---|---|
| Token format | Signed JWT (RS256/ES256) | Opaque string (OCP OAuth token) |
| Validator | `OIDCValidator` | `TokenReviewValidator` |
| Validation | Signature via the issuer's JWKS; `iss` and `aud` claims required | `TokenReview` against the Kubernetes API server |
| Spec pathway | The spec's JWT pathway | The spec's RFC 7662 opaque-token pathway (see below) |
| Typical deployment | External OIDC identity provider | OpenShift / in-cluster (the OCP deployment model) |

## JWT mode: the spec's JWT pathway

`OIDCValidator` discovers the issuer's configuration and JWKS from
`{issuer}/.well-known/openid-configuration`, then validates each token with
PyJWT: signature against the provider's keys, `issuer` against the configured
issuer, `audience` against the configured audience, with `exp`/`iss`/`aud`
required. This is the straightforward mapping to the MCP spec's JWT pathway —
nothing platform-specific to explain.

**Audience guidance (RFC 8707 relationship):** set `RHOAI_MCP_OIDC_AUDIENCE` to
this deployment's own resource URL — the same value advertised as `resource` in
the RFC 9728 metadata (`RHOAI_MCP_OIDC_RESOURCE_URL`, e.g. the OpenShift Route
URL). The `aud` check is the resource-server side of RFC 8707 Resource
Indicators: the authorization server mints `aud` = the `resource` the client
requested, and the server rejects tokens minted for a different resource. With a
per-deployment audience this closes the cross-deployment confused-deputy window;
with the generic default (`"rhoai-mcp"`) any two deployments sharing an issuer
family accept each other's tokens, so treat the default as a development
convenience only.

## TokenReview mode: the RFC 7662 equivalent

The MCP spec's authorization model supports opaque tokens validated by
introspection (RFC 7662): the resource server presents the token to an
introspection endpoint and receives an assertion of its validity and the
associated identity. OpenShift has no RFC 7662-shaped endpoint — and it should
not need one:

**On OpenShift, the OAuth authorization server *is* the cluster.** Its native
introspection API is `authentication.k8s.io/v1/tokenreviews`: the resource
server POSTs the opaque token and receives back the authenticated identity.
The semantics map field-for-field onto RFC 7662's introspection response:

| RFC 7662 introspection response | K8s TokenReview response |
|---|---|
| `active: true` | `status.authenticated: true` |
| `active: false` (revoked/expired/unknown) | `status.authenticated: false` |
| `username` | `status.user.username` |
| `sub` | `status.user.uid` |
| `groups` / `scope` | `status.user.groups` (plus OCP User groups, merged) |
| client authentication to the introspection endpoint | the server's ServiceAccount credentials (RBAC: `create tokenreviews`) |

Why no standard endpoint exists, and why that is correct: an RFC 7662 facade in
front of TokenReview would add a URL shape without changing a single semantic —
the same actor (the cluster) makes the same decision (is this token live, and
whose is it) through the same authority (the API server's token validation).
TokenReview *is* the introspection mechanism for the platform that issued the
token; wrapping it would be interop theater. The MCP spec requires the resource
server to validate tokens by an appropriate mechanism for its authorization
server — it does not require that mechanism to be an HTTP POST to an
RFC 7662 URL when the authorization server is the Kubernetes API itself.

**Why no major implementer documents this pattern:** MCP servers that
authenticate users generally target standard OAuth 2.1 identity providers with
JWT access tokens (Cloudflare, GitHub, and Google implementations all do).
A production MCP server consuming platform-native opaque tokens via
TokenReview is, to our knowledge, unique to rhoai-mcp — correct and necessary
for the OpenShift deployment model, but with no reference implementation to
cite. This note is that citation.

### Security properties of TokenReview mode

- **Cluster scoping** is the token's boundary: the API server that issued the
  token is the only one that will validate it. A token cannot be replayed
  against a different cluster.
- **Per-caller RBAC** is enforced on every tool call regardless of token
  validity: a valid token authorizes the caller to act *as themselves*, within
  their own Kubernetes permissions. This is a damage bound (no privilege
  escalation), not a confused-deputy defense — audience restriction plays that
  role for JWT mode, and has no equivalent (or need) for cluster-scoped tokens.
- **Audit and TTL** bound residual risk: TokenReview calls are audited in the
  API server log, and tokens expire per the platform's OAuth configuration.

## Configuration reference

| Variable | Mode | Meaning |
|---|---|---|
| `RHOAI_MCP_OIDC_ENABLED` | both | Enable the auth middleware |
| `RHOAI_MCP_OIDC_TOKEN_MODE` | both | `jwt` (default) or `token-review` |
| `RHOAI_MCP_OIDC_ISSUER_URL` | jwt | OIDC issuer (discovery + `iss` validation) |
| `RHOAI_MCP_OIDC_AUDIENCE` | jwt | Expected `aud` — set to this deployment's resource URL (see above) |
| `RHOAI_MCP_OIDC_REQUIRED_SCOPES` | both | Advertised in the RFC 9728 metadata |
| `RHOAI_MCP_OIDC_RESOURCE_URL` | both | This server's external URL (the RFC 9728 `resource`) |
| `RHOAI_MCP_OIDC_KUBE_AUTH_STRATEGY` | both | `user-token` (forward caller's token to K8s) or `impersonation` (SA + `Impersonate-*` headers) |
| `RHOAI_MCP_OIDC_OCP_API_URL` | token-review | API server URL for the metadata document |

The server's own permissions in `token-review` mode: `create` on
`authentication.k8s.io/tokenreviews`, `create` on
`authorization.k8s.io/subjectaccessreviews` (per-caller RBAC checks), and —
when using the impersonation strategy — `impersonate` on
users/groups/serviceaccounts. See the kustomize overlays under `deploy/` for
the canonical RBAC manifests.
