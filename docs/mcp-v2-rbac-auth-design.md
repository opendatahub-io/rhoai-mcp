# MCP v2 RBAC & Auth Redesign — Design Doc

**Story:** 1.6 / [RHAI-14612](https://redhat.atlassian.net/browse/RHAI-14612) · **Status:** DRAFT — everything final except the two 1.3 ratification slots (owner meeting 2026-11-11)
**Inputs:** 1.1 (RHAI-14607), 1.2 (RHAI-14608), 1.4 (RHAI-14610), 1.5 (RHAI-14611); consolidated evidence in the fpj-dev-notes `mcp-v2-branch-findings` record. Verified against installed mcp 2.3.0.
**Implements:** 3.1 (RHAI-14627), 3.2 (RHAI-14629), 3.3 (RHAI-14630), 3.4 (RHAI-14631), 3.5 (RHAI-14632), 3.6 (RHAI-14633)

---

## 1. Interception approach (implements 3.1)

**RBAC enforcement moves from the `request_handlers` monkey-patch to a `ServerMiddleware` passed via the public constructor parameter.**

```python
# server.py — create_mcp()
mcp = MCPServer(
    name="rhoai-mcp",
    instructions=...,
    lifespan=self._create_lifespan(),
    middleware=[RBACMiddleware(self)],   # public API, zero private access
)
```

```python
class RBACMiddleware:
    """Dual-point RBAC enforcement: tools/list filtering + tools/call denial."""

    def __init__(self, server: RHOAIServer) -> None:
        self._server = server

    async def __call__(self, ctx: ServerRequestContext, call_next: CallNext) -> HandlerResult:
        if ctx.method == "tools/call":
            return await self._enforce_call(ctx, call_next)
        if ctx.method == "tools/list":
            return await self._filter_list(ctx, call_next)
        return await call_next(ctx)
```

Semantics (carried over from v1 exactly — same `get_allowed_tools()` contract, same allowed/governed set logic, unmapped tools allowed):

- **`tools/call`** — pre-check before `call_next`: if the tool is governed and not allowed, return `CallToolResult(is_error=True, content=[TextContent("Tool '<name>' is not permitted for the current user")])` **without dispatching the handler**. No K8s API call is made with the caller's credentials for a denied tool.
- **`tools/list`** — call through, then filter: build a new `ListToolsResult` from `result.tools` minus governed-and-denied entries (construct a fresh result rather than mutating — pydantic models may be frozen; `ListToolsResult` in v2 carries `ttl_ms`/`cache_scope` fields which must be left unset, see §5).
- **RBAC re-evaluation per call** — `check = await asyncio.to_thread(self._server.get_allowed_tools)` inside the middleware body on every request. Never a closure-captured permitted set.
- **Fail-closed** — if the RBAC check raises: `tools/list` returns an empty tool list; `tools/call` returns the denied result. If `UserContext.current()` is `None` under OIDC: deny all (as today).
- **Denial path (1.5, decided 2026-11-10):** in-band `is_error` result, message text unchanged. No HTTP 403 for per-tool denial — `403 + WWW-Authenticate: Bearer error="insufficient_scope"` is reserved for request-level scope denial (SDK `RequireAuthMiddleware` convention), applicable only if `oidc_required_scopes` ever moves from metadata-only to enforcement. The in-band message is `ToolError`-backed since 2.10 (commit `3fd190f`) and survives v2 error scrubbing verbatim.

**UserContext ordering (1.2, verified by test):** under Streamable HTTP — stateless and stateful — `OIDCAuthMiddleware` populates `UserContext` before dispatch reaches the middleware, so `RBACMiddleware` reads `UserContext.current()` directly. Under SSE (legacy), dispatch runs in the GET /sse task: `UserContext` reflects connection time, per-message identity comes from `ctx.meta` (see §3), and the SDK pins each session to the creating credential (mismatched POST → 404).

**Dual enforcement (list + call):** ⏳ *PENDING 1.3 RATIFICATION — owner meeting 2026-11-11.* Recommendation on the table: keep both (call enforcement is the security boundary; list filtering is UX and the gateway broker's catalog import depends on it). If call-time-only is chosen instead, `_filter_list` drops out and the broker-team impact needs a 4.3 follow-up.

## 2. Install verification (implements 3.2)

0.1's tripwire promotes to the v2 world: installation is now a constructor argument, so the install assertion becomes (a) the `RBACMiddleware` instance is present in the server's middleware list (`mcp._lowlevel_server.middleware` — the list itself is public per 1.1; accessor private, so prefer asserting through behavior), and (b) behavioral regression: a governed-denied `tools/call` is blocked and a filtered `tools/list` omits governed-denied tools, driven through `MCPServer.call_tool()` / the JSON-RPC harness. The existing `tests/test_tool_error_surface.py` already pins the error-text contract; `tests/auth/test_user_context_ordering.py` pins the ordering precondition.

## 3. Token sourcing — the `get_k8s(ctx)` refactor (implements 3.4, D1)

The `k8s` property becomes a context-taking accessor:

```python
def get_k8s(self, ctx: Context) -> K8sClient:
    """Per-request K8s client for the calling user."""
```

Token-source precedence, per 1.2's per-transport findings:

1. **`ctx.meta.request_context.headers["authorization"]`** when `ctx.meta` is present — the per-message Request (SSE: fresh per POST; the SDK carries it on `SessionMessage.metadata`)
2. **`UserContext.current().token`** otherwise — per-POST fresh under Streamable HTTP (contextvar set by `OIDCAuthMiddleware`); connection-time under SSE, bounded by the SDK's session-owner pinning

`_get_per_message_token()` (private `request_ctx` ContextVar — gone in v2) is deleted; `tests/auth/test_server_oidc.py:161-229` (its request_ctx tests) is deleted or rewritten against the new accessor. K8s client construction per `oidc_kube_auth_strategy` (`user-token` / `impersonation`) is unchanged in semantics — only the token source moves.

*Verification item inside 3.4:* confirm the exact accessor path from the high-level tool `Context` to the per-message headers (`Context.request_context.meta.request_context.headers`) against mcp 2.3.0 — 1.2 verified the middleware-side `ctx.meta`; the tool-side path needs the same one-time source check.

**3.5 (ctx threading):** add the `ctx: Context` parameter to every domain/composite tool that transitively touches `server.k8s`, update call sites, grep-audit for bare `server.k8s` property access. Wide but mechanical; splittable per domain. Sequencing note: #30 (async K8s client) touches the same call sites — not concurrent landing (recorded on the issue).

**3.6 (rotation/freshness tests):** run in the 0.4 KinD harness on both channels — stateful Streamable HTTP (contextvar path) and SSE (`ctx.meta` path) — since rotation semantics differ by transport per 1.2: a rotated token on message N+1 must be honored on N+1; a live RoleBinding change picked up on the next call.

## 4. Auth configuration (from 1.4, decided: accept)

- **Audience guidance** (folds into 0.2's auth doc): JWT-mode deployments SHOULD set `oidc_audience` to the deployment's own resource URL — the same value as `oidc_resource_url` in the RFC 9728 metadata — making `aud` per-resource, which is the server-side entirety of RFC 8707. Optional hardening for 3.1 to carry: a startup warning when JWT mode runs with the generic default audience (`"rhoai-mcp"`).
- **TokenReview cluster-scoping mitigation** documented in 0.2: cluster scoping + per-caller RBAC + TTL/audit bounds; per-caller RBAC is a damage bound (no escalation), not a confused-deputy defense.
- Revisit triggers (not in scope): federated multi-instance topologies → enforce `aud == own resource URL`; sender-constrained tokens (DPoP / RFC 8705-class) when the ecosystem supports them.

## 5. Cache-hints policy

**Never on `tools/list` or `tools/call`.** Both are per-caller filtered; any caching would leak one user's view to another. Concretely: never set `ttl_ms`/`cache_scope` on the `ListToolsResult`/`CallToolResult` we construct in `RBACMiddleware`, and never pass `cache_hints` entries for `tools/list`/`tools/call` to `MCPServer(...)`/`Server(...)` (v2's `cache_hints` constructor parameter exists on both). 4.3 re-verifies the policy holds end-to-end through the gateway.

## 6. Verified non-issues and watch items

Non-issues (verified — nothing to migrate):
- No `ctx.meta` use in the v1 code (the migration was inert; §3 now introduces *deliberate* use for SSE per 1.2)
- No structured-output declarations
- No client-side MCP SDK in `src/`
- Our own `RHOAI_MCP_*` env vars — v2's dropped SDK env config is inert for us

E5 watch items (for 5.2's eval runs, not design constraints): 4 MiB request-body limit (large tool payloads), session idle expiry / 10k-session cap if sessions are retained, same-origin-only redirect following behind the gateway.

## 7. Story-impact summary

| Story | Effect of this design |
|---|---|
| 3.1 (RHAI-14627) | Implement `RBACMiddleware` + constructor wiring. Complexity **down** from the 5-pt estimate: public API, no handler-dict surgery; the 1.2 ordering test already proves the precondition |
| 3.2 (RHAI-14629) | Tripwire promotion: middleware-presence + behavioral enforcement (§2) |
| 3.3 (RHAI-14630) | **Scope reduced** (1.5 decision): in-band denial is ToolError-backed and pinned by `test_tool_error_surface.py`; remaining work is the 5.3 live smoke verification |
| 3.4 (RHAI-14631) | `get_k8s(ctx)` with the §3 token precedence; `_get_per_message_token` deleted |
| 3.5 (RHAI-14632) | ctx threading, unchanged from outline; grep audit |
| 3.6 (RHAI-14633) | Rotation tests on both transport channels (§3) |

## 8. Pending ratification & sign-off

- ⏳ **1.3 dual enforcement** — owner meeting 2026-11-11; recommendation: keep both
- ⏳ **1.3 interception ratification** — same meeting; recommendation: the `ServerMiddleware` approach above (evolved Option A — cleaner than the v1-era wrap-after-registration, zero private API)
- Sign-off: RBAC/OIDC owner + Epic 3 executor (to be recorded on RHAI-14612 after the meeting)
