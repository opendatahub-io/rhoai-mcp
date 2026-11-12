#!/usr/bin/env python3
"""In-repo KinD smoke E2E for rhoai-mcp (RHAI-14598).

SDK-agnostic by design: raw JSON-RPC over streamable-http using only
the Python standard library — no MCP SDK client imports, so the harness
is inert to the v1->v2 SDK migration. It asserts on the JSON wire
surface only (tool names, "isError", error text), which is identical
across SDK versions; camelCase survives on the wire in both.

Environment:
  SMOKE_BASE_URL     server base URL (default http://127.0.0.1:8000)
  SMOKE_READER_TOKEN   bearer for the smoke-reader persona
  SMOKE_READER_TOKEN_2 a second, freshly minted token for smoke-reader
                       (the token-rotation case)
  SMOKE_DEPLOYER_TOKEN bearer for the smoke-deployer persona

Exit code 0 iff every check passes.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000")
READER_TOKEN = os.environ["SMOKE_READER_TOKEN"]
READER_TOKEN_2 = os.environ["SMOKE_READER_TOKEN_2"]
DEPLOYER_TOKEN = os.environ["SMOKE_DEPLOYER_TOKEN"]

GOVERNED_READ_TOOL = "list_inference_services"
GOVERNED_DEPLOY_TOOL = "deploy_model"
# Ungoverned meta-composite tool (no K8s permission mapping) — note the
# gateway imports these with a "rhoai_" prefix; the server's own name is
# unprefixed.
UNGOVERNED_TOOL = "suggest_tools"


def parse_body(text: str) -> dict:
    """Parse a JSON-RPC response body: plain JSON or SSE-framed."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    data_lines = [
        line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")
    ]
    if not data_lines:
        raise AssertionError("no JSON or SSE data in response body")
    return json.loads(data_lines[0])


def rpc(
    method: str, params: dict, token: str, session_id: str | None = None, msg_id: int = 1
) -> tuple[dict, Any]:
    """POST one JSON-RPC message; return (parsed body, response headers)."""
    payload = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    request = urllib.request.Request(
        f"{BASE_URL}/mcp", data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return parse_body(response.read().decode()), response.headers


def initialize(token: str) -> str:
    body, headers = rpc(
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "kind-smoke", "version": "0"},
        },
        token,
    )
    assert body.get("error") is None, "initialize returned an error response"
    # http.client headers are case-insensitive
    session_id = headers.get("Mcp-Session-Id")
    assert session_id, "initialize response missing the session id"
    return session_id


def tool_names(token: str, session_id: str) -> set[str]:
    body, _ = rpc("tools/list", {}, token, session_id)
    assert body.get("error") is None, "tools/list returned an error response"
    return {t["name"] for t in body["result"]["tools"]}


def call_tool(token: str, session_id: str, name: str, arguments: dict) -> dict:
    body, _ = rpc("tools/call", {"name": name, "arguments": arguments}, token, session_id)
    assert body.get("error") is None, "tools/call returned a protocol error"
    return body["result"]


# --- checks -----------------------------------------------------------------

failures: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
    except AssertionError as exc:
        failures.append(f"{name}: {exc}")
        print(f"FAIL {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        # Safe details only — exception reprs can embed response bodies
        detail = f"{type(exc).__name__}"
        failures.append(f"{name}: {detail}")
        print(f"FAIL {name}: {detail}")
    else:
        print(f"PASS {name}")


def check_health() -> None:
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as response:
        assert response.status == 200, f"health returned {response.status}"


def check_unauthenticated_rejected() -> None:
    request = urllib.request.Request(
        f"{BASE_URL}/mcp",
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=10)
        raise AssertionError("unauthenticated initialize was accepted")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, f"expected 401, got {exc.code}"
        assert "Bearer" in (exc.headers.get("WWW-Authenticate") or ""), "no WWW-Authenticate on 401"


def check_deployer_sees_governed_tools() -> None:
    session = initialize(DEPLOYER_TOKEN)
    names = tool_names(DEPLOYER_TOKEN, session)
    assert GOVERNED_READ_TOOL in names, f"{GOVERNED_READ_TOOL} missing for deployer"
    assert GOVERNED_DEPLOY_TOOL in names, f"{GOVERNED_DEPLOY_TOOL} missing for deployer"


def check_reader_list_filtered() -> None:
    session = initialize(READER_TOKEN)
    names = tool_names(READER_TOKEN, session)
    assert GOVERNED_READ_TOOL in names, f"{GOVERNED_READ_TOOL} missing for reader"
    assert GOVERNED_DEPLOY_TOOL not in names, (
        f"{GOVERNED_DEPLOY_TOOL} visible to reader — RBAC list filtering not enforced"
    )


def check_reader_call_denied() -> None:
    session = initialize(READER_TOKEN)
    result = call_tool(READER_TOKEN, session, GOVERNED_DEPLOY_TOOL, {})
    assert result.get("isError") is True, "denial not signaled as isError"
    text = json.dumps(result)
    assert "not permitted" in text, "denial text missing from the error response"


def check_ungoverned_call_succeeds() -> None:
    session = initialize(READER_TOKEN)
    result = call_tool(
        READER_TOKEN, session, UNGOVERNED_TOOL, {"intent": "list data science projects"}
    )
    assert result.get("isError") is not True, "ungoverned tool call failed"


def _kubectl(*args: str, stdin: str | None = None) -> str:
    """Run kubectl (the smoke script runs in the workflow's cluster context).

    On failure, raises RuntimeError carrying the command, exit code, and
    stderr — kubectl writes diagnostics (never tokens) to stderr, and the
    command identity is what the check runner needs to localize a failure.
    """
    import subprocess

    proc = subprocess.run(
        ["kubectl", *args], capture_output=True, text=True, check=False, input=stdin
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"kubectl {' '.join(args)} exited {proc.returncode}: {proc.stderr.strip()[:300]}"
        )
    return proc.stdout.strip()


def check_token_rotation_mid_session() -> None:
    """A rotated token on message N+1 is honored on N+1, with no cached auth (RHAI-14633).

    Proves per-message authentication on the session-based transport: a
    token bound to a throwaway pod is used to open a session, then the pod
    is deleted — invalidating the bound token immediately (TokenRequest
    durations have a 600s floor, so waiting for expiry is not viable) —
    and the invalidated token is rejected (the server is not caching
    authentication at the session level) while a freshly minted token
    continues the same session, including a governed, Kubernetes-backed
    tool call whose result data is asserted against a seeded resource,
    so an empty result cannot pass.
    """
    # A dummy pod to bind the throwaway token to: deleting the pod
    # invalidates the token deterministically.
    _kubectl(
        "run",
        "smoke-token-rotator",
        "-n",
        "rhoai-mcp",
        "--image=rhoai-mcp:smoke",
        "--restart=Never",
        "--command",
        "--",
        "python",
        "-c",
        "import time; time.sleep(600)",
    )
    try:
        # Bound-token validation checks the pod's existence (name+uid), not
        # its readiness — no need to wait for the pod to run.
        pod_uid = _kubectl(
            "get", "pod", "smoke-token-rotator", "-n", "rhoai-mcp", "-o", "jsonpath={.metadata.uid}"
        )
        bound_token = _kubectl(
            "create",
            "token",
            "smoke-reader",
            "-n",
            "rhoai-mcp",
            "--bound-object-kind=Pod",
            "--bound-object-name=smoke-token-rotator",
            f"--bound-object-uid={pod_uid}",
        )
        assert bound_token and not hmac.compare_digest(bound_token, READER_TOKEN), (
            "rotation check needs a distinct bound token"
        )

        # Seed a known InferenceService so the governed listing must return
        # real data — an exception-swallowed empty list cannot pass.
        seeded = {
            "apiVersion": "serving.kserve.io/v1beta1",
            "kind": "InferenceService",
            "metadata": {"name": "smoke-seeded-isvc", "namespace": "rhoai-mcp"},
        }
        _kubectl("apply", "-f", "-", stdin=json.dumps(seeded))

        session = initialize(bound_token)
        before = tool_names(bound_token, session)
        assert GOVERNED_READ_TOOL in before, "bound token lacked the reader persona"

        # Invalidate the bound token, then prove the server re-authenticates
        # every message rather than caching at session level.
        _kubectl("delete", "pod", "smoke-token-rotator", "-n", "rhoai-mcp", "--wait=false")
        _kubectl(
            "wait", "--for=delete", "pod/smoke-token-rotator", "-n", "rhoai-mcp", "--timeout=60s"
        )
        try:
            tool_names(bound_token, session)
            raise AssertionError(
                "invalidated token still accepted — authentication is cached at session level"
            )
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, f"invalidated token got HTTP {exc.code}, expected 401"

        # The fresh token continues the SAME session: filtered view unchanged
        after = tool_names(READER_TOKEN, session)
        assert after == before, "filtered tool list changed across token rotation"

        # A governed, Kubernetes-backed call with the rotated token returns
        # the seeded resource — proving the listing reached the API.
        result = call_tool(READER_TOKEN, session, GOVERNED_READ_TOOL, {"namespace": "rhoai-mcp"})
        assert result.get("isError") is not True, "governed call with rotated token failed"
        structured = result.get("structuredContent")
        assert isinstance(structured, dict), "governed call returned no structured result data"
        assert any(
            item.get("name") == "smoke-seeded-isvc"
            for item in structured.get("items", [])
            if isinstance(item, dict)
        ), "seeded InferenceService missing from the governed listing"
    finally:
        _kubectl("delete", "pod", "smoke-token-rotator", "-n", "rhoai-mcp", "--ignore-not-found")
        _kubectl(
            "delete",
            "inferenceservice",
            "smoke-seeded-isvc",
            "-n",
            "rhoai-mcp",
            "--ignore-not-found",
        )


def check_rolebinding_freshness() -> None:
    """A live RoleBinding change is picked up on the next call (RHAI-14633).

    Opens a session as the reader, grants the deployer's ClusterRole
    mid-session, and asserts the NEXT tools/list reflects the grant —
    proving RBAC is re-evaluated per call, never a closure-captured set.
    Then revokes the grant and asserts the view reverts. RBAC propagation
    is awaited via kubectl auth can-i (Kubernetes-side truth) before each
    MCP assertion, so the check tests the server's freshness rather than
    racing the authorizer's informer.
    """
    import time

    binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "ClusterRoleBinding",
        "metadata": {"name": "rhoai-mcp-smoke-reader-deployer-grant"},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "ClusterRole",
            "name": "rhoai-mcp-smoke-deployer",
        },
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": "smoke-reader",
                "namespace": "rhoai-mcp",
            }
        ],
    }

    def can_i_create() -> bool:
        import subprocess

        proc = subprocess.run(
            [
                "kubectl",
                "auth",
                "can-i",
                "create",
                "inferenceservices",
                "--as",
                "system:serviceaccount:rhoai-mcp:smoke-reader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip() == "yes"

    def await_rbac(expected: bool, timeout_s: int = 30) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if can_i_create() == expected:
                return
            time.sleep(1)
        raise AssertionError(f"RBAC propagation did not reach {expected} within {timeout_s}s")

    session = initialize(READER_TOKEN)
    names = tool_names(READER_TOKEN, session)
    assert GOVERNED_DEPLOY_TOOL not in names, "precondition: deploy tool visible before grant"

    try:
        _kubectl("apply", "-f", "-", stdin=json.dumps(binding))
        await_rbac(True)
        names = tool_names(READER_TOKEN, session)
        assert GOVERNED_DEPLOY_TOOL in names, (
            "grant not visible on the next call — RBAC is cached, not re-evaluated per call"
        )
    finally:
        _kubectl(
            "delete",
            "clusterrolebinding",
            "rhoai-mcp-smoke-reader-deployer-grant",
            "--ignore-not-found",
        )

    await_rbac(False)
    names = tool_names(READER_TOKEN, session)
    assert GOVERNED_DEPLOY_TOOL not in names, "revocation not visible on the next call"


def main() -> int:
    check("T01 health endpoint answers", check_health)
    check("T02 unauthenticated initialize rejected with 401", check_unauthenticated_rejected)
    check("T03 deployer sees governed tools", check_deployer_sees_governed_tools)
    check("T04 reader tools/list is RBAC-filtered", check_reader_list_filtered)
    check("T05 reader tools/call is denied", check_reader_call_denied)
    check("T06 ungoverned tool call succeeds", check_ungoverned_call_succeeds)
    check("T07 token rotation honored mid-session", check_token_rotation_mid_session)
    check("T08 RoleBinding freshness on next call", check_rolebinding_freshness)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        return 1
    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
