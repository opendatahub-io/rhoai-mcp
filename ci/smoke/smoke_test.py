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

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000")
READER_TOKEN = os.environ["SMOKE_READER_TOKEN"]
READER_TOKEN_2 = os.environ["SMOKE_READER_TOKEN_2"]
DEPLOYER_TOKEN = os.environ["SMOKE_DEPLOYER_TOKEN"]

GOVERNED_READ_TOOL = "list_inference_services"
GOVERNED_DEPLOY_TOOL = "deploy_model"
UNGOVERNED_TOOL = "rhoai_suggest_tools"


def parse_body(text: str) -> dict:
    """Parse a JSON-RPC response body: plain JSON or SSE-framed."""
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    data_lines = [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]
    if not data_lines:
        raise AssertionError(f"no JSON or SSE data in response: {text[:200]!r}")
    return json.loads(data_lines[0])


def rpc(method: str, params: dict, token: str, session_id: str | None = None, msg_id: int = 1) -> tuple[dict, dict]:
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
        return parse_body(response.read().decode()), dict(response.headers.items())


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
    assert body.get("error") is None, f"initialize failed: {body}"
    session_id = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
    assert session_id, f"no session id in initialize response headers: {headers}"
    return session_id


def tool_names(token: str, session_id: str) -> set[str]:
    body, _ = rpc("tools/list", {}, token, session_id)
    assert body.get("error") is None, f"tools/list failed: {body}"
    return {t["name"] for t in body["result"]["tools"]}


def call_tool(token: str, session_id: str, name: str, arguments: dict) -> dict:
    body, _ = rpc("tools/call", {"name": name, "arguments": arguments}, token, session_id)
    assert body.get("error") is None, f"tools/call protocol error: {body}"
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
        failures.append(f"{name}: {exc!r}")
        print(f"FAIL {name}: {exc!r}")
    else:
        print(f"PASS {name}")


def check_health() -> None:
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as response:
        assert response.status == 200, f"health returned {response.status}"


def check_unauthenticated_rejected() -> None:
    request = urllib.request.Request(
        f"{BASE_URL}/mcp",
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
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
    assert result.get("isError") is True, f"denial not signaled: {result}"
    text = json.dumps(result)
    assert "not permitted" in text, f"denial text missing: {result}"


def check_ungoverned_call_succeeds() -> None:
    session = initialize(READER_TOKEN)
    result = call_tool(READER_TOKEN, session, UNGOVERNED_TOOL, {"intent": "list data science projects"})
    assert result.get("isError") is not True, f"ungoverned call failed: {result}"


def check_token_rotation_mid_session() -> None:
    """A freshly minted token on message N+1 is honored on N+1 (RHAI-14633's smoke case)."""
    assert READER_TOKEN != READER_TOKEN_2, "rotation check needs two distinct tokens"
    session = initialize(READER_TOKEN)
    before = tool_names(READER_TOKEN, session)
    # Same session, new bearer: identity must validate and the filtered
    # view must be identical (still the reader persona).
    after = tool_names(READER_TOKEN_2, session)
    assert after == before, "filtered tool list changed across token rotation"
    result = call_tool(READER_TOKEN_2, session, UNGOVERNED_TOOL, {"intent": "list data science projects"})
    assert result.get("isError") is not True, f"call with rotated token failed: {result}"


def main() -> int:
    check("T01 health endpoint answers", check_health)
    check("T02 unauthenticated initialize rejected with 401", check_unauthenticated_rejected)
    check("T03 deployer sees governed tools", check_deployer_sees_governed_tools)
    check("T04 reader tools/list is RBAC-filtered", check_reader_list_filtered)
    check("T05 reader tools/call is denied", check_reader_call_denied)
    check("T06 ungoverned tool call succeeds", check_ungoverned_call_succeeds)
    check("T07 token rotation honored mid-session", check_token_rotation_mid_session)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        return 1
    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
