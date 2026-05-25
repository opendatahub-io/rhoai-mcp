"""HMAC-signed workflow tokens for multi-step tool ordering enforcement.

Provides stateless, tamper-proof tokens that chain MCP tool calls into
ordered workflows. Each tool signs its output; the next tool verifies
the token before proceeding.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from rhoai_mcp.config import get_config


def _get_secret() -> bytes:
    """Get the HMAC secret from configuration."""
    return get_config().workflow_hmac_secret.get_secret_value().encode()


def _get_ttl() -> int:
    """Get the token TTL in seconds from configuration."""
    return get_config().workflow_token_ttl


def sign_step(step: str, data: dict[str, Any]) -> str:
    """Sign a workflow step with its output data.

    Args:
        step: Step identifier (e.g., "intent_extracted").
        data: The step's output data to embed in the token.

    Returns:
        An opaque, HMAC-signed token string.
    """
    payload = json.dumps(
        {"step": step, "data": data, "ts": int(time.time())},
        sort_keys=True,
    )
    sig = hmac.new(_get_secret(), payload.encode(), hashlib.sha256).hexdigest()
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{encoded}.{sig}"


def verify_step(token: str, expected_step: str) -> dict[str, Any]:
    """Verify a workflow token and extract the previous step's data.

    Args:
        token: The opaque token from the previous tool's output.
        expected_step: The step name that must have produced this token.

    Returns:
        The previous step's data dict on success, or
        ``{"error": "..."}`` on failure.
    """
    if not token:
        return {"error": "Workflow token is required — call the previous step first"}

    try:
        encoded, sig = token.rsplit(".", 1)
    except ValueError:
        return {"error": "Invalid workflow token format"}

    try:
        payload_bytes = base64.urlsafe_b64decode(encoded)
    except Exception:
        return {"error": "Invalid workflow token encoding"}

    expected_sig = hmac.new(_get_secret(), payload_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return {"error": "Invalid workflow token — call the previous step first"}

    try:
        parsed = json.loads(payload_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"error": "Invalid workflow token payload"}

    if parsed.get("step") != expected_step:
        actual = parsed.get("step", "unknown")
        return {
            "error": f"Wrong workflow order: expected step '{expected_step}' "
            f"but got '{actual}'. Call the required preceding tool first.",
        }

    ts = parsed.get("ts")
    data = parsed.get("data")
    if not isinstance(ts, int) or not isinstance(data, dict):
        return {"error": "Invalid workflow token payload"}

    age = int(time.time()) - ts
    if age > _get_ttl():
        return {"error": "Workflow token expired — restart from the first step"}

    return data
