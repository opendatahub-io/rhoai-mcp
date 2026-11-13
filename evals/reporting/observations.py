"""Read agent usage and tool counts from saved artifacts, without model libraries."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.reporting.models import EvalRecord


@dataclass
class AgentObservations:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    model_responses: int | None = None
    mcp_calls: int | None = None
    mcp_errors: int | None = None
    unreturned_mcp_calls: int | None = None
    capability_loads: int | None = None
    artifact_run_id: str = ""
    artifact_sha256: str = ""
    messages_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _name(name: str) -> str:
    return name.split("__", 2)[-1] if name.startswith("mcp__") else name


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _tool_error(content: Any) -> bool:
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (ValueError, TypeError):
            return False
    if not isinstance(content, dict):
        return False
    if content.get("error") or content.get("isError") or content.get("success") is False:
        return True
    if _tool_error(content.get("structuredContent")):
        return True
    return any(
        _tool_error(part.get("text"))
        for part in content.get("content") or []
        if isinstance(part, dict) and part.get("type") == "text"
    )


def read_agent_observations(
    record: EvalRecord, artifacts_root: Path | None = None
) -> AgentObservations:
    """Missing artifacts or incomplete usage stay unavailable, never zero-filled.

    With ``artifacts_root``, resolve portable artifacts as root/run/attempt even
    when a historical record contains an absolute path from another checkout.
    """
    result = AgentObservations()
    recorded = Path(record.artifact_path) if record.artifact_path else None
    original_run = (
        recorded.parent.name if recorded and recorded.name == record.attempt_id else record.run_id
    )
    path = (
        artifacts_root / original_run / record.attempt_id
        if artifacts_root is not None
        else Path(record.artifact_path)
        if record.artifact_path
        else None
    )
    if path is None:
        result.warnings.append("No artifact path; agent usage and tool counts are unavailable.")
        return result
    try:
        raw = (path / "artifact.json").read_bytes()
        artifact = json.loads(raw)
        result.artifact_sha256 = hashlib.sha256(raw).hexdigest()
        if artifact["attempt_id"] != record.attempt_id:
            raise ValueError(f"Artifact attempt identity differs for {record.attempt_id}")
        result.artifact_run_id = artifact["run_id"]
        result.metadata = artifact.get("metadata", {})
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        result.warnings.append(f"Artifact unavailable ({type(exc).__name__}); usage is n/a.")
        return result
    try:
        raw = (path / "messages.json").read_bytes()
        messages = json.loads(raw)
        result.messages_sha256 = hashlib.sha256(raw).hexdigest()
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError("Native messages must be a list")
    except (OSError, ValueError) as exc:
        result.warnings.append(f"Native messages unavailable ({type(exc).__name__}); usage is n/a.")
        return result

    # The runner saves one cumulative history. Never add each turn's cumulative
    # history or a nested RunUsage to these per-response RequestUsage values.
    unique = {json.dumps(m, sort_keys=True): m for m in messages}
    if len(unique) != len(messages):
        result.warnings.append("Identical native messages deduplicated before counting usage.")
    messages = list(unique.values())
    responses = [m for m in messages if m.get("kind") == "response"]
    result.model_responses = len(responses)
    usage_fields = {
        "input_tokens": ("input_tokens", "request_tokens"),
        "output_tokens": ("output_tokens", "response_tokens"),
        "reasoning_tokens": ("output_reasoning_tokens",),
        "cache_read_tokens": ("cache_read_tokens",),
    }
    for target, aliases in usage_fields.items():
        counts: list[int] = []
        for response in responses:
            usage = response.get("usage") or {}
            value = next((usage[key] for key in aliases if key in usage), None)
            count = _count(value)
            if count is None or _count(usage.get("unmeasured_requests", 0)) != 0:
                break
            counts.append(count)
        if responses and len(counts) == len(responses):
            setattr(result, target, sum(counts))
    if result.output_tokens is None:
        result.warnings.append("Output usage missing from at least one response; total is n/a.")

    names = {_name(tool["name"]) for tool in artifact.get("tool_schemas", [])}
    calls: list[dict[str, Any]] = []
    pending: dict[str, deque[tuple[int, int | None]]] = {}
    loads = 0
    for index, message in enumerate(messages):
        for part in message.get("parts", []):
            call_id = part.get("tool_call_id") or ""
            if part.get("part_kind") == "tool-call":
                call_index = None
                if part.get("tool_kind") == "capability-load":
                    loads += 1
                elif _name(part.get("tool_name", "")) in names:
                    call_index = len(calls)
                    calls.append({"returned": False, "error": False})
                if call_id:
                    queue = pending.setdefault(call_id, deque())
                    if queue and queue[-1][0] != index:
                        queue.clear()
                    queue.append((index, call_index))
            elif part.get("part_kind") in {"tool-return", "retry-prompt"}:
                queue = pending.get(call_id)
                if queue:
                    _, call_index = queue.popleft()
                    if call_index is not None:
                        calls[call_index]["returned"] = True
                        calls[call_index]["error"] = (
                            part.get("part_kind") == "retry-prompt"
                            or part.get("outcome") not in (None, "success")
                            or _tool_error(part.get("content"))
                        )
    if names:
        result.mcp_calls = len(calls)
        result.mcp_errors = sum(call["error"] for call in calls)
        result.unreturned_mcp_calls = sum(not call["returned"] for call in calls)
    else:
        result.warnings.append("MCP schemas unavailable; MCP call counts are n/a.")
    result.capability_loads = loads
    return result
