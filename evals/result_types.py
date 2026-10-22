"""Serializable Claude results and a shared evaluation interface; no SDK imports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class ToolCallResult(Protocol):
    name: str
    arguments: dict[str, Any]
    result: Any
    call_id: str


class AgentResult(Protocol):
    task: str
    final_output: str
    messages: list[dict[str, Any]]
    turns: int

    @property
    def tool_calls(self) -> Sequence[ToolCallResult]: ...

    @property
    def tool_names_used(self) -> list[str]: ...


@dataclass
class ClaudeToolCall:
    """One MCP call, paired with its result by SDK tool-use ID."""

    name: str
    arguments: dict[str, Any]
    result: Any
    call_id: str
    is_error: bool = False


@dataclass
class ClaudeCodeResult:
    """An observed exchange, or concatenated exchanges from one saved scenario."""

    task: str
    final_output: str = ""
    tool_calls: list[ClaudeToolCall] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    conversation_id: str = ""

    @property
    def tool_names_used(self) -> list[str]:
        return [call.name for call in self.tool_calls]
