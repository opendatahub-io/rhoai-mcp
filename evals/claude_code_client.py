"""Persistent Claude Code sessions for workflow checks, independent of DeepEval.

Install the optional ``claude-check`` dependency group before importing this module.
Use one async context per scenario; query once, then follow up on that connection.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import McpServerConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_PREFIX = "mcp__rhoai-mcp__"
NAVIGATOR_TOOLS = (
    "recommend_model",
    "get_use_case_defaults",
    "get_expected_rps",
    "list_use_cases",
    "list_data_science_projects",
    "get_cluster_resources",
    "create_data_science_project",
    "list_serving_runtimes",
    "create_serving_runtime",
    "list_inference_services",
    "get_inference_service",
    "get_model_endpoint",
    "plan_deployment",
    "execute_deployment",
)


def build_mcp_servers() -> dict[str, McpServerConfig]:
    """Shared stdio configuration for workflow runs and later schema discovery."""
    return {
        "rhoai-mcp": {
            "type": "stdio",
            "command": sys.executable,
            "args": ["-m", "rhoai_mcp", "--transport", "stdio", "--mock-cluster"],
            "env": {
                "RHOAI_MCP_PLANNER_MODE": "local",
                "RHOAI_MCP_PLANNER_MODEL_CATALOG_URL": "",
                "PLANNER_DETECT_CLUSTER_GPUS": "false",
            },
        }
    }


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
    """Only the current user exchange, with messages in observed order."""

    task: str
    final_output: str = ""
    tool_calls: list[ClaudeToolCall] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    conversation_id: str = ""

    @property
    def tool_names_used(self) -> list[str]:
        return [call.name for call in self.tool_calls]


class ClaudeCodeError(RuntimeError):
    """Failed exchange; ``result`` retains the partial conversation for artifacts."""

    def __init__(self, message: str, result: ClaudeCodeResult) -> None:
        super().__init__(message)
        self.result = result


class _Exchange:
    """Normalize typed SDK messages without mixing built-in tools into MCP traces."""

    def __init__(self, result: ClaudeCodeResult, diagnostics: list[dict[str, Any]]) -> None:
        self.result = result
        self.diagnostics = diagnostics
        self.calls: dict[str, ClaudeToolCall] = {}
        self.completed: set[str] = set()
        self.final_parts: list[str] = []
        self.terminal: ResultMessage | None = None

    def accept(self, message: Any) -> None:
        if isinstance(message, AssistantMessage):
            self._assistant(message)
        elif isinstance(message, UserMessage):
            for block in message.content if isinstance(message.content, list) else []:
                if isinstance(block, ToolResultBlock):
                    self._tool_result(block)
            # Echoed prompts and SDK metadata are kept out of the user transcript.
            self.diagnostics.append({"type": "user", **asdict(message)})
        elif isinstance(message, ResultMessage):
            self.terminal = message
            self.result.turns = message.num_turns
            self.result.conversation_id = message.session_id
            self.result.final_output = "\n".join(self.final_parts) or message.result or ""
            self.diagnostics.append({"type": "result", **asdict(message)})
        else:
            self.diagnostics.append({"type": type(message).__name__, **asdict(message)})

    def _assistant(self, message: AssistantMessage) -> None:
        normalized: dict[str, Any] = {"role": "assistant", "content": ""}
        text: list[str] = []
        tool_calls = []
        for block in message.content:
            if isinstance(block, TextBlock):
                text.append(block.text)
                self.final_parts.append(block.text)
                self.result.final_output = "\n".join(self.final_parts)
            elif isinstance(block, ToolUseBlock):
                if block.id in self.calls:
                    raise ValueError(f"Duplicate tool-use ID: {block.id}")
                self.final_parts.clear()
                self.result.final_output = ""
                is_mcp = block.name.startswith(MCP_PREFIX)
                call = ClaudeToolCall(
                    name=block.name.removeprefix(MCP_PREFIX) if is_mcp else block.name,
                    arguments=block.input,
                    result=None,
                    call_id=block.id,
                )
                self.calls[block.id] = call
                if is_mcp:
                    self.result.tool_calls.append(call)
                    tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    )
                else:
                    self.diagnostics.append({"type": "tool_use", **asdict(block)})
        normalized["content"] = "\n".join(text)
        if tool_calls:
            normalized["tool_calls"] = tool_calls
        if text or tool_calls:
            self.result.messages.append(normalized)
        if message.error:
            raise ValueError(f"Assistant execution error: {message.error}")

    def _tool_result(self, block: ToolResultBlock) -> None:
        call = self.calls.get(block.tool_use_id)
        if call is None:
            raise ValueError(f"Result for unknown tool-use ID: {block.tool_use_id}")
        if block.tool_use_id in self.completed:
            raise ValueError(f"Duplicate tool result: {block.tool_use_id}")
        call.result = block.content
        call.is_error = bool(block.is_error)
        self.completed.add(block.tool_use_id)
        if call in self.result.tool_calls:
            self.result.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": call.result,
                    "is_error": call.is_error,
                }
            )
        else:
            self.diagnostics.append({"type": "tool_result", **asdict(block)})

    def validate(self) -> None:
        terminal = self.terminal
        if terminal is None:
            raise ValueError("Response ended without a terminal ResultMessage")
        if terminal.is_error or terminal.subtype != "success" or terminal.errors:
            raise ValueError(
                f"Claude execution failed ({terminal.subtype}): "
                f"{terminal.errors or terminal.result or terminal.stop_reason}"
            )
        if terminal.permission_denials:
            raise ValueError(f"Tool permission denied: {terminal.permission_denials}")
        if terminal.terminal_reason in {"aborted_streaming", "aborted_tools"}:
            raise ValueError(f"Claude exchange interrupted: {terminal.terminal_reason}")
        if not terminal.session_id:
            raise ValueError("Terminal result has no session ID")
        missing = self.calls.keys() - self.completed
        if missing:
            raise ValueError(f"Missing tool results: {sorted(missing)}")
        failed = [call.name for call in self.calls.values() if call.is_error]
        if failed:
            raise ValueError(f"Tool execution failed: {failed}")
        if not self.result.final_output.strip():
            raise ValueError("Claude returned no final assistant response")


class ClaudeCodeClient:
    """A single scenario's persistent SDK connection and mock MCP process.

    ``initialization`` retains discovery metadata. ``diagnostics`` and
    ``last_result`` describe the latest exchange, including failures. Failed
    exchanges cannot be retried on this client; create a fresh scenario instead.
    """

    def __init__(self, repo_root: Path = REPO_ROOT, timeout: float = 300) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.repo_root = Path(repo_root).resolve()
        self.timeout = timeout
        self.options = ClaudeAgentOptions(
            cwd=self.repo_root,
            cli_path=shutil.which("claude"),
            setting_sources=["project"],
            system_prompt={"type": "preset", "preset": "claude_code"},
            mcp_servers=build_mcp_servers(),
            strict_mcp_config=True,
            allowed_tools=["Skill", *(MCP_PREFIX + name for name in NAVIGATOR_TOOLS)],
            permission_mode="dontAsk",
        )
        self.initialization: dict[str, Any] = {}
        self.diagnostics: list[dict[str, Any]] = []
        self.last_result: ClaudeCodeResult | None = None
        self._sdk: ClaudeSDKClient | None = None
        self._conversation_id = ""
        self._started = False
        self._failed = False
        self._busy = False

    @property
    def available_commands(self) -> list[str]:
        entries = [
            *self.initialization.get("commands", []),
            *self.initialization.get("slash_commands", []),
        ]
        return sorted({entry if isinstance(entry, str) else entry["name"] for entry in entries})

    async def __aenter__(self) -> ClaudeCodeClient:
        if self._sdk is not None or self._started:
            raise RuntimeError("Use a fresh ClaudeCodeClient for each scenario")
        sdk = ClaudeSDKClient(options=self.options)
        await sdk.__aenter__()
        try:
            self.initialization = await sdk.get_server_info() or {}
        except BaseException:
            await sdk.__aexit__(*sys.exc_info())
            raise
        self._sdk = sdk
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        sdk, self._sdk = self._sdk, None
        if sdk is not None:
            await sdk.__aexit__(exc_type, exc, traceback)

    async def query(self, task: str, *, skill: str) -> ClaudeCodeResult:
        """Start with an explicit /command (identical dispatch before and after migration)."""
        if self._started:
            raise RuntimeError("Scenario already started; use follow_up()")
        if skill not in self.available_commands:
            result = ClaudeCodeResult(task=task)
            self.last_result = result
            raise ClaudeCodeError(f"Command /{skill} was not discovered", result)
        return await self._exchange(f"/{skill} {task}")

    async def follow_up(self, reply: str, conversation_id: str) -> ClaudeCodeResult:
        """Continue on the same connection, rejecting IDs from other scenarios."""
        if not self._conversation_id or conversation_id != self._conversation_id:
            raise ValueError("conversation_id does not match this client's active scenario")
        return await self._exchange(reply)

    async def _exchange(self, prompt: str) -> ClaudeCodeResult:
        if self._sdk is None:
            raise RuntimeError("Use ClaudeCodeClient as an async context manager")
        if self._failed:
            raise RuntimeError("Scenario failed; start a fresh client")
        if self._busy:
            raise RuntimeError("Concurrent exchanges on one client are not supported")
        self._busy = True
        self._started = True
        result = ClaudeCodeResult(
            task=prompt,
            messages=[{"role": "user", "content": prompt}],
            conversation_id=self._conversation_id,
        )
        self.last_result = result
        self.diagnostics = []
        exchange = _Exchange(result, self.diagnostics)
        try:
            async with asyncio.timeout(self.timeout):
                await self._sdk.query(prompt)
                async for message in self._sdk.receive_response():
                    if isinstance(message, SystemMessage) and message.subtype == "init":
                        self.initialization.update(message.data)
                    exchange.accept(message)
            exchange.validate()
            if self._conversation_id and result.conversation_id != self._conversation_id:
                raise ValueError("Claude session ID changed during the scenario")
            self._conversation_id = result.conversation_id
        except Exception as exc:
            self._failed = True
            detail = (
                f"Exchange timed out after {self.timeout:g}s"
                if isinstance(exc, TimeoutError)
                else str(exc)
            )
            raise ClaudeCodeError(detail, result) from exc
        except BaseException:
            self._failed = True
            raise
        finally:
            self._busy = False
        return result
