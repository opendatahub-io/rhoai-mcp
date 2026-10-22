"""MCP schema discovery and conversion of observed agent results to DeepEval."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MCPServer,
    MCPToolCall,
    Turn,
)
from mcp.types import CallToolResult, TextContent, Tool

from evals.mcp_config import REPO_ROOT, build_mcp_servers
from evals.result_types import AgentResult, ToolCallResult

logger = logging.getLogger(__name__)


async def fetch_stdio_tool_schemas(
    repo_root: Path = REPO_ROOT, *, timeout: float = 60
) -> list[Tool]:
    """Read all tool definitions from the same local MCP configuration as Claude."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    config = build_mcp_servers()["rhoai-mcp"]
    parameters = StdioServerParameters(
        command=config["command"], args=config["args"], env=config["env"], cwd=repo_root
    )
    tools: list[Tool] = []
    async with (
        asyncio.timeout(timeout),
        stdio_client(parameters) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        cursor = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if cursor is None:
                break
    return tools


async def fetch_tool_schemas(rhoai_mcp_url: str, transport: str = "sse") -> list[Tool]:
    """Fetch tool schemas from a running rhoai-mcp server.

    Args:
        rhoai_mcp_url: Base URL of the rhoai-mcp server (e.g. http://localhost:8000).
        transport: MCP transport to use: 'streamable-http' or 'sse'.

    Returns:
        List of mcp.types.Tool objects.
    """
    from mcp import ClientSession

    base = rhoai_mcp_url.rstrip("/")
    if transport == "sse":
        from mcp.client.sse import sse_client

        client_context = sse_client(f"{base}/sse")
    elif transport == "streamable-http":
        from mcp.client.streamable_http import streamable_http_client

        client_context = streamable_http_client(f"{base}/mcp")
    else:
        raise ValueError(f"Unsupported transport {transport!r}: must be 'sse' or 'streamable-http'")

    async with client_context as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

    logger.info(f"Fetched {len(result.tools)} tool schemas from {rhoai_mcp_url}")
    return list(result.tools)


def build_mcp_server_from_schemas(tool_schemas: list[Tool]) -> MCPServer:
    """Build a DeepEval MCPServer from pre-fetched tool schemas.

    Args:
        tool_schemas: List of mcp.types.Tool objects.

    Returns:
        MCPServer instance for DeepEval metrics.
    """
    return MCPServer(
        server_name="rhoai-mcp",
        available_tools=tool_schemas,
    )


def tool_call_to_deepeval(tc: ToolCallResult) -> MCPToolCall:
    """Preserve MCP content/structured results instead of stringifying Python objects."""
    value = tc.result
    is_error = bool(getattr(tc, "is_error", False))
    if isinstance(value, CallToolResult):
        result = value.model_copy(update={"isError": value.isError or is_error})
    else:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                decoded = None
        else:
            decoded = value
        if isinstance(decoded, dict) and (
            isinstance(decoded.get("content"), list) or "structuredContent" in decoded
        ):
            result = CallToolResult.model_validate({"content": [], **decoded})
            result.isError |= is_error
        elif isinstance(decoded, list):
            result = CallToolResult(content=decoded, isError=is_error)
        else:
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            result = CallToolResult(
                content=[TextContent(type="text", text=text)],
                structuredContent=decoded if isinstance(decoded, dict) else None,
                isError=is_error,
            )
    payload: object = result.structuredContent
    if payload is None:
        if all(isinstance(block, TextContent) for block in result.content):
            text = "\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            )
            try:
                payload = json.loads(text)
            except ValueError:
                payload = text
        else:
            payload = [block.model_dump(mode="json") for block in result.content]
    # DeepEval's MCP renderer requires this envelope. Keep the original content
    # and structured payload intact; never turn dictionaries into Python strings.
    result.structuredContent = {"result": payload}
    # Strip server_label from args - it's injected by LCS infrastructure
    # and is not part of the tool's input schema
    args = {k: v for k, v in tc.arguments.items() if k != "server_label"}
    return MCPToolCall(
        name=tc.name,
        args=args,
        result=result,
    )


def result_to_conversational_test_case(
    result: AgentResult,
    mcp_server: MCPServer,
) -> ConversationalTestCase:
    """Convert an observed conversation, matching tool calls by ID rather than position.

    Maps the reconstructed message history into Turn objects, attaching
    MCPToolCall records to assistant turns that made tool calls.
    """
    turns: list[Turn] = []
    by_id = {call.call_id: call for call in result.tool_calls}
    if len(by_id) != len(result.tool_calls):
        raise ValueError("Duplicate tool call IDs in agent result")
    used: set[str] = set()

    for msg in result.messages:
        role = msg.get("role", "")

        if role == "user":
            turns.append(Turn(role="user", content=msg.get("content", "")))

        elif role == "assistant":
            content = msg.get("content") or ""
            tool_calls_in_msg = msg.get("tool_calls") or []

            mcp_tools_called = []
            for call in tool_calls_in_msg:
                call_id = call.get("id", "")
                if call_id not in by_id or call_id in used:
                    raise ValueError(f"Missing or repeated tool call ID in messages: {call_id}")
                mcp_tools_called.append(tool_call_to_deepeval(by_id[call_id]))
                used.add(call_id)

            if mcp_tools_called:
                turn = Turn(
                    role="assistant",
                    content=content,
                    mcp_tools_called=mcp_tools_called,
                )
                # Older DeepEval versions need this PrivateAttr workaround;
                # newer versions compute a read-only property from the tool calls.
                if not turn._mcp_interaction:
                    object.__setattr__(turn, "_mcp_interaction", True)
                turns.append(turn)
            else:
                turns.append(Turn(role="assistant", content=content))

        # Skip "tool" role messages - represented in MCPToolCall.result

    if used != set(by_id):
        raise ValueError("Agent tool calls are missing from the observed messages")
    return ConversationalTestCase(
        turns=turns,
        mcp_servers=[mcp_server],
    )


def result_to_single_turn_test_case(
    result: AgentResult,
    mcp_server: MCPServer,
) -> LLMTestCase:
    """Convert an agent result into a single-turn LLMTestCase.

    Used for simpler scenarios where multi-turn tracking isn't needed.
    """
    mcp_tools_called = [tool_call_to_deepeval(tc) for tc in result.tool_calls]

    return LLMTestCase(
        input=result.task,
        actual_output=result.final_output,
        mcp_servers=[mcp_server],
        mcp_tools_called=mcp_tools_called,
    )


# Existing LCS scenarios retain their public helper names.
lcs_result_to_conversational_test_case = result_to_conversational_test_case
lcs_result_to_single_turn_test_case = result_to_single_turn_test_case
