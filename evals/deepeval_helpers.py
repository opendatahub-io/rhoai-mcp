"""Helpers to convert LCS results into DeepEval test cases."""

from __future__ import annotations

import logging
from typing import Any

from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MCPServer,
    MCPToolCall,
    Turn,
)

from evals.lcs_client import LCSResult

logger = logging.getLogger(__name__)


async def fetch_tool_schemas(rhoai_mcp_url: str) -> list[dict[str, Any]]:
    """Fetch tool schemas from a running rhoai-mcp server via MCP SSE client.

    Args:
        rhoai_mcp_url: Base URL of the rhoai-mcp server (e.g. http://localhost:8000).

    Returns:
        List of tool dicts with 'name', 'description', 'inputSchema'.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    sse_url = f"{rhoai_mcp_url.rstrip('/')}/sse"
    tools: list[dict[str, Any]] = []

    async with sse_client(sse_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            for tool in result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema,
                })

    logger.info(f"Fetched {len(tools)} tool schemas from {rhoai_mcp_url}")
    return tools


def build_mcp_server_from_schemas(tool_schemas: list[dict[str, Any]]) -> MCPServer:
    """Build a DeepEval MCPServer from pre-fetched tool schemas.

    Args:
        tool_schemas: List of dicts with 'name', 'description', 'inputSchema'.

    Returns:
        MCPServer instance for DeepEval metrics.
    """
    return MCPServer(
        server_name="rhoai-mcp",
        available_tools=tool_schemas,
    )


def _lcs_tool_call_to_deepeval(tc: Any) -> MCPToolCall:
    """Convert an LCSToolCall to a DeepEval MCPToolCall."""
    return MCPToolCall(
        name=tc.name,
        args=tc.arguments,
        result=tc.result,
    )


def lcs_result_to_conversational_test_case(
    result: LCSResult,
    mcp_server: MCPServer,
) -> ConversationalTestCase:
    """Convert an LCSResult into a DeepEval ConversationalTestCase.

    Maps the reconstructed message history into Turn objects, attaching
    MCPToolCall records to assistant turns that made tool calls.
    """
    turns: list[Turn] = []
    tc_index = 0

    for msg in result.messages:
        role = msg.get("role", "")

        if role == "user":
            turns.append(Turn(role="user", content=msg.get("content", "")))

        elif role == "assistant":
            content = msg.get("content") or ""
            tool_calls_in_msg = msg.get("tool_calls") or []

            mcp_tools_called = []
            for _ in tool_calls_in_msg:
                if tc_index < len(result.tool_calls):
                    mcp_tools_called.append(
                        _lcs_tool_call_to_deepeval(result.tool_calls[tc_index])
                    )
                    tc_index += 1

            if mcp_tools_called:
                turns.append(
                    Turn(
                        role="assistant",
                        content=content,
                        mcp_tools_called=mcp_tools_called,
                    )
                )
            else:
                turns.append(Turn(role="assistant", content=content))

        # Skip "tool" role messages - represented in MCPToolCall.result

    return ConversationalTestCase(
        turns=turns,
        mcp_servers=[mcp_server],
    )


def lcs_result_to_single_turn_test_case(
    result: LCSResult,
    mcp_server: MCPServer,
) -> LLMTestCase:
    """Convert an LCSResult into a single-turn LLMTestCase.

    Used for simpler scenarios where multi-turn tracking isn't needed.
    """
    mcp_tools_called = [
        _lcs_tool_call_to_deepeval(tc) for tc in result.tool_calls
    ]

    return LLMTestCase(
        input=result.task,
        actual_output=result.final_output,
        mcp_servers=[mcp_server],
        mcp_tools_called=mcp_tools_called,
    )
