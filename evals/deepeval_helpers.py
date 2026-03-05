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
from mcp.types import CallToolResult, TextContent, Tool

from evals.lcs_client import LCSResult

logger = logging.getLogger(__name__)


async def fetch_tool_schemas(rhoai_mcp_url: str) -> list[Tool]:
    """Fetch tool schemas from a running rhoai-mcp server via MCP SSE client.

    Args:
        rhoai_mcp_url: Base URL of the rhoai-mcp server (e.g. http://localhost:8000).

    Returns:
        List of mcp.types.Tool objects.
    """
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    sse_url = f"{rhoai_mcp_url.rstrip('/')}/sse"

    async with sse_client(sse_url) as (read_stream, write_stream):
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


def _lcs_tool_call_to_deepeval(tc: Any) -> MCPToolCall:
    """Convert an LCSToolCall to a DeepEval MCPToolCall."""
    result_text = str(tc.result)
    # Strip server_label from args - it's injected by LCS infrastructure
    # and is not part of the tool's input schema
    args = {k: v for k, v in tc.arguments.items() if k != "server_label"}
    return MCPToolCall(
        name=tc.name,
        args=args,
        result=CallToolResult(
            content=[TextContent(type="text", text=result_text)],
            structuredContent={"result": result_text},
        ),
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
                turn = Turn(
                    role="assistant",
                    content=content,
                    mcp_tools_called=mcp_tools_called,
                )
                # Work around DeepEval/Pydantic V2 bug where PrivateAttr
                # _mcp_interaction doesn't get set from model_validator data dict
                turn._mcp_interaction = True
                turns.append(turn)
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
