"""Helpers to convert agent results into DeepEval test cases."""

from __future__ import annotations

from typing import Any

from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    MCPServer,
    MCPToolCall,
    Turn,
)

from evals.agent import AgentResult
from evals.mcp_harness import MCPHarness


def build_mcp_server(harness: MCPHarness) -> MCPServer:
    """Build a DeepEval MCPServer from the harness's registered tools."""
    tools = harness.list_tools()

    # DeepEval MCPServer accepts tool dicts or MCP Tool objects.
    # We provide dicts with name, description, inputSchema.
    tool_objects: list[dict[str, Any]] = [
        {
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["parameters"],
        }
        for t in tools
    ]

    return MCPServer(
        server_name="rhoai-mcp",
        available_tools=tool_objects,
    )


def _agent_tool_call_to_deepeval(tc: Any) -> MCPToolCall:
    """Convert an agent ToolCall to a DeepEval MCPToolCall."""
    return MCPToolCall(
        name=tc.name,
        args=tc.arguments,
        result=tc.result,
    )


def result_to_conversational_test_case(
    result: AgentResult,
    mcp_server: MCPServer,
) -> ConversationalTestCase:
    """Convert an AgentResult into a DeepEval ConversationalTestCase.

    Maps the agent's message history into Turn objects, attaching
    MCPToolCall records to assistant turns that made tool calls.
    """
    turns: list[Turn] = []
    # Track tool calls by index in the result
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
                        _agent_tool_call_to_deepeval(result.tool_calls[tc_index])
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

        # Skip "tool" role messages - they're represented in MCPToolCall.result

    return ConversationalTestCase(
        turns=turns,
        mcp_servers=[mcp_server],
    )


def result_to_single_turn_test_case(
    result: AgentResult,
    mcp_server: MCPServer,
) -> LLMTestCase:
    """Convert an AgentResult into a single-turn LLMTestCase.

    Used for simpler scenarios where multi-turn tracking isn't needed.
    """
    mcp_tools_called = [
        _agent_tool_call_to_deepeval(tc) for tc in result.tool_calls
    ]

    return LLMTestCase(
        input=result.task,
        actual_output=result.final_output,
        mcp_servers=[mcp_server],
        mcp_tools_called=mcp_tools_called,
    )
