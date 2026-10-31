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

from evals.agent.transcript import AgentTranscript, ToolCallRecord
from evals.lcs_client import LCSResult

logger = logging.getLogger(__name__)


class NothingToScoreError(ValueError):
    """Raised when a transcript has no turns and so cannot be scored.

    A harness error can legitimately end an attempt before any message exists
    (failed agent construction, failed MCP handshake). The artifact is still
    saved, but DeepEval rejects an empty ``ConversationalTestCase``, so callers
    must record ``n/a`` with the missing precondition rather than a zero score
    (plan §3.3).
    """


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
                    mcp_tools_called.append(_lcs_tool_call_to_deepeval(result.tool_calls[tc_index]))
                    tc_index += 1

            if mcp_tools_called:
                # No _mcp_interaction assignment: as of deepeval 4.3.1 it is a
                # derived read-only property that is already True whenever
                # mcp_tools_called is non-empty, and assigning to it raises.
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


def _record_to_mcp_tool_call(record: ToolCallRecord) -> MCPToolCall:
    """Convert a shared-contract tool call into a typed DeepEval MCPToolCall.

    DeepEval validates by isinstance against ``mcp.types``, so the result must be
    a real ``CallToolResult``; a dict-shaped result is rejected. Structured
    content and the error flag are preserved rather than flattened to a string,
    since ``MultiTurnMCPUseMetric`` scores arguments as well as tool choice.
    """
    structured = record.structured_result
    payload = structured if structured is not None else record.result_text
    return MCPToolCall(
        name=record.name,
        args=record.arguments,
        result=CallToolResult(
            content=[TextContent(type="text", text=record.result_text or "")],
            # DeepEval reads ``structuredContent["result"]`` unconditionally
            # (metrics/mcp/utils.py: no .get(), no fallback), so the payload
            # must be wrapped under that key or every MCP metric raises
            # KeyError before the judge is even called. Passing the raw tool
            # result straight through looks more faithful but breaks scoring.
            structuredContent={"result": payload},
            isError=record.is_error,
        ),
    )


def transcript_to_conversational_test_case(
    transcript: AgentTranscript,
    mcp_server: MCPServer,
    *,
    include_skill_body: bool = False,
    skill: str | None = None,
) -> ConversationalTestCase:
    """Build a DeepEval conversational test case from a shared transcript.

    Calls are already matched to their own turn by call ID upstream, and harness
    tools are already separated from MCP tools, so this is a straight mapping.

    Args:
        transcript: The adapted transcript, normally rebuilt from the saved
            attempt artifact so live scoring and re-scoring share one path.
        mcp_server: Tool schemas captured from the same server session.
        include_skill_body: Whether to prepend the skill instructions the model
            actually received. Required for the skill-fidelity rubric, which
            cannot judge adherence to a document it has not been shown; left off
            for the MCP metrics, which only score tool use.
        skill: Which loaded skill's body to include. Defaults to the first one.

    Returns:
        A ``ConversationalTestCase`` whose assistant turns carry only genuine MCP
        interactions. Skill loading stays visible as conversational context but
        is never counted as an MCP tool call.
    """
    if not transcript.turns:
        raise NothingToScoreError(
            "transcript has no turns; the attempt produced no conversation "
            "(check ConversationResult.outcome and error_stage)"
        )

    turns: list[Turn] = []
    # The skill body is inserted *after* the opening user turn, not before it.
    # DeepEval builds a unit's `task` by consuming leading user turns and
    # stopping at the first non-user turn, so prepending an assistant turn would
    # leave the first unit with an empty task and demote the real opening
    # message to a step. Placing it after is also truthful: the agent loaded the
    # skill in response to that first message.
    pending_body = transcript.skill_body(skill) if include_skill_body else None

    for source in transcript.turns:
        if source.role == "user":
            turns.append(Turn(role="user", content=source.content))
            if pending_body:
                turns.append(
                    Turn(
                        role="assistant",
                        content=(f"[skill instructions delivered to the agent]\n\n{pending_body}"),
                    )
                )
                pending_body = None
            continue

        mcp_tools_called = [_record_to_mcp_tool_call(c) for c in source.mcp_tool_calls]
        content = source.content
        if not content and source.harness_tool_calls and not mcp_tools_called:
            # Keep the skill-loading step in the conversation without letting an
            # empty assistant turn look like a non-response.
            loaded = ", ".join(
                str(c.arguments.get("id") or c.arguments.get("name") or c.name)
                for c in source.harness_tool_calls
            )
            content = f"[loaded skill: {loaded}]"

        if mcp_tools_called:
            turns.append(Turn(role="assistant", content=content, mcp_tools_called=mcp_tools_called))
        elif content:
            turns.append(Turn(role="assistant", content=content))

    return ConversationalTestCase(turns=turns, mcp_servers=[mcp_server])


def lcs_result_to_single_turn_test_case(
    result: LCSResult,
    mcp_server: MCPServer,
) -> LLMTestCase:
    """Convert an LCSResult into a single-turn LLMTestCase.

    Used for simpler scenarios where multi-turn tracking isn't needed.
    """
    mcp_tools_called = [_lcs_tool_call_to_deepeval(tc) for tc in result.tool_calls]

    return LLMTestCase(
        input=result.task,
        actual_output=result.final_output,
        mcp_servers=[mcp_server],
        mcp_tools_called=mcp_tools_called,
    )
