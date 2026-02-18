"""LLM agent wrapper for RHOAI MCP evaluations.

Implements a provider-agnostic agent loop that sends tasks and tool
schemas to any supported LLM, processes tool calls via the MCP harness,
and records tool calls and conversation turns for DeepEval metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from evals.config import EvalConfig
from evals.mcp_harness import MCPHarness
from evals.providers import create_agent_provider
from evals.providers.base import AgentLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an AI assistant interacting with a Red Hat OpenShift AI "
    "(RHOAI) environment through MCP tools. Use the available tools "
    "to complete the user's request. Call tools as needed, then provide "
    "a final summary of what you found or accomplished."
)


@dataclass
class ToolCall:
    """Record of a single tool call made by the agent."""

    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class AgentResult:
    """Result of running an agent on a task."""

    task: str
    final_output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0

    @property
    def tool_names_used(self) -> list[str]:
        """Get the list of tool names called, in order."""
        return [tc.name for tc in self.tool_calls]


class MCPAgent:
    """LLM agent that interacts with the MCP server via tool calling.

    Uses a provider abstraction to support OpenAI, Anthropic, and
    Google Gemini LLMs.
    """

    def __init__(self, config: EvalConfig, harness: MCPHarness) -> None:
        self._config = config
        self._harness = harness
        self._provider: AgentLLMProvider = create_agent_provider(config)

    async def run(self, task: str) -> AgentResult:
        """Run the agent on a task until completion or max turns.

        The agent loop:
        1. Send the task + tool schemas to the LLM
        2. If the LLM returns tool calls, execute them and feed results back
        3. Repeat until the LLM responds with plain text or max turns reached

        Args:
            task: The natural language task for the agent to perform.

        Returns:
            AgentResult with tool calls, messages, and final output.
        """
        mcp_tools = self._harness.list_tools()
        tools = self._provider.format_tools(mcp_tools)
        messages = self._provider.build_initial_messages(_SYSTEM_PROMPT, task)

        result = AgentResult(task=task, final_output="")
        max_turns = self._config.max_agent_turns

        for turn in range(max_turns):
            result.turns = turn + 1
            logger.debug(f"Agent turn {turn + 1}/{max_turns}")

            response = await self._provider.send(messages, tools)

            # Add assistant message to history
            self._provider.append_assistant_message(messages, response)

            # If the model didn't call any tools, we're done
            if not response.has_tool_calls:
                result.final_output = response.text or ""
                break

            # Process each tool call
            tool_call_results: list[ToolCallResult] = []
            for tc in response.tool_calls:
                logger.debug(f"Calling tool: {tc.name}({tc.arguments})")
                tool_result = await self._harness.call_tool(tc.name, tc.arguments)

                record = ToolCall(name=tc.name, arguments=tc.arguments, result=tool_result)
                result.tool_calls.append(record)

                tool_call_results.append(
                    ToolCallResult(id=tc.id, name=tc.name, result=tool_result)
                )

            # Feed tool results back to the LLM
            self._provider.append_tool_results(messages, response, tool_call_results)
        else:
            # Max turns reached without a final text response
            result.final_output = (
                f"Agent reached maximum turns ({max_turns}) without completing the task."
            )

        result.messages = self._provider.messages_for_deepeval(messages)
        return result
