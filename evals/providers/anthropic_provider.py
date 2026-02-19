"""Anthropic agent LLM provider.

Covers direct Anthropic API and Anthropic on Vertex AI.
Both use the anthropic Python SDK — Vertex uses AnthropicVertex client.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic, AsyncAnthropicVertex

from evals.config import EvalConfig, LLMProvider
from evals.providers.base import (
    AgentLLMProvider,
    ProviderResponse,
    ProviderToolCall,
    ToolCallResult,
)

_MAX_TOKENS = 4096


class AnthropicAgentProvider(AgentLLMProvider):
    """Agent provider for Anthropic Claude (direct API and Vertex AI)."""

    def __init__(self, config: EvalConfig) -> None:
        self._model = config.llm_model
        self._client = self._create_client(config)
        # Anthropic requires system prompt as a separate parameter,
        # so we store it and exclude it from the messages list.
        self._system_prompt: str = ""

    @staticmethod
    def _create_client(config: EvalConfig) -> AsyncAnthropic | AsyncAnthropicVertex:
        """Create an Anthropic async client."""
        if config.llm_provider == LLMProvider.ANTHROPIC_VERTEX:
            if not config.vertex_project_id:
                raise ValueError("vertex_project_id is required for anthropic-vertex provider")
            return AsyncAnthropicVertex(
                project_id=config.vertex_project_id,
                region=config.vertex_location,
            )
        return AsyncAnthropic(api_key=config.llm_api_key)

    def format_tools(self, mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MCP tools to Anthropic tool format."""
        anthropic_tools = []
        for tool in mcp_tools:
            anthropic_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
            )
        return anthropic_tools

    def build_initial_messages(
        self, system_prompt: str, user_task: str
    ) -> list[dict[str, Any]]:
        """Build Anthropic-format initial messages.

        System prompt is stored separately since Anthropic takes it
        as a top-level parameter, not as a message.
        """
        self._system_prompt = system_prompt
        return [{"role": "user", "content": user_task}]

    async def send(
        self, messages: list[Any], tools: Any
    ) -> ProviderResponse:
        """Send messages to the Anthropic API."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "messages": messages,
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ProviderToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return ProviderResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw=response,
        )

    def append_assistant_message(
        self, messages: list[Any], response: ProviderResponse
    ) -> None:
        """Append the assistant response as an Anthropic message."""
        # Anthropic expects the full content blocks list
        content = []
        for block in response.raw.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input if isinstance(block.input, dict) else {},
                    }
                )
        messages.append({"role": "assistant", "content": content})

    def append_tool_results(
        self,
        messages: list[Any],
        response: ProviderResponse,
        results: list[ToolCallResult],
    ) -> None:
        """Append tool results as an Anthropic user message with tool_result blocks."""
        tool_result_blocks = []
        for result in results:
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.id,
                    "content": result.result,
                }
            )
        messages.append({"role": "user", "content": tool_result_blocks})

    def messages_for_deepeval(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert Anthropic messages to OpenAI-style dicts for DeepEval."""
        openai_messages: list[dict[str, Any]] = []

        # Re-inject the system prompt as an OpenAI system message
        if self._system_prompt:
            openai_messages.append({"role": "system", "content": self._system_prompt})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                # Could be a plain text message or tool_result blocks
                if isinstance(content, str):
                    openai_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # Check if these are tool_result blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": block.get("content", ""),
                                }
                            )

            elif role == "assistant":
                if isinstance(content, str):
                    openai_messages.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    # Extract text and tool_use blocks
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                tool_calls.append(
                                    {
                                        "id": block.get("id", ""),
                                        "type": "function",
                                        "function": {
                                            "name": block.get("name", ""),
                                            "arguments": json.dumps(
                                                block.get("input", {})
                                            ),
                                        },
                                    }
                                )

                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                    }
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    openai_messages.append(assistant_msg)

        return openai_messages
