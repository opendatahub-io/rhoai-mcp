"""OpenAI-compatible agent LLM provider.

Covers OpenAI, Azure OpenAI, and vLLM endpoints — all use the
same OpenAI Python client with different base URLs.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from evals.config import EvalConfig, LLMProvider
from evals.providers.base import (
    AgentLLMProvider,
    ProviderResponse,
    ProviderToolCall,
    ToolCallResult,
)


class OpenAIAgentProvider(AgentLLMProvider):
    """Agent provider for OpenAI-compatible APIs (OpenAI, Azure, vLLM)."""

    def __init__(self, config: EvalConfig) -> None:
        self._model = config.llm_model
        self._client = self._create_client(config)

    @staticmethod
    def _create_client(config: EvalConfig) -> AsyncOpenAI:
        """Create an OpenAI-compatible async client."""
        kwargs: dict[str, Any] = {"api_key": config.llm_api_key}

        if config.llm_base_url:
            kwargs["base_url"] = config.llm_base_url
        elif config.llm_provider == LLMProvider.VLLM:
            raise ValueError("llm_base_url is required for vLLM provider")

        return AsyncOpenAI(**kwargs)

    def format_tools(self, mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MCP tools to OpenAI function-calling format."""
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )
        return openai_tools

    def build_initial_messages(
        self, system_prompt: str, user_task: str
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format initial messages."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_task},
        ]

    async def send(
        self, messages: list[Any], tools: Any
    ) -> ProviderResponse:
        """Send messages to an OpenAI-compatible endpoint."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools if tools else None,
        )

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ProviderToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return ProviderResponse(
            text=message.content,
            tool_calls=tool_calls,
            raw=message,
        )

    def append_assistant_message(
        self, messages: list[Any], response: ProviderResponse
    ) -> None:
        """Append the raw OpenAI message to the conversation."""
        messages.append(response.raw.model_dump(exclude_none=True))

    def append_tool_results(
        self,
        messages: list[Any],
        response: ProviderResponse,
        results: list[ToolCallResult],
    ) -> None:
        """Append tool results as OpenAI tool-role messages."""
        for result in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.id,
                    "content": result.result,
                }
            )

    def messages_for_deepeval(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Messages are already in OpenAI format."""
        return messages
