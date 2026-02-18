"""Base classes and dataclasses for LLM provider abstraction.

Defines the AgentLLMProvider ABC that each provider implements,
plus shared data structures for tool calls and responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderToolCall:
    """A tool call extracted from a provider response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    """Result of executing a tool call."""

    id: str
    name: str
    result: str


@dataclass
class ProviderResponse:
    """Normalized response from any LLM provider."""

    text: str | None
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the response contains tool calls."""
        return len(self.tool_calls) > 0


class AgentLLMProvider(ABC):
    """Abstract base class for LLM providers used by the agent loop.

    Each provider implements format conversion between MCP tool schemas
    and its native format, message construction, and API communication.
    """

    @abstractmethod
    def format_tools(self, mcp_tools: list[dict[str, Any]]) -> Any:
        """Convert MCP tool definitions to the provider's native tool format.

        Args:
            mcp_tools: List of dicts with 'name', 'description', 'parameters'.

        Returns:
            Provider-specific tool definitions.
        """

    @abstractmethod
    def build_initial_messages(self, system_prompt: str, user_task: str) -> list[Any]:
        """Build the initial message list for a new conversation.

        Args:
            system_prompt: System instructions for the agent.
            user_task: The user's task description.

        Returns:
            Provider-specific message list.
        """

    @abstractmethod
    async def send(self, messages: list[Any], tools: Any) -> ProviderResponse:
        """Send messages to the LLM and return a normalized response.

        Args:
            messages: Provider-specific message list.
            tools: Provider-specific tool definitions.

        Returns:
            Normalized ProviderResponse.
        """

    @abstractmethod
    def append_assistant_message(
        self, messages: list[Any], response: ProviderResponse
    ) -> None:
        """Append the assistant's response to the message list in-place.

        Args:
            messages: Provider-specific message list to modify.
            response: The provider response to append.
        """

    @abstractmethod
    def append_tool_results(
        self,
        messages: list[Any],
        response: ProviderResponse,
        results: list[ToolCallResult],
    ) -> None:
        """Append tool execution results to the message list in-place.

        Args:
            messages: Provider-specific message list to modify.
            response: The provider response that triggered the tool calls.
            results: Results from executing each tool call.
        """

    @abstractmethod
    def messages_for_deepeval(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert provider-specific messages to OpenAI-style dicts.

        DeepEval expects OpenAI-format messages for its test cases.

        Args:
            messages: Provider-specific message list.

        Returns:
            List of OpenAI-style message dicts.
        """
