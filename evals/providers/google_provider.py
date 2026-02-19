"""Google GenAI agent LLM provider.

Covers Google Gemini via API key and Google Gemini on Vertex AI.
Both use the google-genai SDK — Vertex uses vertexai=True with project/location.
"""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from evals.config import EvalConfig, LLMProvider
from evals.providers.base import (
    AgentLLMProvider,
    ProviderResponse,
    ProviderToolCall,
    ToolCallResult,
)


def _strip_unsupported_schema_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON Schema fields not supported by Google GenAI.

    Google's FunctionDeclaration doesn't support 'additionalProperties',
    'default', '$schema', and some other standard JSON Schema fields.
    """
    unsupported = {"additionalProperties", "default", "$schema", "$ref", "$defs"}
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in unsupported:
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_unsupported_schema_fields(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_unsupported_schema_fields(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


class GoogleAgentProvider(AgentLLMProvider):
    """Agent provider for Google Gemini (API key and Vertex AI)."""

    def __init__(self, config: EvalConfig) -> None:
        self._model = config.llm_model
        self._client = self._create_client(config)
        self._system_prompt: str = ""

    @staticmethod
    def _create_client(config: EvalConfig) -> genai.Client:
        """Create a Google GenAI client."""
        if config.llm_provider == LLMProvider.GOOGLE_VERTEX:
            if not config.vertex_project_id:
                raise ValueError("vertex_project_id is required for google-vertex provider")
            return genai.Client(
                vertexai=True,
                project=config.vertex_project_id,
                location=config.vertex_location,
            )
        return genai.Client(api_key=config.llm_api_key)

    def format_tools(self, mcp_tools: list[dict[str, Any]]) -> list[types.Tool]:
        """Convert MCP tools to Google GenAI FunctionDeclaration format."""
        declarations = []
        for tool in mcp_tools:
            params = _strip_unsupported_schema_fields(tool["parameters"])
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=params,
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def build_initial_messages(
        self, system_prompt: str, user_task: str
    ) -> list[types.Content]:
        """Build Google GenAI initial messages.

        System prompt is stored separately since Google takes it as
        a config parameter, not as a message.
        """
        self._system_prompt = system_prompt
        return [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_task)],
            )
        ]

    async def send(
        self, messages: list[Any], tools: Any
    ) -> ProviderResponse:
        """Send messages to the Google GenAI API."""
        config = types.GenerateContentConfig(
            tools=tools if tools else None,
        )
        if self._system_prompt:
            config.system_instruction = self._system_prompt

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=messages,
            config=config,
        )

        text_parts = []
        tool_calls = []

        if response.candidates and response.candidates[0].content:
            for i, part in enumerate(response.candidates[0].content.parts):
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    fc = part.function_call
                    # Google FunctionCall has no id field; synthesize one
                    synthetic_id = f"{fc.name}_{i}"
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ProviderToolCall(
                            id=synthetic_id,
                            name=fc.name,
                            arguments=args,
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
        """Append the model's response Content to the messages."""
        if response.raw.candidates and response.raw.candidates[0].content:
            messages.append(response.raw.candidates[0].content)

    def append_tool_results(
        self,
        messages: list[Any],
        response: ProviderResponse,
        results: list[ToolCallResult],
    ) -> None:
        """Append tool results as Google function response parts."""
        parts = []
        for result in results:
            # Parse JSON result back to dict if possible
            try:
                result_data = json.loads(result.result)
            except (json.JSONDecodeError, TypeError):
                result_data = {"result": result.result}

            parts.append(
                types.Part.from_function_response(
                    name=result.name,
                    response=result_data,
                )
            )
        messages.append(types.Content(role="user", parts=parts))

    def messages_for_deepeval(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Convert Google Content objects to OpenAI-style dicts for DeepEval."""
        openai_messages: list[dict[str, Any]] = []

        # Re-inject the system prompt
        if self._system_prompt:
            openai_messages.append({"role": "system", "content": self._system_prompt})

        for msg in messages:
            if isinstance(msg, types.Content):
                role = "assistant" if msg.role == "model" else msg.role or "user"

                text_parts = []
                tool_calls = []
                tool_results = []

                for i, part in enumerate(msg.parts):
                    if part.text:
                        text_parts.append(part.text)
                    elif part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            {
                                "id": f"{fc.name}_{i}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(
                                        dict(fc.args) if fc.args else {}
                                    ),
                                },
                            }
                        )
                    elif part.function_response:
                        fr = part.function_response
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": f"{fr.name}_response",
                                "content": json.dumps(
                                    dict(fr.response) if fr.response else {}
                                ),
                            }
                        )

                # Tool results go as separate tool messages
                if tool_results:
                    openai_messages.extend(tool_results)
                else:
                    assistant_msg: dict[str, Any] = {
                        "role": role,
                        "content": "\n".join(text_parts) if text_parts else None,
                    }
                    if tool_calls:
                        assistant_msg["tool_calls"] = tool_calls
                    openai_messages.append(assistant_msg)
            elif isinstance(msg, dict):
                # Already dict-based (shouldn't happen normally, but handle gracefully)
                openai_messages.append(msg)

        return openai_messages
