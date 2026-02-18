"""DeepEvalBaseLLM subclasses for judge LLMs across providers.

These wrap various LLM APIs to serve as the judge model for DeepEval
metrics. Only text generation is needed (no tool calling).
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepeval.models import DeepEvalBaseLLM


class OpenAIJudgeLLM(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for OpenAI-compatible endpoints.

    Replaces the former CustomEvalLLM in evals/metrics/custom_llm.py.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._model_name = model_name
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self) -> Any:
        return self._client

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self.a_generate(prompt, **kwargs)
        )


class AnthropicJudgeLLM(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for Anthropic Claude (direct and Vertex)."""

    def __init__(
        self,
        model_name: str,
        api_key: str = "",
        vertex_project_id: str | None = None,
        vertex_location: str = "us-central1",
    ) -> None:
        self._model_name = model_name
        from anthropic import AsyncAnthropic, AsyncAnthropicVertex

        if vertex_project_id:
            self._client: AsyncAnthropic | AsyncAnthropicVertex = AsyncAnthropicVertex(
                project_id=vertex_project_id,
                region=vertex_location,
            )
        else:
            self._client = AsyncAnthropic(api_key=api_key)

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self) -> Any:
        return self._client

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        response = await self._client.messages.create(
            model=self._model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self.a_generate(prompt, **kwargs)
        )


class GoogleJudgeLLM(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for Google Gemini (API key and Vertex)."""

    def __init__(
        self,
        model_name: str,
        api_key: str = "",
        vertex_project_id: str | None = None,
        vertex_location: str = "us-central1",
    ) -> None:
        self._model_name = model_name
        from google import genai

        if vertex_project_id:
            self._client = genai.Client(
                vertexai=True,
                project=vertex_project_id,
                location=vertex_location,
            )
        else:
            self._client = genai.Client(api_key=api_key)

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self) -> Any:
        return self._client

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )
        return response.text or ""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self.a_generate(prompt, **kwargs)
        )
