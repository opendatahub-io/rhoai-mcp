"""DeepEvalBaseLLM subclasses for judge LLMs across providers.

These wrap various LLM APIs to serve as the judge model for DeepEval
metrics. Only text generation is needed (no tool calling).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from deepeval.models import DeepEvalBaseLLM


def _run_async(coro: Any) -> Any:
    """Run a coroutine, handling both sync and async calling contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class OpenAIJudgeLLM(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for OpenAI-compatible endpoints.

    Replaces the former CustomEvalLLM in evals/metrics/custom_llm.py.

    Holds a **sync client alongside the async one**. DeepEval drives metrics
    synchronously by default, and ``_run_async`` starts a fresh event loop per
    call; an ``AsyncOpenAI`` client pools connections against the loop that
    created them, so the second sync call fails with
    ``RuntimeError: Event loop is closed``. Using the sync client on the sync
    path avoids the loop entirely.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._model_name = model_name
        from openai import AsyncOpenAI, OpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._sync_client = OpenAI(api_key=api_key, base_url=base_url)

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self) -> Any:
        return self._client

    def _prepare(self, prompt: str, schema: Any, kwargs: dict[str, Any]) -> str:
        """Apply JSON-mode settings when a schema is requested."""
        if schema is not None:
            # OpenAI-compatible JSON mode requires "json" to appear in the prompt.
            kwargs.setdefault("response_format", {"type": "json_object"})
            if "json" not in prompt.lower():
                prompt = f"{prompt}\n\nRespond with a single JSON object."
        return prompt

    def _parse(self, content: str, schema: Any) -> Any:
        if schema is None:
            return content
        try:
            return schema.model_validate_json(content)
        except Exception:  # noqa: BLE001
            # Returning the text lets DeepEval's trimAndLoadJson salvage a
            # response wrapped in prose or a code fence.
            return content

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        """Generate a judge response, optionally parsed into a pydantic schema.

        DeepEval metrics call ``a_generate_with_schema(prompt, schema=...)``.
        The base class forwards ``schema`` here and falls back to a plain call
        only if this raises ``TypeError`` — so accepting it explicitly is what
        keeps judging on the structured path instead of string-scraping.

        Args:
            prompt: The judge prompt.
            schema: Optional pydantic model the response must conform to.
            **kwargs: Passed through to the chat completion.

        Returns:
            A ``schema`` instance when one is requested and the response parses,
            otherwise the raw text for DeepEval's own JSON salvage to handle.
        """
        prompt = self._prepare(prompt, schema, kwargs)
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return self._parse(response.choices[0].message.content or "", schema)

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        """Synchronous generation, used by DeepEval's default (non-async) path.

        Deliberately does not route through ``a_generate``: see the class
        docstring on event-loop reuse.
        """
        prompt = self._prepare(prompt, schema, kwargs)
        response = self._sync_client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return self._parse(response.choices[0].message.content or "", schema)


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
        return _run_async(self.a_generate(prompt, **kwargs))


class GoogleJudgeLLM(DeepEvalBaseLLM):
    """DeepEval LLM wrapper for Google Gemini (API key and Vertex)."""

    def __init__(
        self,
        model_name: str,
        api_key: str = "",
        vertex_project_id: str | None = None,
        vertex_location: str = "us-central1",
        use_vertex: bool = False,
    ) -> None:
        self._model_name = model_name
        from google import genai

        if use_vertex:
            if api_key:
                # Vertex AI Express mode: API key auth (no project/location needed)
                self._client = genai.Client(vertexai=True, api_key=api_key)
            elif vertex_project_id:
                self._client = genai.Client(
                    vertexai=True,
                    project=vertex_project_id,
                    location=vertex_location,
                )
            else:
                raise ValueError("google-vertex provider requires an API key or vertex_project_id")
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
        return _run_async(self.a_generate(prompt, **kwargs))
