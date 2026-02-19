"""Factory functions for creating agent providers and judge LLMs.

Dispatches on LLMProvider enum values to instantiate the correct
provider or judge implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evals.config import LLMProvider

if TYPE_CHECKING:
    from evals.config import EvalConfig
    from evals.providers.base import AgentLLMProvider


def create_agent_provider(config: EvalConfig) -> AgentLLMProvider:
    """Create an agent LLM provider based on the configured provider type.

    Args:
        config: Evaluation configuration.

    Returns:
        An AgentLLMProvider instance for the configured provider.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.llm_provider

    if provider in (LLMProvider.OPENAI, LLMProvider.VLLM, LLMProvider.AZURE):
        from evals.providers.openai_provider import OpenAIAgentProvider

        return OpenAIAgentProvider(config)

    if provider in (LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC_VERTEX):
        from evals.providers.anthropic_provider import AnthropicAgentProvider

        return AnthropicAgentProvider(config)

    if provider in (LLMProvider.GOOGLE_GENAI, LLMProvider.GOOGLE_VERTEX):
        from evals.providers.google_provider import GoogleAgentProvider

        return GoogleAgentProvider(config)

    raise ValueError(f"Unsupported agent LLM provider: {provider}")


def create_judge_llm(config: EvalConfig) -> Any:
    """Create a judge LLM for DeepEval metrics.

    For plain OpenAI (no custom base URL), returns the model name string
    so DeepEval uses its built-in OpenAI integration. For all other
    providers, returns a DeepEvalBaseLLM instance.

    Args:
        config: Evaluation configuration.

    Returns:
        A model name string or DeepEvalBaseLLM instance.
    """
    provider = config.eval_provider

    if provider == LLMProvider.VLLM and not config.eval_model_base_url:
        raise ValueError(
            "VLLM judge provider requires eval_model_base_url to be set. "
            "Set RHOAI_EVAL_EVAL_MODEL_BASE_URL to your vLLM endpoint (e.g. http://localhost:8000/v1)."
        )

    if provider in (LLMProvider.OPENAI, LLMProvider.VLLM, LLMProvider.AZURE):
        # If there's a custom base URL, wrap in OpenAIJudgeLLM
        if config.eval_model_base_url:
            from evals.providers.judge import OpenAIJudgeLLM

            return OpenAIJudgeLLM(
                model_name=config.eval_model,
                api_key=config.eval_api_key,
                base_url=config.eval_model_base_url,
            )
        # Plain OpenAI: return model name for DeepEval's built-in support
        return config.eval_model

    if provider in (LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC_VERTEX):
        from evals.providers.judge import AnthropicJudgeLLM

        return AnthropicJudgeLLM(
            model_name=config.eval_model,
            api_key=config.eval_api_key,
            vertex_project_id=config.vertex_project_id if provider == LLMProvider.ANTHROPIC_VERTEX else None,
            vertex_location=config.vertex_location,
        )

    if provider in (LLMProvider.GOOGLE_GENAI, LLMProvider.GOOGLE_VERTEX):
        from evals.providers.judge import GoogleJudgeLLM

        return GoogleJudgeLLM(
            model_name=config.eval_model,
            api_key=config.eval_api_key,
            vertex_project_id=config.vertex_project_id,
            vertex_location=config.vertex_location,
            use_vertex=provider == LLMProvider.GOOGLE_VERTEX,
        )

    raise ValueError(f"Unsupported judge LLM provider: {provider}")
