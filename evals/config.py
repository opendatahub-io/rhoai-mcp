"""Configuration for RHOAI MCP evaluation framework."""

from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """LLM provider for the agent under test."""

    OPENAI = "openai"
    VLLM = "vllm"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    ANTHROPIC_VERTEX = "anthropic-vertex"
    GOOGLE_GENAI = "google-genai"
    GOOGLE_VERTEX = "google-vertex"


class ClusterMode(str, Enum):
    """Whether to use mock or live K8s cluster."""

    MOCK = "mock"
    LIVE = "live"


class EvalConfig(BaseSettings):
    """Configuration for RHOAI MCP evaluation runs.

    Loaded from environment variables with RHOAI_EVAL_ prefix
    or from a .env.eval file.
    """

    model_config = SettingsConfigDict(
        env_prefix="RHOAI_EVAL_",
        env_file=".env.eval",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Agent LLM settings
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GOOGLE_GENAI,
        description="LLM provider for the agent",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Model name for the agent LLM",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the agent LLM",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Base URL for vLLM or Azure endpoint",
    )

    # Judge LLM settings (for DeepEval metrics)
    eval_model: str = Field(
        default="gemini-2.5-flash",
        description="Model name for the DeepEval judge LLM",
    )
    eval_model_base_url: str | None = Field(
        default=None,
        description="Base URL for the judge model endpoint",
    )
    eval_api_key: str = Field(
        default="",
        description="API key for the judge LLM",
    )

    # Judge LLM provider (defaults to same as agent provider)
    eval_provider: LLMProvider = Field(
        default=LLMProvider.GOOGLE_GENAI,
        description="LLM provider for the DeepEval judge",
    )

    # Vertex AI settings (for anthropic-vertex and google-vertex providers)
    vertex_project_id: str | None = Field(
        default=None,
        description="Google Cloud project ID for Vertex AI",
    )
    vertex_location: str = Field(
        default="us-central1",
        description="Google Cloud region for Vertex AI",
    )

    # Cluster settings
    cluster_mode: ClusterMode = Field(
        default=ClusterMode.MOCK,
        description="Whether to use mock or live K8s cluster",
    )

    # Metric thresholds
    mcp_use_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score for MCPUseMetric",
    )
    task_completion_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum score for MCPTaskCompletionMetric",
    )

    # Agent loop settings
    max_agent_turns: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum LLM turns per scenario",
    )
