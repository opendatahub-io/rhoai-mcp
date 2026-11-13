"""Skill-specific judge defaults, without changing the legacy LCS suite."""

from pydantic import Field

from evals.config import DEFAULT_JUDGE_MODEL, EvalConfig, LLMProvider


class SkillEvalConfig(EvalConfig):
    eval_model: str = DEFAULT_JUDGE_MODEL
    eval_provider: LLMProvider = LLMProvider.ENMAAS
    mcp_use_threshold: float = Field(default=0.6, ge=0, le=1)
    task_completion_threshold: float = Field(default=0.75, ge=0, le=1)
