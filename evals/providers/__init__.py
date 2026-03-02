"""LLM provider abstraction for the evaluation framework.

Only judge LLM creation is needed (agent loop is now handled by LCS).
"""

from evals.providers.factory import create_judge_llm

__all__ = ["create_judge_llm"]
