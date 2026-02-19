"""Data models for eval result recording.

Dataclasses matching the JSONL schema for eval result persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricRecord:
    """A single metric result from a DeepEval evaluation."""

    name: str
    score: float
    success: bool
    threshold: float
    reason: str


@dataclass
class GitRecord:
    """Git metadata for the eval run."""

    commit: str
    branch: str


@dataclass
class EnvironmentRecord:
    """Eval environment configuration snapshot."""

    llm_provider: str
    llm_model: str
    eval_provider: str
    eval_model: str
    cluster_mode: str
    mcp_use_threshold: float
    task_completion_threshold: float
    max_agent_turns: int


@dataclass
class EvalRecord:
    """A single scenario evaluation result, one JSONL line."""

    run_id: str
    timestamp: str
    scenario: str
    git: GitRecord
    environment: EnvironmentRecord
    metrics: list[MetricRecord] = field(default_factory=list)
    turns: int = 0
    tool_names_used: list[str] = field(default_factory=list)
    passed: bool = False
    duration_seconds: float = 0.0
