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

    agent_backend: str
    agent_model: str
    lcs_url: str
    eval_provider: str
    eval_model: str
    cluster_mode: str
    mcp_use_threshold: float
    task_completion_threshold: float


@dataclass
class CheckRecord:
    """One named deterministic check result (plan §3.1).

    ``status`` is ``pass`` | ``fail`` | ``n/a`` | ``unexercised``. The last two
    are deliberately distinct from ``pass``: a conditional branch that never ran
    is not coverage, and a check whose precondition was absent is not a success.
    """

    name: str
    status: str
    detail: str = ""


#: Attempt outcomes, kept distinct from pass/fail (plan §3.4). Only an
#: ``evaluated`` attempt is meaningfully a pass or a fail; the rest describe why
#: the attempt did not reach a verdict. A model failure is an evaluation result,
#: not automatically a harness bug.
OUTCOME_EVALUATED = "evaluated"
OUTCOME_HARNESS_ERROR = "harness_error"
OUTCOME_MODEL_FAILURE = "model_failure"
OUTCOME_ROUTING_FAILURE = "routing_failure"


@dataclass
class EvalRecord:
    """A single scenario evaluation result, one JSONL line.

    The fields below ``duration_seconds`` were added for the skill-eval path and
    are absent from historical LCS records; ``reader.load_records`` supplies
    defaults so old lines still parse (plan §3.5).
    """

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

    # -- skill-eval additions (plan §3.5) ----------------------------------- #

    attempt_id: str = ""
    """Identifies the agent run. A re-score reuses it."""
    scoring_id: str = ""
    """Identifies this scoring pass, so re-scores of one attempt stay distinct."""
    skill: str = ""
    golden_id: str = ""
    skills_loaded: list[str] = field(default_factory=list)
    outcome: str = OUTCOME_EVALUATED
    error_stage: str = ""
    error_detail: str = ""
    checks: list[CheckRecord] = field(default_factory=list)
    artifact_path: str = ""
    trace_id: str = ""
    """MLflow session grouping this attempt's per-turn traces. Equal to
    ``attempt_id`` by construction, since tracing follows MLflow's model of one
    trace per turn rather than one per conversation (plan §3.6)."""
    dry_run: bool = False
    """True when no judge ran. Such a record carries no metric scores and must
    never be read as a judged pass."""
    agent_seconds: float = 0.0
    """Agent execution time, separate from scoring time (``duration_seconds``)."""
    skill_sha: str = ""
    scenario_sha: str = ""
    rubric_version: str = ""
