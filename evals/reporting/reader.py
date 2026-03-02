"""JSONL reader for eval result history."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from evals.reporting.models import (
    EnvironmentRecord,
    EvalRecord,
    GitRecord,
    MetricRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "eval_history.jsonl"


def load_records(path: str | None = None) -> list[EvalRecord]:
    """Load eval records from a JSONL file.

    Returns an empty list if the file doesn't exist.
    Handles both old (llm_provider/llm_model) and new (agent_backend/agent_model)
    field names for backwards compatibility.
    """
    p = Path(path) if path else DEFAULT_RESULTS_PATH
    if not p.exists():
        logger.debug(f"No eval history found at {p}")
        return []

    records: list[EvalRecord] = []
    for line_num, line in enumerate(p.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            env_data = data.get("environment", {})
            environment = EnvironmentRecord(
                agent_backend=env_data.get(
                    "agent_backend", env_data.get("llm_provider", "unknown")
                ),
                agent_model=env_data.get(
                    "agent_model", env_data.get("llm_model", "unknown")
                ),
                lcs_url=env_data.get("lcs_url", ""),
                eval_provider=env_data.get("eval_provider", "unknown"),
                eval_model=env_data.get("eval_model", "unknown"),
                cluster_mode=env_data.get("cluster_mode", "unknown"),
                mcp_use_threshold=env_data.get("mcp_use_threshold", 0.5),
                task_completion_threshold=env_data.get("task_completion_threshold", 0.6),
            )
            record = EvalRecord(
                run_id=data["run_id"],
                timestamp=data["timestamp"],
                scenario=data["scenario"],
                git=GitRecord(**data["git"]),
                environment=environment,
                metrics=[MetricRecord(**m) for m in data.get("metrics", [])],
                turns=data.get("turns", 0),
                tool_names_used=data.get("tool_names_used", []),
                passed=data.get("passed", False),
                duration_seconds=data.get("duration_seconds", 0.0),
            )
            records.append(record)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Skipping malformed record at line {line_num}: {e}")
    return records
