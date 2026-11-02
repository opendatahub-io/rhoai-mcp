"""JSONL reader for eval result history."""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

from evals.reporting.models import (
    CheckRecord,
    EnvironmentRecord,
    EvalRecord,
    GitRecord,
    MetricRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "eval_history.jsonl"


def record_from_dict(data: dict[str, Any]) -> EvalRecord:
    """Read known schema fields, including legacy LCS environment aliases."""
    if not isinstance(data, dict):
        raise TypeError("An eval record must be an object")
    env: dict[str, Any] = data.get("environment", {})
    if not isinstance(env, dict):
        raise TypeError("An eval environment must be an object")
    environment = EnvironmentRecord(
        agent_backend=env.get("agent_backend") or env.get("llm_provider") or "unknown",
        agent_model=env.get("agent_model") or env.get("llm_model") or "unknown",
        lcs_url=env.get("lcs_url", ""),
        eval_provider=env.get("eval_provider", "unknown"),
        eval_model=env.get("eval_model", "unknown"),
        cluster_mode=env.get("cluster_mode", "unknown"),
        mcp_use_threshold=env.get("mcp_use_threshold", 0.5),
        task_completion_threshold=env.get("task_completion_threshold", 0.6),
        eval_model_base_url=env.get("eval_model_base_url", ""),
    )
    nested = {"environment", "git", "metrics", "checks"}
    values = {
        f.name: data[f.name] for f in fields(EvalRecord) if f.name not in nested and f.name in data
    }
    return EvalRecord(
        **values,
        environment=environment,
        git=GitRecord(**data["git"]),
        metrics=[MetricRecord(**m) for m in data.get("metrics", [])],
        checks=[CheckRecord(**c) for c in data.get("checks", [])],
    )


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
            records.append(record_from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Skipping malformed record at line {line_num}: {e}")
    return records
