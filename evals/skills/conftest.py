"""Fixtures for the stdio skill evaluations.

These deliberately **override** the LCS/HTTP fixtures in ``evals/conftest.py``:
this path drives ``rhoai-mcp`` as a stdio subprocess and needs no LCORE stack,
no compose, and no running server (plan §2.1). Tool schemas come from the same
MCP session the conversation runs against, not from a separate HTTP endpoint.

Judge cost is opt-in. By default these run as a **dry run**: the agent runs, the
artifact is written, and the deterministic checks grade it — no judge, no spend.
Set ``RHOAI_EVAL_SKILL_JUDGE=1`` to add the judged metrics.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from evals.config import EvalConfig
    from evals.reporting.recorder import EvalRecorder

#: Opt in to judged metrics. Absent or falsey means dry run.
JUDGE_ENV_VAR = "RHOAI_EVAL_SKILL_JUDGE"


def judge_enabled() -> bool:
    return os.environ.get(JUDGE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


@pytest.fixture(scope="session")
def dry_run() -> bool:
    """Whether this session skips judged metrics (and so can never 'pass')."""
    return not judge_enabled()


@pytest.fixture(scope="session")
def skill_run_id() -> str:
    """One id for the whole sweep, shared by every attempt's artifact."""
    return uuid.uuid4().hex[:12]


@pytest.fixture(scope="session")
def agent_specs() -> list[Any]:
    """Model specs selected by ``RHOAI_EVAL_AGENT_MODELS`` (plan §2.3)."""
    from evals.agent.models import selected_specs

    return selected_specs()


def pytest_generate_tests(metafunc: Any) -> None:
    """Parametrise over the selected models so CI can run one per job."""
    if "agent_spec" not in metafunc.fixturenames:
        return
    from evals.agent.models import selected_specs

    specs = selected_specs()
    metafunc.parametrize("agent_spec", specs, ids=[s.id for s in specs])


@pytest.fixture(scope="session")
def skill_eval_config() -> EvalConfig:
    """Eval config.

    Note this does **not** require the LCS settings: the stdio path ignores
    ``lcs_url`` and ``rhoai_mcp_url`` entirely.
    """
    from evals.config import EvalConfig

    return EvalConfig()


@pytest.fixture(scope="session")
def skill_recorder(skill_eval_config: EvalConfig, skill_run_id: str) -> EvalRecorder:
    """Recorder writing one JSONL line per attempt, including failures."""
    from evals.reporting.recorder import EvalRecorder

    recorder = EvalRecorder(skill_eval_config)
    recorder.run_id = skill_run_id
    return recorder


@pytest.fixture(scope="session")
def tracing_session() -> Any:
    """OTel tracing, if ``RHOAI_EVAL_OTLP_ENDPOINT`` is set.

    Yields a disabled no-op session otherwise, so tests never branch on it. The
    teardown flush matters: ``BatchSpanProcessor`` buffers, so without it a
    short run exits before its spans reach the collector.
    """
    from evals.agent.tracing import setup_tracing

    session = setup_tracing()
    yield session
    session.shutdown()


@pytest.fixture(scope="session")
def judge_llm(skill_eval_config: EvalConfig) -> Any:
    """The judge, held fixed across agent models (plan §2.3).

    Built once per session so a misconfigured judge fails fast rather than once
    per scenario.
    """
    if not judge_enabled():
        return None
    from evals.providers import create_judge_llm

    return create_judge_llm(skill_eval_config)
