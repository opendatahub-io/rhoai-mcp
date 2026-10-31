"""Drive a scripted multi-turn conversation against the stdio rhoai-mcp server.

One agent, both skills in the catalog, the full default MCP tool surface, and a
fixed script of user replies. The agent must discover which skill applies, load
it, and follow it (plan §2.1 "discovery mode").

Lifecycle rules this module exists to enforce:

- **One server per attempt.** A stdio server restarted mid-conversation loses
  mock-cluster state, and state from a previous scenario must never leak into
  the next one.
- **One agent context across all turns.** Without an explicit ``async with``,
  toolsets open and close per run, which restarts the server.
- **History threaded, not concatenated.** Each turn passes the previous run's
  cumulative ``all_messages()``; concatenating successive snapshots duplicates
  earlier turns.
- **Partial history is preserved on failure.** A timeout, usage-limit breach or
  failed checkpoint still produces a saved artifact (plan §3.4, §3.6).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.agent.transcript import AgentTranscript, AttemptArtifact, from_pydantic_ai

if TYPE_CHECKING:
    from mcp.types import Tool
    from pydantic_ai.messages import ModelMessage

    from evals.agent.models import AgentModelSpec

#: Repository root, so the workspace and the server subprocess do not depend on
#: the caller's current directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SKILLS_DIR = ".claude/skills"

#: Guardrails against a model looping on tool calls and burning budget.
DEFAULT_REQUEST_LIMIT = 30
DEFAULT_TOOL_CALLS_LIMIT = 40
DEFAULT_TURN_TIMEOUT_S = 300.0


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #

#: Kept distinct from pass/fail (plan §3.4). An attempt that merely fails its
#: checks is still ``evaluated``; only infrastructure and model breakdowns are not.
OUTCOME_EVALUATED = "evaluated"
OUTCOME_HARNESS_ERROR = "harness_error"
OUTCOME_MODEL_FAILURE = "model_failure"
OUTCOME_ROUTING_FAILURE = "routing_failure"


@dataclass
class ConversationResult:
    """The outcome of one scripted conversation, successful or not."""

    attempt_id: str
    run_id: str
    spec_id: str
    messages: list[ModelMessage] = field(default_factory=list)
    tool_schemas: list[Tool] = field(default_factory=list)
    outcome: str = OUTCOME_EVALUATED
    error: str | None = None
    error_stage: str | None = None
    turns_completed: int = 0
    elapsed_s: float = 0.0
    final_output: str = ""
    task: str = ""

    def transcript(self) -> AgentTranscript:
        """Adapt the captured history to the shared contract."""
        return from_pydantic_ai(self.messages, task=self.task, final_output=self.final_output)

    def artifact(self, metadata: dict[str, Any] | None = None) -> AttemptArtifact:
        """Build the attempt artifact, including for a partial failed run."""
        meta: dict[str, Any] = {
            "spec_id": self.spec_id,
            "outcome": self.outcome,
            "error": self.error,
            "error_stage": self.error_stage,
            "turns_completed": self.turns_completed,
            "elapsed_s": round(self.elapsed_s, 2),
        }
        meta.update(metadata or {})
        return AttemptArtifact.from_messages(
            self.messages,
            attempt_id=self.attempt_id,
            run_id=self.run_id,
            task=self.task,
            final_output=self.final_output,
            tool_schemas=self.tool_schemas,
            metadata=meta,
        )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

#: A checkpoint inspects the transcript so far and returns an error string if the
#: expected state was not reached, or ``None`` to allow the script to continue.
Checkpoint = Callable[[AgentTranscript], str | None]


@dataclass
class ScriptedTurn:
    """One scripted user message, optionally gated on a checkpoint.

    The checkpoint runs *before* the reply is sent. If it fails, the script stops
    and the partial transcript is saved: sending a generic affirmative into an
    unexpected state could approve an action the scenario never intended
    (plan §2.1).
    """

    prompt: str
    checkpoint: Checkpoint | None = None
    name: str = ""


def build_agent(
    spec: AgentModelSpec,
    *,
    skills_dir: str = DEFAULT_SKILLS_DIR,
    mock_cluster: bool = True,
) -> tuple[Any, Any]:
    """Construct the agent and return it with its MCP toolset.

    The toolset is returned separately so tool schemas can be read from the very
    same server session the conversation runs against.
    """
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai.mcp import MCPToolset, StdioTransport
    from pydantic_ai_harness import Skills

    args = ["run", "rhoai-mcp", "--transport", "stdio"]
    if mock_cluster:
        args.append("--mock-cluster")

    toolset = MCPToolset(StdioTransport(command="uv", args=args, cwd=str(REPO_ROOT)))
    agent = Agent(
        model=spec.build(),
        capabilities=[
            LocalWorkspace(str(REPO_ROOT)),
            # Both skills stay in the catalog: the point is to test discovery
            # from a natural-language request, not to hand over the right one.
            Skills(str(REPO_ROOT / skills_dir)),
        ],
        toolsets=[toolset],
    )
    return agent, toolset


async def run_conversation(
    spec: AgentModelSpec,
    script: Sequence[ScriptedTurn | str],
    *,
    run_id: str,
    attempt_id: str | None = None,
    skills_dir: str = DEFAULT_SKILLS_DIR,
    mock_cluster: bool = True,
    request_limit: int = DEFAULT_REQUEST_LIMIT,
    tool_calls_limit: int = DEFAULT_TOOL_CALLS_LIMIT,
    turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
    on_turn: Callable[[int, AgentTranscript], Awaitable[None]] | None = None,
) -> ConversationResult:
    """Run a scripted conversation and capture everything needed to score it.

    Args:
        spec: Which model to drive.
        script: User messages, optionally wrapped in :class:`ScriptedTurn` to
            attach a checkpoint that must hold before the reply is sent.
        run_id: Identifier for the sweep this attempt belongs to.
        attempt_id: Identifier for this attempt; generated when omitted.
        skills_dir: Skill library, relative to the repository root.
        mock_cluster: Run the server against the fixed mock cluster state.
        request_limit: Maximum model requests per turn.
        tool_calls_limit: Maximum tool calls per turn.
        turn_timeout_s: Per-turn wall-clock budget.
        on_turn: Optional hook invoked after each completed turn.

    Returns:
        A :class:`ConversationResult`. Never raises for model or checkpoint
        problems: those are recorded as outcomes so the attempt is still saved.
    """
    import asyncio

    from pydantic_ai import capture_run_messages
    from pydantic_ai.exceptions import UsageLimitExceeded
    from pydantic_ai.usage import UsageLimits

    turns = [ScriptedTurn(prompt=t) if isinstance(t, str) else t for t in script]
    result = ConversationResult(
        attempt_id=attempt_id or uuid.uuid4().hex[:12],
        run_id=run_id,
        spec_id=spec.id,
        task=turns[0].prompt if turns else "",
    )
    limits = UsageLimits(request_limit=request_limit, tool_calls_limit=tool_calls_limit)
    started = time.monotonic()

    try:
        agent, toolset = build_agent(spec, skills_dir=skills_dir, mock_cluster=mock_cluster)
    except Exception as exc:  # noqa: BLE001 - setup failure is a harness error
        result.outcome = OUTCOME_HARNESS_ERROR
        result.error_stage = "agent_build"
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = time.monotonic() - started
        return result

    history: list[ModelMessage] = []
    try:
        # One context for the whole conversation: the stdio server stays up and
        # keeps its mock-cluster state across every turn.
        async with agent:
            try:
                result.tool_schemas = list(await toolset.list_tools())
            except Exception as exc:  # noqa: BLE001
                result.outcome = OUTCOME_HARNESS_ERROR
                result.error_stage = "tool_schema_fetch"
                result.error = f"{type(exc).__name__}: {exc}"
                return result

            for index, turn in enumerate(turns):
                # Checkpoints gate the *reply*, so the first prompt is ungated.
                if turn.checkpoint is not None and index > 0:
                    failure = turn.checkpoint(
                        from_pydantic_ai(
                            history, task=result.task, final_output=result.final_output
                        )
                    )
                    if failure is not None:
                        result.outcome = OUTCOME_EVALUATED
                        result.error_stage = f"checkpoint:{turn.name or index}"
                        result.error = failure
                        break

                # capture_run_messages keeps the partial history of a run that
                # raises, so a timeout still yields a saveable transcript.
                with capture_run_messages() as captured:
                    try:
                        async with asyncio.timeout(turn_timeout_s):
                            run = await agent.run(
                                turn.prompt,
                                message_history=history or None,
                                usage_limits=limits,
                            )
                    except TimeoutError:
                        history = _extend(history, captured)
                        result.outcome = OUTCOME_MODEL_FAILURE
                        result.error_stage = f"turn_timeout:{index}"
                        result.error = f"turn {index} exceeded {turn_timeout_s}s"
                        break
                    except UsageLimitExceeded as exc:
                        history = _extend(history, captured)
                        result.outcome = OUTCOME_MODEL_FAILURE
                        result.error_stage = f"usage_limit:{index}"
                        result.error = str(exc)
                        break
                    except Exception as exc:  # noqa: BLE001
                        history = _extend(history, captured)
                        result.outcome = OUTCOME_MODEL_FAILURE
                        result.error_stage = f"model_error:{index}"
                        result.error = f"{type(exc).__name__}: {exc}"
                        break

                # Cumulative history; never concatenate successive snapshots.
                history = list(run.all_messages())
                result.final_output = str(run.output or "")
                result.turns_completed = index + 1

                if on_turn is not None:
                    await on_turn(index, from_pydantic_ai(history, task=result.task))

    except Exception as exc:  # noqa: BLE001 - connection/teardown failure
        result.outcome = OUTCOME_HARNESS_ERROR
        result.error_stage = result.error_stage or "mcp_session"
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.messages = history
        result.elapsed_s = time.monotonic() - started

    return result


def _extend(history: list[ModelMessage], captured: Sequence[ModelMessage]) -> list[ModelMessage]:
    """Merge a failed run's captured messages into the cumulative history.

    ``capture_run_messages`` yields the messages of the current run only, and it
    includes the history that was passed in, so prefer it wholesale when it is
    the longer of the two.
    """
    captured_list = list(captured)
    return captured_list if len(captured_list) >= len(history) else history
