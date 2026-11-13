"""Shared transcript contract, Pydantic AI adapter, and attempt artifacts.

The persisted artifact is the scoring input, in live runs as well as re-scores
(plan §3.6): every attempt goes agent -> write artifact -> load artifact -> adapt
-> score. That is one scoring path rather than two, so a lossy artifact fails the
live run instead of being discovered months later.

Two representations exist here:

``AgentTranscript``
    Our own framework-neutral view, consumed **mid-conversation** by scripted
    checkpoints as well as afterwards by the DeepEval adapter and the
    deterministic checks. See the class docstring for why a DeepEval
    ``ConversationalTestCase`` cannot serve this role.
``AttemptArtifact``
    What is written to disk: the *native* Pydantic AI message history (which
    round-trips through JSON, unlike DeepEval objects) plus the MCP tool schemas
    and run metadata needed to rebuild everything else.

Two correctness rules this module exists to enforce:

1. **Calls and returns are matched by call ID and occurrence.** The
   previous LCS helper consumed ``result.tool_calls`` positionally, which
   attached a later tool's result to an earlier turn whenever a harness call was
   interleaved. Parallel tool calls can also return out of order.
2. **Harness tools are distinguished from MCP tools by type, not by name.**
   ``LoadCapabilityCallPart`` is a *subclass* of ``ToolCallPart`` and reports the
   same ``part_kind='tool-call'``, so an ``isinstance`` check against
   ``ToolCallPart`` alone silently counts skill loading as an MCP interaction.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mcp.types import Tool
    from pydantic_ai.messages import ModelMessage

ToolKind = Literal["mcp", "harness"]
Role = Literal["user", "assistant"]

#: Claude spells MCP tools ``mcp__<server>__<tool>``; the server and Pydantic AI
#: both use the bare name. Skill ``allowed-tools`` use the prefixed spelling.
MCP_TOOL_PREFIX = "mcp__"

#: Harness-provided tools, reported separately from MCP tools (plan §3.2).
HARNESS_TOOL_NAMES = frozenset({"load_capability"})

#: ``load_capability`` names the skill under this key, as a JSON-encoded string.
LOAD_CAPABILITY_ARG_KEYS = ("id", "name", "capability")

#: Instruction id under which the harness publishes the skill catalog (name +
#: description only). In discovery mode this is all the model sees before it
#: routes, which makes the descriptions themselves a tested artifact (plan §6.4).
CATALOG_INSTRUCTION_ID = "capability:deferred_capabilities"

#: The catalog is itself carried by a pseudo-capability. It must never be filed
#: as a delivered skill body: on a routing failure that would make
#: :meth:`AgentTranscript.skill_body` return the two-line catalog, and a fidelity
#: rubric would then score adherence to a document that is not the skill.
CATALOG_CAPABILITY_ID = "deferred_capabilities"


def normalize_tool_name(name: str) -> str:
    """Strip Claude's ``mcp__<server>__`` prefix to get the server tool name.

    >>> normalize_tool_name("mcp__rhoai-mcp__recommend_model")
    'recommend_model'
    >>> normalize_tool_name("recommend_model")
    'recommend_model'
    """
    if name.startswith(MCP_TOOL_PREFIX):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return name


# --------------------------------------------------------------------------- #
# Shared contract
# --------------------------------------------------------------------------- #


@dataclass
class ToolCallRecord:
    """One tool call paired with its own result, matched by call ID."""

    call_id: str
    name: str
    """Normalised server tool name."""
    raw_name: str
    """Name exactly as the model emitted it."""
    arguments: dict[str, Any]
    kind: ToolKind
    turn_index: int
    """Index into :attr:`AgentTranscript.turns` of the assistant turn that issued it."""
    result_text: str | None = None
    structured_result: Any | None = None
    outcome: str | None = None
    """Pydantic AI outcome: success | failed | denied | interrupted | retried."""
    error: str | None = None

    @property
    def is_error(self) -> bool:
        """Whether this call failed, was denied, or never returned."""
        return self.error is not None or (self.outcome is not None and self.outcome != "success")

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "raw_name": self.raw_name,
            "arguments": self.arguments,
            "kind": self.kind,
            "turn_index": self.turn_index,
            "result_text": self.result_text,
            "structured_result": self.structured_result,
            "outcome": self.outcome,
            "error": self.error,
            "is_error": self.is_error,
        }


@dataclass
class TranscriptTurn:
    """One conversational turn, with the tool calls the assistant made in it."""

    role: Role
    content: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    thinking: str = ""

    @property
    def mcp_tool_calls(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.kind == "mcp"]

    @property
    def harness_tool_calls(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.kind == "harness"]


@dataclass
class AgentTranscript:
    """Framework-neutral view of one conversation. Our construct, not a library type.

    **Primary consumer: mid-conversation checkpoints.** Each scripted reply in
    :mod:`evals.agent.runner` is gated on a ``Callable[[AgentTranscript], str | None]``
    evaluated *before* the reply is sent, so the script can refuse to send a
    generic "yes" into an unexpected state (plan §2.1). At that moment the
    conversation is still running and no DeepEval case exists, so checkpoints
    need a cheap read-only view of "what has happened so far".

    **Why a DeepEval ``ConversationalTestCase`` cannot play this role:**

    - It rejects an empty turn list, so it cannot represent a conversation that
      has not started, or one a harness error ended before any message.
    - ``MCPToolCall`` carries only ``name``/``args``/``result`` — no call ID and
      no turn index, so call attribution cannot be re-derived or verified.
    - Harness calls are deliberately excluded from ``mcp_tools_called``, so the
      §3.1 routing checks (which ``load_capability`` calls succeeded, in what
      order) have nothing to read.
    - Instructions are not part of its model at all, so the delivered skill body
      and the routing catalog are unrepresentable.

    The other reason this type exists is deduplication of the *risky* logic:
    without it, every deterministic check would re-implement call/return matching by ID
    and harness-vs-MCP discrimination — precisely the code that produced the
    known adapter bugs. Centralising it means a fix lands everywhere at once.

    Note the "shared with the LCS path" rationale in plan §2.2 is currently
    aspirational: ``evals/lcs_client.py`` still has its own shape and nothing
    converts it. The two reasons above stand on their own.
    """

    task: str
    final_output: str
    turns: list[TranscriptTurn] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    """Every tool call in emission order, harness and MCP alike."""
    skill_catalog: str = ""
    """The name+description catalog the model routed from, verbatim."""
    delivered_instructions: dict[str, str] = field(default_factory=dict)
    """Capability id -> the instruction body the harness actually delivered.

    This, not the ``load_capability`` tool result, is where the skill body lives
    (see :func:`from_pydantic_ai`).
    """

    @property
    def mcp_tool_calls(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.kind == "mcp"]

    @property
    def harness_tool_calls(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.kind == "harness"]

    @property
    def assistant_text(self) -> str:
        """Every assistant message joined, in order.

        Checks that look for emitted artefacts — the recommendation table, the
        handoff block — must search this rather than :attr:`final_output`. A
        phase-gated skill emits the table in its Phase 3 message and then keeps
        talking, so by the end of the conversation the table is several turns
        behind the last message.
        """
        return "\n".join(t.content for t in self.turns if t.role == "assistant" and t.content)

    @property
    def tool_names_used(self) -> list[str]:
        """Normalised MCP tool names, in first-call order, deduplicated."""
        seen: dict[str, None] = {}
        for call in self.mcp_tool_calls:
            seen.setdefault(call.name, None)
        return list(seen)

    @property
    def skills_loaded(self) -> list[str]:
        """Skill names successfully loaded via ``load_capability``, in order.

        Failed loads are excluded: routing is judged on skills the model actually
        received (plan §3.1).
        """
        names: list[str] = []
        for call in self.harness_tool_calls:
            if call.name != "load_capability" or call.is_error:
                continue
            for key in LOAD_CAPABILITY_ARG_KEYS:
                value = call.arguments.get(key)
                if isinstance(value, str):
                    names.append(value)
                    break
        return names

    def skill_body(self, skill: str | None = None) -> str | None:
        """The skill body as actually delivered to the model.

        Note this reads the delivered *instructions*, not the ``load_capability``
        tool result. Plan §3.6 assumed the body arrives as the tool result; in
        pydantic-ai-harness 0.64 that result is only an acknowledgement
        (``{"message": "Capability 'navigator' loaded..."}``) and the body is
        pushed as an instruction delta. A fidelity rubric shown the tool result
        would therefore be shown nothing.
        """
        if skill is not None:
            return self.delivered_instructions.get(skill)
        loaded = self.skills_loaded
        if loaded:
            return self.delivered_instructions.get(loaded[0])
        return next(iter(self.delivered_instructions.values()), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "final_output": self.final_output,
            "tool_names_used": self.tool_names_used,
            "skills_loaded": self.skills_loaded,
            "skill_catalog": self.skill_catalog,
            "delivered_instructions": dict(self.delivered_instructions),
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "thinking": t.thinking,
                    "tool_calls": [c.to_dict() for c in t.tool_calls],
                }
                for t in self.turns
            ],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
        }


# --------------------------------------------------------------------------- #
# Pydantic AI adapter
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a tool result to something JSON-serialisable."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _jsonable(dump(mode="json"))
        except Exception:  # noqa: BLE001 - fall through to repr
            pass
    return repr(value)


def _result_text(content: Any) -> str:
    """Render a tool result as text for the judge and for debugging."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(_jsonable(content), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


def _args_dict(part: Any) -> dict[str, Any]:
    """Extract tool-call arguments as a dict, tolerating string-encoded args."""
    args = getattr(part, "args", None)
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args or "{}")
        except (json.JSONDecodeError, TypeError):
            return {"__unparsed_args__": args}
        return parsed if isinstance(parsed, dict) else {"__args__": parsed}
    return {}


@dataclass
class _Return:
    """A tool return or retry associated with one call occurrence."""

    content: Any = None
    outcome: str | None = None
    error: str | None = None


def _index_returns(messages: list[ModelMessage]) -> dict[tuple[int, int], _Return]:
    """Match each return to an earlier call occurrence, including reused IDs.

    Calls sharing an ID in one response are matched in order. As in Pydantic AI,
    reusing an unresolved ID in a later response leaves earlier calls unanswered.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        ToolCallPart,
        ToolReturnPart,
    )

    returns: dict[tuple[int, int], _Return] = {}
    pending: dict[str, deque[tuple[int, int]]] = {}
    for message_index, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            for part_index, part in enumerate(message.parts):
                if not isinstance(part, ToolCallPart) or not part.tool_call_id:
                    continue
                calls = pending.setdefault(part.tool_call_id, deque())
                if calls and calls[-1][0] != message_index:
                    calls.clear()
                calls.append((message_index, part_index))
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                # LoadCapabilityReturnPart is also a ToolReturnPart.
                if isinstance(part, ToolReturnPart):
                    ret = _Return(content=part.content, outcome=part.outcome)
                elif isinstance(part, RetryPromptPart) and part.tool_name is not None:
                    ret = _Return(error=_result_text(part.content), outcome="retried")
                else:
                    continue
                calls = pending.get(part.tool_call_id or "")
                if calls:
                    returns[calls.popleft()] = ret
    return returns


def _capability_id(instruction_id: Any) -> str | None:
    """Pull the capability name out of an ``InstructionId``.

    The harness stamps skill instructions with
    ``InstructionId(source=CapabilityInstructionSource(id='navigator'))``.
    """
    source = getattr(instruction_id, "source", None)
    capability = getattr(source, "id", None)
    return capability if isinstance(capability, str) else None


def _is_catalog(identifier: Any, capability: str | None) -> bool:
    """Whether this instruction is the deferred-capability catalog, not a skill."""
    return capability == CATALOG_CAPABILITY_ID or str(identifier) == CATALOG_INSTRUCTION_ID


def _record_instruction(
    identifier: Any, content: str, catalog: list[str], delivered: dict[str, list[str]]
) -> None:
    """File one instruction as either the catalog or a delivered skill body."""
    if not content:
        return
    capability = _capability_id(identifier)
    if _is_catalog(identifier, capability):
        if content not in catalog:
            catalog.append(content)
        return
    if not capability:
        return
    # Accumulate: a capability may contribute several instruction parts, and
    # overwriting would silently hand a fidelity rubric a truncated skill.
    # Deduplicated because instructions are re-stated across messages.
    parts = delivered.setdefault(capability, [])
    if content not in parts:
        parts.append(content)


def _collect_instructions(
    message: Any, catalog: list[str], delivered: dict[str, list[str]]
) -> None:
    """Record the skill catalog and any delivered skill bodies from a message."""
    from pydantic_ai.messages import InstructionBaselinePart, InstructionDeltaPart

    for part in message.parts:
        if isinstance(part, InstructionBaselinePart):
            for instruction in part.instructions or ():
                _record_instruction(
                    getattr(instruction, "id", None),
                    str(getattr(instruction, "content", "") or ""),
                    catalog,
                    delivered,
                )
        elif isinstance(part, InstructionDeltaPart):
            _record_instruction(part.id, str(part.content or ""), catalog, delivered)


def _classify(part: Any) -> ToolKind:
    """Decide whether a call part is a harness tool or an MCP tool.

    Checked by type first: ``LoadCapabilityCallPart`` subclasses ``ToolCallPart``
    and reports ``part_kind='tool-call'``, so a name check alone would be the
    only safeguard against counting skill loading as an MCP interaction.
    """
    from pydantic_ai.messages import LoadCapabilityCallPart

    if isinstance(part, LoadCapabilityCallPart):
        return "harness"
    if normalize_tool_name(getattr(part, "tool_name", "")) in HARNESS_TOOL_NAMES:
        return "harness"
    return "mcp"


def from_pydantic_ai(
    messages: list[ModelMessage],
    *,
    task: str = "",
    final_output: str = "",
) -> AgentTranscript:
    """Adapt a native Pydantic AI message history to the shared contract.

    Args:
        messages: Cumulative history, e.g. the final ``result.all_messages()``.
            Do not concatenate successive ``all_messages()`` snapshots; that
            duplicates earlier turns (plan §2.1).
        task: The initial user request. Defaults to the first user prompt found.
        final_output: The agent's last textual output. Defaults to the last
            assistant text in the history.

    Returns:
        An :class:`AgentTranscript` preserving message order, with every tool
        call matched to its own result by call ID.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        UserPromptPart,
    )

    returns = _index_returns(messages)
    transcript = AgentTranscript(task=task, final_output=final_output)
    catalog: list[str] = []
    delivered: dict[str, list[str]] = {}

    for message_index, message in enumerate(messages):
        if isinstance(message, ModelRequest):
            _collect_instructions(message, catalog, delivered)
            # A ModelRequest carrying only tool returns is not a user turn; its
            # content is already attributed to the call that produced it.
            prompts = [p for p in message.parts if isinstance(p, UserPromptPart)]
            for prompt in prompts:
                content = prompt.content
                text = content if isinstance(content, str) else _result_text(content)
                transcript.turns.append(TranscriptTurn(role="user", content=text))

        elif isinstance(message, ModelResponse):
            texts = [p.content for p in message.parts if isinstance(p, TextPart)]
            thinking = [
                p.content for p in message.parts if isinstance(p, ThinkingPart) and p.content
            ]
            turn = TranscriptTurn(
                role="assistant",
                content="\n".join(t for t in texts if t),
                thinking="\n".join(thinking),
            )
            turn_index = len(transcript.turns)

            for part_index, part in enumerate(message.parts):
                if not isinstance(part, ToolCallPart):
                    continue
                call_id = part.tool_call_id or ""
                ret = returns.get((message_index, part_index), _Return())
                record = ToolCallRecord(
                    call_id=call_id,
                    name=normalize_tool_name(part.tool_name),
                    raw_name=part.tool_name,
                    arguments=_args_dict(part),
                    kind=_classify(part),
                    turn_index=turn_index,
                    result_text=None if ret.content is None else _result_text(ret.content),
                    structured_result=_jsonable(ret.content),
                    outcome=ret.outcome,
                    error=ret.error,
                )
                turn.tool_calls.append(record)
                transcript.tool_calls.append(record)

            transcript.turns.append(turn)

    transcript.skill_catalog = "\n\n".join(catalog)
    transcript.delivered_instructions = {k: "\n\n".join(v) for k, v in delivered.items()}
    if not transcript.task:
        transcript.task = next((t.content for t in transcript.turns if t.role == "user"), "")
    if not transcript.final_output:
        transcript.final_output = next(
            (t.content for t in reversed(transcript.turns) if t.role == "assistant" and t.content),
            "",
        )
    return transcript


# --------------------------------------------------------------------------- #
# Attempt artifacts
# --------------------------------------------------------------------------- #

ARTIFACT_SCHEMA_VERSION = 1


@dataclass
class AttemptArtifact:
    """Everything needed to re-score one attempt without re-running the agent.

    A stale-schema artifact fails to load and is re-run; there is deliberately no
    migration path (plan §3.6).
    """

    attempt_id: str
    run_id: str
    messages_json: bytes
    """Native Pydantic AI history, serialised with ``ModelMessagesTypeAdapter``."""
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # No separate skill-catalog field: the catalog the model routed from is an
    # InstructionBaselinePart inside messages_json, and AgentTranscript derives
    # it on load. Storing it twice would let the copies disagree.
    task: str = ""
    final_output: str = ""
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_messages(
        cls,
        messages: list[ModelMessage],
        *,
        attempt_id: str,
        run_id: str,
        task: str = "",
        final_output: str = "",
        tool_schemas: list[Tool] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AttemptArtifact:
        """Capture an attempt. Safe to call on a partial/failed conversation."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        return cls(
            attempt_id=attempt_id,
            run_id=run_id,
            messages_json=ModelMessagesTypeAdapter.dump_json(messages),
            tool_schemas=[t.model_dump(mode="json") for t in (tool_schemas or [])],
            metadata=dict(metadata or {}),
            task=task,
            final_output=final_output,
        )

    # -- persistence --------------------------------------------------------- #

    def save(self, root: Path) -> Path:
        """Write this artifact under ``<root>/<run_id>/<attempt_id>/``."""
        directory = root / self.run_id / self.attempt_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "messages.json").write_bytes(self.messages_json)
        (directory / "artifact.json").write_text(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "attempt_id": self.attempt_id,
                    "run_id": self.run_id,
                    "created_at": self.created_at,
                    "task": self.task,
                    "final_output": self.final_output,
                    "tool_schemas": self.tool_schemas,
                    "metadata": self.metadata,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> AttemptArtifact:
        """Load an artifact written by :meth:`save`.

        Raises:
            ValueError: If the artifact was written by a different schema version.
        """
        meta = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
        version = meta.get("schema_version")
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact {directory} has schema_version {version!r}, "
                f"expected {ARTIFACT_SCHEMA_VERSION}; re-run the attempt"
            )
        return cls(
            attempt_id=meta["attempt_id"],
            run_id=meta["run_id"],
            messages_json=(directory / "messages.json").read_bytes(),
            tool_schemas=meta.get("tool_schemas", []),
            metadata=meta.get("metadata", {}),
            task=meta.get("task", ""),
            final_output=meta.get("final_output", ""),
            schema_version=version,
            created_at=meta.get("created_at", ""),
        )

    # -- derived views ------------------------------------------------------- #

    def messages(self) -> list[ModelMessage]:
        """Rehydrate the native Pydantic AI history."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        return list(ModelMessagesTypeAdapter.validate_json(self.messages_json))

    def transcript(self) -> AgentTranscript:
        """Rebuild the shared transcript. This is the scoring input (plan §3.6)."""
        return from_pydantic_ai(self.messages(), task=self.task, final_output=self.final_output)

    def mcp_tools(self) -> list[Tool]:
        """Rehydrate the MCP tool schemas captured from the same server session."""
        from mcp.types import Tool

        return [Tool.model_validate(t) for t in self.tool_schemas]
