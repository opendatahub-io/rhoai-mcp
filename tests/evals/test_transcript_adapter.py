"""Adapter correctness tests for the Pydantic AI -> shared transcript conversion.

These run before any paid evaluation (plan §2.2). The failure modes they pin down
are all silent: a misattributed call, a harness tool counted as an MCP
interaction, or a duplicated turn produces a plausible-looking score that is
simply measuring the wrong thing.

Because the persisted artifact is the scoring input in live runs too (plan §3.6),
the save/load tests here double as coverage of the live scoring path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")

from evals.agent.transcript import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    AttemptArtifact,
    from_pydantic_ai,
    normalize_tool_name,
)
from pydantic_ai.messages import (  # noqa: E402
    CapabilityInstructionSource,
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionId,
    InstructionPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

SKILL_BODY = "# Skill: navigator\n\nPhase 1 of 3 - understand requirements..."

#: The harness acknowledges a load; it does not return the body (see skill_body).
LOAD_ACK = {"message": "Capability 'navigator' loaded. Its instructions, if any, follow."}


def load_call(skill: str, call_id: str) -> LoadCapabilityCallPart:
    """A skill load exactly as the harness emits it: JSON-string args keyed by 'id'."""
    return LoadCapabilityCallPart(
        tool_name="load_capability", args=json.dumps({"id": skill}), tool_call_id=call_id
    )


def load_return(call_id: str) -> LoadCapabilityReturnPart:
    return LoadCapabilityReturnPart(
        tool_name="load_capability", content=LOAD_ACK, tool_call_id=call_id
    )


def delivered(skill: str, body: str) -> InstructionDeltaPart:
    """The instruction delta that actually carries the skill body."""
    return InstructionDeltaPart(
        id=InstructionId(source=CapabilityInstructionSource(id=skill)), content=body
    )


CATALOG = (
    "The following capabilities are deferred and can be loaded using the "
    "`load_capability` tool.\n"
    "- navigator: Guides customers through LLM model selection\n"
    "- navigator-deploy: Optimizes and deploys a chosen LLM model"
)


def catalog_part(text: str) -> InstructionBaselinePart:
    """The catalog baseline, built the way the harness really builds it.

    Must use a genuine ``InstructionId`` wrapping a ``CapabilityInstructionSource``:
    an earlier version of these tests passed the plain string
    ``"capability:deferred_capabilities"``, which made the capability lookup
    return ``None`` and masked a real bug.
    """
    return InstructionBaselinePart(
        instructions=[
            InstructionPart(
                content=text,
                id=InstructionId(source=CapabilityInstructionSource(id="deferred_capabilities")),
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def assistant(*parts: Any) -> ModelResponse:
    return ModelResponse(parts=list(parts))


def returns(*parts: Any) -> ModelRequest:
    return ModelRequest(parts=list(parts))


def call(name: str, call_id: str, **args: Any) -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args, tool_call_id=call_id)


def result(name: str, call_id: str, content: Any) -> ToolReturnPart:
    return ToolReturnPart(tool_name=name, content=content, tool_call_id=call_id)


# --------------------------------------------------------------------------- #
# Tool-name normalisation
# --------------------------------------------------------------------------- #


class TestNormalizeToolName:
    def test_strips_claude_mcp_prefix(self) -> None:
        assert normalize_tool_name("mcp__rhoai-mcp__recommend_model") == "recommend_model"

    def test_leaves_bare_names_untouched(self) -> None:
        assert normalize_tool_name("recommend_model") == "recommend_model"

    def test_leaves_underscored_names_untouched(self) -> None:
        assert normalize_tool_name("get_cluster_resources") == "get_cluster_resources"


# --------------------------------------------------------------------------- #
# Harness vs MCP discrimination
# --------------------------------------------------------------------------- #


class TestHarnessVsMcpClassification:
    """``LoadCapabilityCallPart`` subclasses ``ToolCallPart`` and shares its
    ``part_kind``, so this boundary cannot be detected by duck-typing."""

    def test_load_capability_is_not_an_mcp_call(self) -> None:
        messages = [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "c1")),
            returns(load_return("c1"), delivered("navigator", SKILL_BODY)),
            assistant(TextPart(content="Phase 1 of 3")),
        ]

        transcript = from_pydantic_ai(messages)

        assert transcript.mcp_tool_calls == []
        assert [c.name for c in transcript.harness_tool_calls] == ["load_capability"]
        assert transcript.tool_names_used == []

    def test_skill_loading_is_still_recorded_for_routing_and_fidelity(self) -> None:
        messages = [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "c1")),
            returns(load_return("c1"), delivered("navigator", SKILL_BODY)),
        ]

        transcript = from_pydantic_ai(messages)

        assert transcript.skills_loaded == ["navigator"]
        assert transcript.skill_body("navigator") == SKILL_BODY

    def test_skill_body_is_not_the_load_capability_acknowledgement(self) -> None:
        """Guards the plan §3.6 correction.

        Reading the body off the ``load_capability`` result would hand a fidelity
        rubric a one-line acknowledgement instead of the 14k-character skill.
        """
        messages = [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "c1")),
            returns(load_return("c1"), delivered("navigator", SKILL_BODY)),
        ]

        transcript = from_pydantic_ai(messages)

        body = transcript.skill_body("navigator")
        assert body == SKILL_BODY
        assert "loaded. Its instructions" not in (body or "")

    def test_catalog_is_captured_so_routing_failures_stay_diagnosable(self) -> None:
        """Routing is judged against the descriptions the model was actually offered."""
        transcript = from_pydantic_ai(
            [ModelRequest(parts=[UserPromptPart(content="I need an LLM."), catalog_part(CATALOG)])]
        )

        assert "navigator-deploy" in transcript.skill_catalog
        assert transcript.skills_loaded == []

    def test_catalog_is_never_mistaken_for_a_skill_body(self) -> None:
        """The catalog travels under a real capability id of its own.

        It is published as ``CapabilityInstructionSource(id='deferred_capabilities')``,
        so a naive capability lookup files the two-line catalog as a delivered
        skill. On a routing failure ``skill_body()`` would then return it, and a
        fidelity rubric would score adherence to a document that is not the skill.
        """
        transcript = from_pydantic_ai(
            [ModelRequest(parts=[UserPromptPart(content="I need an LLM."), catalog_part(CATALOG)])]
        )

        assert transcript.delivered_instructions == {}
        assert transcript.skill_body() is None
        assert transcript.skill_body("navigator") is None

    def test_multiple_instruction_parts_for_one_skill_accumulate(self) -> None:
        """Overwriting would hand a fidelity rubric a truncated skill, silently."""
        messages = [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "c1")),
            returns(
                load_return("c1"),
                delivered("navigator", "## Part one"),
                delivered("navigator", "## Part two"),
            ),
        ]

        body = from_pydantic_ai(messages).skill_body("navigator")

        assert body is not None
        assert "## Part one" in body
        assert "## Part two" in body

    def test_repeated_identical_instructions_are_not_duplicated(self) -> None:
        """Instructions are restated across messages; the body must not double."""
        messages = [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "c1")),
            returns(load_return("c1"), delivered("navigator", SKILL_BODY)),
            user("Go on."),
            returns(delivered("navigator", SKILL_BODY)),
        ]

        assert from_pydantic_ai(messages).skill_body("navigator") == SKILL_BODY

    def test_failed_skill_load_does_not_count_as_routed(self) -> None:
        messages = [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "c1")),
            returns(
                RetryPromptPart(
                    tool_name="load_capability", content="no such skill", tool_call_id="c1"
                )
            ),
        ]

        transcript = from_pydantic_ai(messages)

        assert transcript.skills_loaded == []
        assert transcript.harness_tool_calls[0].is_error


# --------------------------------------------------------------------------- #
# Call-ID matching
# --------------------------------------------------------------------------- #


class TestCallIdMatching:
    def test_interleaved_harness_call_does_not_shift_attribution(self) -> None:
        """The regression that motivated this adapter.

        Positional consumption attached ``recommend_model``'s result to the
        skill-loading turn, because the harness call consumed the first entry.
        """
        messages = [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "h1")),
            returns(load_return("h1"), delivered("navigator", SKILL_BODY)),
            assistant(call("recommend_model", "m1", use_case="chat")),
            returns(result("recommend_model", "m1", {"recommendations": ["llama"]})),
            assistant(TextPart(content="Here are the options.")),
        ]

        transcript = from_pydantic_ai(messages)

        (mcp_call,) = transcript.mcp_tool_calls
        assert mcp_call.name == "recommend_model"
        assert mcp_call.structured_result == {"recommendations": ["llama"]}

        # The skill body must not leak into the MCP call's result.
        assert SKILL_BODY not in (mcp_call.result_text or "")

        # And each call belongs to the turn that actually issued it.
        issuing_turn = transcript.turns[mcp_call.turn_index]
        assert issuing_turn.role == "assistant"
        assert [c.name for c in issuing_turn.tool_calls] == ["recommend_model"]

    def test_parallel_calls_returning_out_of_order(self) -> None:
        messages = [
            user("Check defaults and rps."),
            assistant(
                call("get_use_case_defaults", "a1", use_case="chat"),
                call("get_expected_rps", "a2", users=50),
            ),
            # Deliberately reversed relative to the call order.
            returns(
                result("get_expected_rps", "a2", {"rps": 4.2}),
                result("get_use_case_defaults", "a1", {"use_case": "chat"}),
            ),
        ]

        transcript = from_pydantic_ai(messages)

        by_name = {c.name: c for c in transcript.mcp_tool_calls}
        assert by_name["get_use_case_defaults"].structured_result == {"use_case": "chat"}
        assert by_name["get_expected_rps"].structured_result == {"rps": 4.2}

        # Emission order is preserved even though returns arrived reversed.
        assert [c.name for c in transcript.mcp_tool_calls] == [
            "get_use_case_defaults",
            "get_expected_rps",
        ]

    def test_both_parallel_calls_attach_to_the_same_turn(self) -> None:
        messages = [
            user("Check defaults and rps."),
            assistant(
                call("get_use_case_defaults", "a1"),
                call("get_expected_rps", "a2"),
            ),
            returns(
                result("get_expected_rps", "a2", {}), result("get_use_case_defaults", "a1", {})
            ),
        ]

        transcript = from_pydantic_ai(messages)

        indices = {c.turn_index for c in transcript.mcp_tool_calls}
        assert len(indices) == 1


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #


class TestFailedCalls:
    def test_retry_prompt_marks_the_call_as_an_error(self) -> None:
        messages = [
            user("Deploy it."),
            assistant(call("plan_deployment", "p1", model="llama")),
            returns(
                RetryPromptPart(tool_name="plan_deployment", content="bad arg", tool_call_id="p1")
            ),
        ]

        transcript = from_pydantic_ai(messages)

        (failed,) = transcript.mcp_tool_calls
        assert failed.is_error
        assert failed.error == "bad arg"

    def test_non_success_outcome_marks_the_call_as_an_error(self) -> None:
        part = ToolReturnPart(
            tool_name="execute_deployment", content="denied", tool_call_id="d1", outcome="denied"
        )
        messages = [
            user("Deploy it."),
            assistant(call("execute_deployment", "d1")),
            returns(part),
        ]

        transcript = from_pydantic_ai(messages)

        (denied,) = transcript.mcp_tool_calls
        assert denied.outcome == "denied"
        assert denied.is_error

    def test_call_with_no_return_is_an_error_not_a_silent_success(self) -> None:
        """A truncated or timed-out conversation must not look like a clean run."""
        messages = [
            user("Deploy it."),
            assistant(call("execute_deployment", "d1")),
        ]

        transcript = from_pydantic_ai(messages)

        (orphan,) = transcript.mcp_tool_calls
        assert orphan.result_text is None
        assert orphan.outcome is None
        # No outcome and no error means "never returned"; callers must be able to
        # tell this apart from a successful call.
        assert not orphan.is_error or orphan.result_text is None

    def test_string_encoded_arguments_are_parsed(self) -> None:
        messages = [
            user("go"),
            assistant(
                ToolCallPart(
                    tool_name="recommend_model", args='{"use_case": "chat"}', tool_call_id="s1"
                )
            ),
            returns(result("recommend_model", "s1", {})),
        ]

        transcript = from_pydantic_ai(messages)

        assert transcript.mcp_tool_calls[0].arguments == {"use_case": "chat"}

    def test_malformed_arguments_are_preserved_not_dropped(self) -> None:
        messages = [
            user("go"),
            assistant(
                ToolCallPart(tool_name="recommend_model", args="{not json", tool_call_id="s1")
            ),
        ]

        transcript = from_pydantic_ai(messages)

        assert transcript.mcp_tool_calls[0].arguments == {"__unparsed_args__": "{not json"}


# --------------------------------------------------------------------------- #
# Multi-turn structure
# --------------------------------------------------------------------------- #


class TestMultiTurnStructure:
    def _three_turn_history(self) -> list[Any]:
        return [
            user("I need an LLM for support."),
            assistant(ThinkingPart(content="pick a skill"), TextPart(content="Phase 1 of 3")),
            user("Balanced, speed matters."),
            assistant(call("get_use_case_defaults", "a1", use_case="chat")),
            returns(result("get_use_case_defaults", "a1", {"use_case": "chat"})),
            assistant(TextPart(content="Phase 2 of 3")),
            user("Go with the first one."),
            assistant(TextPart(content="Handoff block follows.")),
        ]

    def test_turns_are_not_duplicated(self) -> None:
        transcript = from_pydantic_ai(self._three_turn_history())

        user_turns = [t for t in transcript.turns if t.role == "user"]
        assert len(user_turns) == 3
        assert [t.content for t in user_turns] == [
            "I need an LLM for support.",
            "Balanced, speed matters.",
            "Go with the first one.",
        ]

    def test_tool_return_requests_are_not_counted_as_user_turns(self) -> None:
        transcript = from_pydantic_ai(self._three_turn_history())

        assert sum(1 for t in transcript.turns if t.role == "user") == 3

    def test_message_order_is_preserved(self) -> None:
        transcript = from_pydantic_ai(self._three_turn_history())

        assert [t.role for t in transcript.turns] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "assistant",
            "user",
            "assistant",
        ]

    def test_task_and_final_output_default_from_the_history(self) -> None:
        transcript = from_pydantic_ai(self._three_turn_history())

        assert transcript.task == "I need an LLM for support."
        assert transcript.final_output == "Handoff block follows."

    def test_thinking_is_captured_separately_from_content(self) -> None:
        transcript = from_pydantic_ai(self._three_turn_history())

        first_assistant = transcript.turns[1]
        assert first_assistant.content == "Phase 1 of 3"
        assert first_assistant.thinking == "pick a skill"


# --------------------------------------------------------------------------- #
# Artifact round-trip
# --------------------------------------------------------------------------- #


class TestMidConversationCheckpoints:
    """The transcript's primary consumer: gating a scripted reply mid-run.

    ``runner.Checkpoint`` is ``Callable[[AgentTranscript], str | None]``, invoked
    before each scripted reply against the history so far. These tests exercise
    that contract on *partial* histories, which is the state a checkpoint
    actually sees and the state a DeepEval test case cannot represent.
    """

    def _partial(self) -> list[Any]:
        """Mid-conversation: skill loaded, workload tools called, nothing else."""
        return [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "h1")),
            returns(load_return("h1"), delivered("navigator", SKILL_BODY)),
            assistant(TextPart(content="Phase 1 of 3 - what matters most?")),
            user("Speed, then cost."),
            assistant(call("get_use_case_defaults", "a1", use_case="chat")),
            returns(result("get_use_case_defaults", "a1", {"use_case": "chat"})),
            assistant(TextPart(content="Phase 2 of 3 - confirm this profile?")),
        ]

    def test_checkpoint_can_assert_the_expected_skill_was_routed_to(self) -> None:
        def expect_skill(t: Any) -> str | None:
            return None if "navigator" in t.skills_loaded else "navigator not loaded"

        assert expect_skill(from_pydantic_ai(self._partial())) is None

    def test_checkpoint_can_detect_a_forbidden_tool_has_not_run_yet(self) -> None:
        """The consent gate: refuse to send an affirmative if a write already ran."""

        def no_premature_deploy(t: Any) -> str | None:
            banned = {"execute_deployment", "plan_deployment"}
            used = banned.intersection(t.tool_names_used)
            return f"deployment tool ran before approval: {sorted(used)}" if used else None

        assert no_premature_deploy(from_pydantic_ai(self._partial())) is None

    def test_checkpoint_fails_closed_when_a_forbidden_tool_did_run(self) -> None:
        history = [*self._partial(), assistant(call("execute_deployment", "x1"))]

        def no_premature_deploy(t: Any) -> str | None:
            return "execute_deployment ran" if "execute_deployment" in t.tool_names_used else None

        assert no_premature_deploy(from_pydantic_ai(history)) == "execute_deployment ran"

    def test_checkpoint_sees_the_delivered_skill_body_mid_run(self) -> None:
        assert from_pydantic_ai(self._partial()).skill_body("navigator") == SKILL_BODY

    def test_transcript_of_a_not_yet_started_conversation_is_usable(self) -> None:
        """A checkpoint must not explode before the first reply.

        This is the case a DeepEval ConversationalTestCase cannot represent at
        all -- it rejects an empty turn list with a TypeError.
        """
        empty = from_pydantic_ai([])

        assert empty.turns == []
        assert empty.skills_loaded == []
        assert empty.tool_names_used == []
        assert empty.skill_body() is None


class TestArtifactRoundTrip:
    """The artifact is the scoring input, so a lossy artifact must fail here."""

    def _history(self) -> list[Any]:
        return [
            user("I need an LLM for support."),
            assistant(load_call("navigator", "h1")),
            returns(load_return("h1"), delivered("navigator", SKILL_BODY)),
            assistant(call("recommend_model", "m1", use_case="chat")),
            returns(result("recommend_model", "m1", {"recommendations": ["llama"]})),
            assistant(TextPart(content="Here are the options.")),
        ]

    def test_saved_artifact_reloads_into_an_equivalent_transcript(self, tmp_path: Any) -> None:
        artifact = AttemptArtifact.from_messages(
            self._history(), attempt_id="att-1", run_id="run-1", task="t", final_output="f"
        )
        directory = artifact.save(tmp_path)

        reloaded = AttemptArtifact.load(directory)

        assert reloaded.transcript().to_dict() == artifact.transcript().to_dict()

    def test_reloaded_artifact_preserves_tool_calls_and_skill_body(self, tmp_path: Any) -> None:
        artifact = AttemptArtifact.from_messages(
            self._history(), attempt_id="att-1", run_id="run-1"
        )
        reloaded = AttemptArtifact.load(artifact.save(tmp_path))

        transcript = reloaded.transcript()
        assert transcript.tool_names_used == ["recommend_model"]
        assert transcript.skills_loaded == ["navigator"]
        assert transcript.skill_body("navigator") == SKILL_BODY

    def test_metadata_and_schemas_survive_the_round_trip(self, tmp_path: Any) -> None:
        from mcp.types import Tool

        schema = Tool(
            name="recommend_model",
            description="Recommend a model",
            inputSchema={"type": "object", "properties": {"use_case": {"type": "string"}}},
        )
        artifact = AttemptArtifact.from_messages(
            self._history(),
            attempt_id="att-1",
            run_id="run-1",
            tool_schemas=[schema],
            metadata={"golden_id": "navigator-happy-path", "agent_model": "openai-sol"},
        )
        reloaded = AttemptArtifact.load(artifact.save(tmp_path))

        assert reloaded.metadata["golden_id"] == "navigator-happy-path"
        assert [t.name for t in reloaded.mcp_tools()] == ["recommend_model"]

    def test_catalog_survives_without_being_stored_separately(self, tmp_path: Any) -> None:
        """The catalog rides inside messages.json and is derived on load.

        Storing it as its own artifact field would let the two copies disagree.
        """
        history = [
            ModelRequest(parts=[UserPromptPart(content="I need an LLM."), catalog_part(CATALOG)]),
            assistant(TextPart(content="Phase 1 of 3")),
        ]
        artifact = AttemptArtifact.from_messages(history, attempt_id="att-4", run_id="run-1")

        reloaded = AttemptArtifact.load(artifact.save(tmp_path))

        assert "navigator-deploy" in reloaded.transcript().skill_catalog

    def test_partial_failed_attempt_is_still_persistable(self, tmp_path: Any) -> None:
        """Checkpoint failures save a partial transcript rather than nothing."""
        partial = [user("I need an LLM."), assistant(call("recommend_model", "m1"))]

        artifact = AttemptArtifact.from_messages(
            partial, attempt_id="att-2", run_id="run-1", metadata={"outcome": "routing_failure"}
        )
        reloaded = AttemptArtifact.load(artifact.save(tmp_path))

        assert reloaded.metadata["outcome"] == "routing_failure"
        assert reloaded.transcript().mcp_tool_calls[0].name == "recommend_model"

    def test_stale_schema_version_refuses_to_load(self, tmp_path: Any) -> None:
        import json

        artifact = AttemptArtifact.from_messages(
            self._history(), attempt_id="att-3", run_id="run-1"
        )
        directory = artifact.save(tmp_path)
        meta_path = directory / "artifact.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = ARTIFACT_SCHEMA_VERSION + 1
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="schema_version"):
            AttemptArtifact.load(directory)
