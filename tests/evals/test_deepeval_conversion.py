"""Tests for building typed DeepEval objects from the shared transcript.

DeepEval validates MCP results by ``isinstance`` against ``mcp.types``, and
silently scores whatever turns it is handed. These tests pin the two things that
would otherwise fail quietly: wrong turn attribution, and harness tools being
counted as MCP interactions.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("pydantic_ai", reason="requires the skill-eval dependency group")

from deepeval.test_case import MCPServer  # noqa: E402
from evals.agent.transcript import from_pydantic_ai  # noqa: E402
from evals.deepeval_helpers import (  # noqa: E402
    NothingToScoreError,
    build_mcp_server_from_schemas,
    transcript_to_conversational_test_case,
)
from mcp.types import CallToolResult, Tool  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    CapabilityInstructionSource,
    InstructionDeltaPart,
    InstructionId,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

SKILL_BODY = "# Skill: navigator\n\nPhase 1 of 3 - understand requirements."


@pytest.fixture
def mcp_server() -> MCPServer:
    return build_mcp_server_from_schemas(
        [
            Tool(
                name="recommend_model",
                description="Recommend a model",
                inputSchema={"type": "object", "properties": {"use_case": {"type": "string"}}},
            )
        ]
    )


@pytest.fixture
def history() -> list[Any]:
    """A realistic navigator conversation: route, gate, call a tool, answer."""
    return [
        ModelRequest(parts=[UserPromptPart(content="I need an LLM for support.")]),
        ModelResponse(
            parts=[
                LoadCapabilityCallPart(
                    tool_name="load_capability",
                    args=json.dumps({"id": "navigator"}),
                    tool_call_id="h1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                LoadCapabilityReturnPart(
                    tool_name="load_capability",
                    content={"message": "Capability 'navigator' loaded."},
                    tool_call_id="h1",
                ),
                InstructionDeltaPart(
                    id=InstructionId(source=CapabilityInstructionSource(id="navigator")),
                    content=SKILL_BODY,
                ),
            ]
        ),
        ModelResponse(parts=[TextPart(content="Phase 1 of 3 - what matters most?")]),
        ModelRequest(parts=[UserPromptPart(content="Speed, then cost.")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="recommend_model", args={"use_case": "chat"}, tool_call_id="m1"
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="recommend_model",
                    content={"recommendations": [{"model": "llama-3.1-8b"}]},
                    tool_call_id="m1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="Phase 2 of 3 - here are the options.")]),
    ]


class TestConversationalTestCase:
    def test_only_mcp_calls_are_counted_as_mcp_interactions(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        called = [t for t in case.turns if t.mcp_tools_called]
        assert len(called) == 1
        assert [c.name for c in called[0].mcp_tools_called] == ["recommend_model"]

    def test_call_is_attached_to_the_turn_that_issued_it(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        """The issuing turn carries the call; the narration lands in the next turn.

        The regression this guards is the call drifting onto the skill-loading
        turn, which is what positional matching used to do.
        """
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        index = next(i for i, t in enumerate(case.turns) if t.mcp_tools_called)

        # Not the skill-loading turn.
        assert "loaded skill" not in (case.turns[index].content or "")
        # The user message that prompted it comes immediately before.
        assert case.turns[index - 1].content == "Speed, then cost."
        # The narration of the result comes immediately after.
        assert "Phase 2 of 3" in (case.turns[index + 1].content or "")

    def test_results_are_typed_call_tool_results(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        """DeepEval rejects dict-shaped results, so this must be the real type."""
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        (turn,) = [t for t in case.turns if t.mcp_tools_called]
        result = turn.mcp_tools_called[0].result
        assert isinstance(result, CallToolResult)
        assert result.structuredContent == {"recommendations": [{"model": "llama-3.1-8b"}]}
        assert result.isError is False

    def test_mcp_interaction_flag_is_derived_not_assigned(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        """Regression guard for the deleted ``turn._mcp_interaction = True`` hack."""
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        (turn,) = [t for t in case.turns if t.mcp_tools_called]
        assert turn._mcp_interaction is True

    def test_skill_loading_remains_visible_without_being_an_mcp_call(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        loading = [t for t in case.turns if "loaded skill" in (t.content or "")]
        assert len(loading) == 1
        assert not loading[0].mcp_tools_called

    def test_user_turns_are_preserved_in_order(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        assert [t.content for t in case.turns if t.role == "user"] == [
            "I need an LLM for support.",
            "Speed, then cost.",
        ]

    def test_fidelity_case_includes_the_delivered_skill_body(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        """A fidelity rubric cannot judge adherence to a document it never saw."""
        case = transcript_to_conversational_test_case(
            from_pydantic_ai(history), mcp_server, include_skill_body=True
        )

        assert any(SKILL_BODY in (t.content or "") for t in case.turns)

    def test_skill_body_does_not_displace_the_opening_user_turn(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        """DeepEval derives a unit's task from the *leading* user turns.

        An assistant turn prepended ahead of them leaves the first unit with an
        empty task and demotes the real opening message to a step.
        """
        case = transcript_to_conversational_test_case(
            from_pydantic_ai(history), mcp_server, include_skill_body=True
        )

        assert case.turns[0].role == "user"
        assert case.turns[0].content == "I need an LLM for support."
        assert SKILL_BODY in (case.turns[1].content or "")

    def test_empty_transcript_raises_rather_than_crashing_deepeval(
        self, mcp_server: MCPServer
    ) -> None:
        """A harness error can end an attempt before any message exists.

        DeepEval rejects an empty ConversationalTestCase with a bare TypeError;
        callers need a typed signal so they can record ``n/a`` with the missing
        precondition instead of a zero score.
        """
        empty = from_pydantic_ai([])

        with pytest.raises(NothingToScoreError, match="no turns"):
            transcript_to_conversational_test_case(empty, mcp_server)

    def test_mcp_case_omits_the_skill_body_by_default(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        assert not any(SKILL_BODY in (t.content or "") for t in case.turns)

    def test_failed_call_is_flagged_as_an_error_result(self, mcp_server: MCPServer) -> None:
        history = [
            ModelRequest(parts=[UserPromptPart(content="Deploy it.")]),
            ModelResponse(
                parts=[ToolCallPart(tool_name="recommend_model", args={}, tool_call_id="x1")]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="recommend_model",
                        content="boom",
                        tool_call_id="x1",
                        outcome="failed",
                    )
                ]
            ),
            ModelResponse(parts=[TextPart(content="That failed because...")]),
        ]

        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        (turn,) = [t for t in case.turns if t.mcp_tools_called]
        assert turn.mcp_tools_called[0].result.isError is True
