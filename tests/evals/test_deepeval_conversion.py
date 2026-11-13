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
    def test_fidelity_sees_matched_plan_warnings_as_internal_evidence(
        self, mcp_server: MCPServer
    ) -> None:
        warning = "Authentication must be configured before external access."
        issue = "A serving runtime must be created first."
        history = [
            ModelRequest(parts=[UserPromptPart(content="Plan the deployment.")]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="plan_deployment", args={}, tool_call_id="plan-1"),
                    ToolCallPart(tool_name="other_tool", args={}, tool_call_id="other-1"),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="other_tool",
                        tool_call_id="other-1",
                        content={"warnings": ["Unrelated warning"]},
                    ),
                    ToolReturnPart(
                        tool_name="plan_deployment",
                        tool_call_id="plan-1",
                        content={
                            "ready": False,
                            "issues": [issue],
                            "warnings": [warning],
                            "configs": {"inferenceservice": "large YAML payload" * 1000},
                        },
                    ),
                ]
            ),
            ModelResponse(parts=[TextPart(content="Here is the plan. Approve?")]),
            ModelRequest(parts=[UserPromptPart(content="Yes, deploy.")]),
        ]
        transcript = from_pydantic_ai(history)
        default_case = transcript_to_conversational_test_case(transcript, mcp_server)
        fidelity = transcript_to_conversational_test_case(
            transcript, mcp_server, annotate_tool_calls=True
        )
        call_turn = next(t for t in fidelity.turns if t.mcp_tools_called)

        assert warning in call_turn.content
        assert issue in call_turn.content
        assert '"ready": false' in call_turn.content
        assert "call_id=plan-1" in call_turn.content
        assert "not customer-visible prose" in call_turn.content
        assert "Unrelated warning" not in call_turn.content
        assert "large YAML payload" not in call_turn.content
        assert fidelity.turns.index(call_turn) < next(
            i for i, t in enumerate(fidelity.turns) if t.content == "Yes, deploy."
        )
        assert warning not in "\n".join(t.content for t in default_case.turns)
        assert warning not in transcript.assistant_text
        assert call_turn.mcp_tools_called[0].result.structuredContent["result"]["warnings"] == [
            warning
        ]

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
        assert result.isError is False

    def test_structured_content_is_wrapped_under_a_result_key(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        """DeepEval indexes ``structuredContent["result"]`` with no fallback.

        Passing the raw tool result through unwrapped reads as more faithful but
        makes every MCP metric raise ``KeyError: 'result'`` before the judge is
        called — a failure invisible to dry runs, which never score.
        """
        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        (turn,) = [t for t in case.turns if t.mcp_tools_called]
        structured = turn.mcp_tools_called[0].result.structuredContent

        assert structured is not None
        assert structured["result"] == {"recommendations": [{"model": "llama-3.1-8b"}]}

    def test_deepeval_can_render_the_interaction_text(
        self, history: list[Any], mcp_server: MCPServer
    ) -> None:
        """End-to-end guard: run DeepEval's own renderer over our turns.

        This is the exact call that failed in the first judged run, so asserting
        against the library rather than our own expectation is what keeps the
        contract honest across DeepEval upgrades.
        """
        from deepeval.metrics.mcp.utils import turn_mcp_interaction_text

        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)
        (turn,) = [t for t in case.turns if t.mcp_tools_called]

        rendered = turn_mcp_interaction_text(turn)

        assert "recommend_model" in rendered
        assert "llama-3.1-8b" in rendered

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

    @pytest.mark.parametrize(
        ("result", "outcome", "diagnostic"),
        [
            (
                ToolReturnPart(
                    tool_name="load_capability",
                    tool_call_id="h1",
                    content="not found",
                    outcome="failed",
                ),
                "failed",
                "not found",
            ),
            (
                ToolReturnPart(
                    tool_name="load_capability",
                    tool_call_id="h1",
                    content="try again",
                    outcome="retried",
                ),
                "retried",
                "try again",
            ),
            (None, "incomplete", "tool call did not return"),
        ],
    )
    def test_failed_skill_load_is_visible_before_recovery(
        self, result: Any, outcome: str, diagnostic: str, mcp_server: MCPServer
    ) -> None:
        history = [
            ModelRequest(parts=[UserPromptPart(content="Help me.")]),
            ModelResponse(
                parts=[LoadCapabilityCallPart(args={"id": "navigator"}, tool_call_id="h1")]
            ),
        ]
        if result is not None:
            history.append(ModelRequest(parts=[result]))
        history.extend(
            [
                ModelResponse(
                    parts=[LoadCapabilityCallPart(args={"id": "navigator"}, tool_call_id="h2")]
                ),
                ModelRequest(
                    parts=[
                        LoadCapabilityReturnPart(content={"message": "loaded"}, tool_call_id="h2")
                    ]
                ),
            ]
        )

        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        failed, recovered = case.turns[1:]
        assert "navigator" in failed.content
        assert f"outcome: {outcome}" in failed.content
        assert diagnostic in failed.content
        assert "loaded skill" not in failed.content
        assert recovered.content == "[loaded skill: navigator]"
        assert not failed.mcp_tools_called and not recovered.mcp_tools_called

    def test_skill_load_is_visible_in_turn_with_text_and_mcp_call(
        self, mcp_server: MCPServer
    ) -> None:
        history = [
            ModelRequest(parts=[UserPromptPart(content="Help me.")]),
            ModelResponse(
                parts=[
                    TextPart(content="Checking."),
                    LoadCapabilityCallPart(args={"capability": "navigator"}, tool_call_id="h1"),
                    ToolCallPart(tool_name="recommend_model", args={}, tool_call_id="m1"),
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="load_capability",
                        tool_call_id="h1",
                        content="denied",
                        outcome="denied",
                    ),
                    ToolReturnPart(tool_name="recommend_model", tool_call_id="m1", content="ok"),
                ]
            ),
        ]

        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        turn = case.turns[1]
        assert "Checking." in turn.content
        assert "navigator; outcome: denied; denied" in turn.content
        assert [call.name for call in turn.mcp_tools_called] == ["recommend_model"]

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

    def test_unreturned_call_is_flagged_as_an_error_result(self, mcp_server: MCPServer) -> None:
        history = [
            ModelRequest(parts=[UserPromptPart(content="Recommend a model.")]),
            ModelResponse(
                parts=[ToolCallPart(tool_name="recommend_model", args={}, tool_call_id="x1")]
            ),
        ]

        case = transcript_to_conversational_test_case(from_pydantic_ai(history), mcp_server)

        (turn,) = [t for t in case.turns if t.mcp_tools_called]
        assert turn.mcp_tools_called[0].result.isError is True
