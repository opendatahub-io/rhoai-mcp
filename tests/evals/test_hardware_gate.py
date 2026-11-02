"""Customer hardware information/consent must precede every recommendation."""

from dataclasses import replace
from typing import Any, Literal

import pytest
from evals.agent.transcript import AgentTranscript, ToolCallRecord, TranscriptTurn, from_pydantic_ai
from evals.skills.checks import check_navigator
from evals.skills.goldens import GPU_REPLY, NAVIGATOR_GOLDENS, NAVIGATOR_HAPPY_PATH

from tests.evals.test_skill_agreement import workflow

CHECK = "navigator_recommend_after_hardware_confirmation"


def conversation(
    indices: list[int], *, hardware_role: Literal["user", "assistant"] = "user"
) -> AgentTranscript:
    transcript = AgentTranscript(
        task="Support chatbot",
        final_output="",
        turns=[
            TranscriptTurn(role="user", content="Support chatbot for 50 users."),
            TranscriptTurn(role="assistant", content="Confirm the workload?"),
            TranscriptTurn(role="user", content="Confirmed."),
            TranscriptTurn(role="assistant", content="Which GPUs are available?"),
            TranscriptTurn(role=hardware_role, content=GPU_REPLY),
            TranscriptTurn(role="assistant", content="Recommendations"),
        ],
    )
    transcript.tool_calls = [
        ToolCallRecord(
            call_id=f"r{i}",
            name="recommend_model",
            raw_name="recommend_model",
            arguments={"preferred_gpu_types": []},
            kind="mcp",
            turn_index=index,
        )
        for i, index in enumerate(indices)
    ]
    return transcript


def status(transcript: AgentTranscript, reply: str | None = GPU_REPLY) -> str:
    report = check_navigator(transcript, hardware_reply=reply, expect_handoff=False)
    return next(c.status for c in report.results if c.name == CHECK)


def test_recommendation_waits_for_customer_hardware_reply() -> None:
    assert status(conversation([5])) == "pass"


@pytest.mark.parametrize("indices", [[3], [3, 5], [4], [0, 5]])
def test_later_valid_calls_cannot_hide_a_premature_recommendation(indices: list[int]) -> None:
    assert status(conversation(indices)) == "fail"


def test_guessing_the_right_gpu_does_not_supply_customer_confirmation() -> None:
    transcript = conversation([3, 5])
    transcript.tool_calls[0].arguments["preferred_gpu_types"] = ["A100-80"]
    assert status(transcript) == "fail"


def test_assistant_claim_is_not_the_customer_hardware_reply() -> None:
    assert status(conversation([5], hardware_role="assistant")) == "fail"


def test_missing_configured_hardware_reply_cannot_be_a_pass() -> None:
    assert status(conversation([5]), "The cluster has H100 GPUs.") == "fail"


@pytest.mark.parametrize("reply", ["", "   "])
def test_empty_hardware_marker_cannot_match_an_unrelated_user_turn(reply: str) -> None:
    assert status(conversation([5]), reply) == "fail"


def test_explicit_unfiltered_consent_is_a_valid_hardware_gate() -> None:
    transcript = conversation([5])
    reply = "I do not know the GPUs; show recommendations without a hardware filter."
    transcript.turns[4].content = reply
    assert status(transcript, reply) == "pass"


def test_unconfigured_gate_and_unattempted_recommendations_have_distinct_statuses() -> None:
    assert status(conversation([5]), None) == "unexercised"
    assert status(conversation([])) == "n/a"


@pytest.mark.parametrize("golden", NAVIGATOR_GOLDENS, ids=lambda g: g.id)
def test_navigator_goldens_bind_their_hardware_reply_and_version_the_intent(golden: Any) -> None:
    assert getattr(golden, "hardware_reply", None) == GPU_REPLY
    assert replace(golden, hardware_reply=None).scenario_sha() != golden.scenario_sha()


def test_real_failure_pattern_is_caught_even_after_a_correct_handoff() -> None:
    transcript = from_pydantic_ai(workflow())
    hardware_at = next(
        i for i, t in enumerate(transcript.turns) if t.role == "user" and t.content == GPU_REPLY
    )
    transcript.tool_calls.insert(
        0,
        ToolCallRecord(
            call_id="premature",
            name="recommend_model",
            raw_name="recommend_model",
            arguments={"preferred_gpu_types": []},
            kind="mcp",
            turn_index=hardware_at - 1,
        ),
    )
    report = NAVIGATOR_HAPPY_PATH.grade(transcript)
    assert {c.name for c in report.failures} == {CHECK}
