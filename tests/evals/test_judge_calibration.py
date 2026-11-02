"""Calibration must preserve provenance and never hide errors or overlapping scores."""

from __future__ import annotations

from dataclasses import asdict


def test_errors_and_unreviewed_examples_cannot_produce_a_threshold() -> None:
    from evals.skills.calibrate import recommend_threshold, score_summary

    assert recommend_threshold([0.8], []) is None
    assert recommend_threshold([0.8], [0.9]) is None
    assert recommend_threshold([0.8, 0.9], [0.1, 0.3]) == 0.55
    summary = score_summary(
        [
            {"score": 0.8, "error": None},
            {"score": None, "error": "timeout"},
        ]
    )
    assert summary["n"] == 1
    assert summary["errors"] == 1
    assert summary["mean"] == 0.8


def test_counterexamples_leave_the_frozen_source_untouched() -> None:
    from evals.agent.transcript import AgentTranscript, ToolCallRecord, TranscriptTurn
    from evals.skills.calibrate import mutate_transcript

    call = ToolCallRecord(
        call_id="a",
        name="recommend_model",
        raw_name="recommend_model",
        kind="mcp",
        arguments={"user_count": 50, "preferred_gpu_types": ["A100-80"]},
        turn_index=1,
    )
    source = AgentTranscript(
        task="Choose a model",
        final_output="Here is the handoff",
        turns=[
            TranscriptTurn("user", "Choose a model"),
            TranscriptTurn("assistant", "Here is the handoff", [call]),
        ],
        tool_calls=[call],
    )
    before = asdict(source)

    altered = mutate_transcript(source, "wrong-arguments")

    assert asdict(source) == before
    assert altered.mcp_tool_calls[0].arguments["user_count"] == 5000
    assert altered.turns[1].mcp_tool_calls[0] is altered.mcp_tool_calls[0]
    assert altered.mcp_tool_calls[0].arguments["preferred_gpu_types"] == ["H100"]


def test_threshold_requires_repeats_and_reports_overlap_without_zero_substitution() -> None:
    from evals.skills.calibrate import summarize

    rows = [
        {
            "model": "judge",
            "metric": "completion",
            "example": example,
            "expected": expected,
            "score": score,
            "error": None,
        }
        for example, expected, score in [("good", True, 0.9), ("bad", False, 0.6)]
        for _ in range(5)
    ]
    assert summarize(rows)["thresholds"][0]["threshold"] == 0.75
    rows[0]["score"] = 0.6
    assert summarize(rows)["thresholds"][0]["threshold"] is None
    rows[0].update(score=None, error="judge unavailable")
    result = summarize(rows)
    assert result["thresholds"][0]["threshold"] is None
    assert result["cells"][0]["mean"] == 0.6
    assert result["cells"][1]["errors"] == 1
