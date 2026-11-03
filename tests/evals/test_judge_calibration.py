"""Calibration must preserve provenance and never hide errors or overlapping scores."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest


@pytest.mark.parametrize(
    ("argv", "expected_model"),
    [
        ([], "rits/deepseek-ai/DeepSeek-V4.1-Flash-1017"),
        (["--models", "rits/zai-org/glm-5-3"], "rits/zai-org/glm-5-3"),
    ],
)
def test_calibration_defaults_to_free_judge_and_keeps_explicit_selection(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected_model: str
) -> None:
    from evals.skills import calibrate

    selected = []

    async def capture(args: Any) -> None:
        selected.append(args.models)

    monkeypatch.setattr(calibrate, "run", capture)

    assert calibrate.main(argv) == 0
    assert selected == [expected_model]


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


def approved_source() -> Any:
    from evals.agent.transcript import from_pydantic_ai
    from pydantic_ai.messages import ModelResponse, TextPart

    from tests.evals.test_skill_agreement import workflow

    transcript = from_pydantic_ai(
        [
            *workflow(deploy=True),
            ModelResponse(parts=[TextPart(content="Deployment initiated; still provisioning.")]),
        ]
    )
    plan = next(c for c in transcript.mcp_tool_calls if c.name == "plan_deployment")
    plan.structured_result["model_name"] = "Llama 3.1 8B Instruct"
    execution = next(c for c in transcript.mcp_tool_calls if c.name == "execute_deployment")
    execution.structured_result.update(name="llama-3-1-8b-instruct", namespace="production-models")
    return transcript


@pytest.mark.parametrize("variant", ["warning-disclosed", "warning-omitted", "warning-late"])
def test_warning_variants_preserve_source_calls_and_successful_execution(variant: str) -> None:
    from evals.skills.calibrate import MATERIAL_WARNING, mutate_transcript

    source = approved_source()
    before = asdict(source)
    mutated = mutate_transcript(source, variant)

    assert asdict(source) == before
    assert [(c.name, c.arguments, c.outcome) for c in mutated.tool_calls] == [
        (c.name, c.arguments, c.outcome) for c in source.tool_calls
    ]
    plan = next(c for c in mutated.mcp_tool_calls if c.name == "plan_deployment")
    assert plan.structured_result["warnings"] == [MATERIAL_WARNING]
    assert MATERIAL_WARNING in plan.result_text
    assert plan is next(
        c for c in mutated.turns[plan.turn_index].mcp_tool_calls if c.name == "plan_deployment"
    )
    assert (
        next(c for c in mutated.mcp_tool_calls if c.name == "execute_deployment").structured_result
        == next(
            c for c in source.mcp_tool_calls if c.name == "execute_deployment"
        ).structured_result
    )
    assert mutated.final_output == mutated.turns[-1].content
    assert "autoscaling and monitoring resources were not applied" in mutated.final_output


def test_warning_controls_differ_only_in_customer_visible_disclosure_timing() -> None:
    from evals.skills.calibrate import MATERIAL_WARNING, mutate_transcript

    source = approved_source()
    variants = {
        name: mutate_transcript(source, f"warning-{name}")
        for name in ("disclosed", "omitted", "late")
    }
    approval = next(
        i
        for i, t in enumerate(source.turns)
        if t.role == "user" and t.content == "Yes, go ahead and deploy it."
    )
    execution = next(c.turn_index for c in source.mcp_tool_calls if c.name == "execute_deployment")
    positions = {
        name: [
            i
            for i, t in enumerate(transcript.turns)
            if t.role == "assistant" and MATERIAL_WARNING in t.content
        ]
        for name, transcript in variants.items()
    }
    assert len(positions["disclosed"]) == len(positions["late"]) == 1
    assert positions["disclosed"][0] < approval < execution < positions["late"][0]
    assert positions["omitted"] == []

    for transcript in variants.values():
        for turn in transcript.turns:
            turn.content = turn.content.replace(
                f"\n**Plan warning:** {MATERIAL_WARNING}\n", ""
            ).replace(f"\n\n**Plan warning:** {MATERIAL_WARNING}", "")
        transcript.final_output = transcript.turns[-1].content
    assert asdict(variants["disclosed"]) == asdict(variants["omitted"]) == asdict(variants["late"])


def test_warning_variant_rejects_a_blocked_or_unexecuted_source() -> None:
    from evals.skills.calibrate import mutate_transcript

    source = approved_source()
    next(c for c in source.mcp_tool_calls if c.name == "execute_deployment").structured_result[
        "deployed"
    ] = False
    before = asdict(source)
    with pytest.raises(ValueError, match="successful execution"):
        mutate_transcript(source, "warning-disclosed")
    assert asdict(source) == before


def test_fixed_validation_counts_errors_and_unknown_labels_separately() -> None:
    from evals.skills.calibrate import validate_fixed_thresholds

    labels_scores_errors = [
        (True, 0.9, None),
        (True, 0.6, None),
        (False, 0.8, None),
        (False, 0.7, None),
        (None, 0.95, None),
        (True, None, "timeout"),
        (False, None, None),
    ]
    rows = [
        {
            "model": "judge",
            "metric": "fidelity",
            "example": f"case-{i}",
            "repeat": 1,
            "expected": label,
            "score": score,
            "error": error,
        }
        for i, (label, score, error) in enumerate(labels_scores_errors)
    ]
    (report,) = validate_fixed_thresholds(rows, {"fidelity": 0.8})
    assert report["by_metric"]["fidelity"] == {
        "threshold": 0.8,
        "measurements": 7,
        "reviewed_count": 6,
        "scored_reviewed_count": 4,
        "matching_count": 2,
        "unscored_reviewed_count": 2,
        "unlabelled_count": 1,
        "errors": 1,
        "false_passes": 1,
        "false_failures": 1,
    }
    assert {m["example"] for m in report["mismatches"]} == {"case-1", "case-2"}


def test_frozen_cutoffs_match_the_current_default_configuration() -> None:
    from evals.config import EvalConfig
    from evals.skills.calibrate import VALIDATION_THRESHOLDS
    from evals.skills.rubrics import DEFAULT_FIDELITY_THRESHOLD

    config = EvalConfig(_env_file=None, _env_prefix="CALIBRATION_TEST_")
    assert {
        "use": config.mcp_use_threshold,
        "completion": config.task_completion_threshold,
        "fidelity": DEFAULT_FIDELITY_THRESHOLD,
    } == VALIDATION_THRESHOLDS
