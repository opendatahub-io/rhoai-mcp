"""Offline comparison policy, usage accounting, portability, and CLI integration."""

from __future__ import annotations

import hashlib
import json
import socket
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from evals.reporting.cli import main
from evals.reporting.comparison import provider_comparison_report
from evals.reporting.formatting import format_summary, format_table
from evals.reporting.models import (
    CheckRecord,
    EnvironmentRecord,
    EvalRecord,
    GitRecord,
    MetricRecord,
)
from evals.reporting.observations import read_agent_observations
from evals.reporting.reader import load_records, record_from_dict
from evals.reporting.skill_comparison import (
    QUALITY_METRICS,
    build_skill_comparison,
    render_skill_comparison,
)


def record(model: str, golden: str) -> EvalRecord:
    return EvalRecord(
        run_id=f"run-{model}",
        timestamp="2026-11-02T12:00:00+00:00",
        scenario=golden,
        git=GitRecord("abc1234", "branch"),
        environment=EnvironmentRecord(
            "pydantic-ai",
            model,
            "",
            "enmaas",
            "judge",
            "mock",
            0.6,
            0.75,
            "https://judge.invalid/v1",
        ),
        attempt_id=f"{model}-{golden}",
        scoring_id=f"score-{model}-{golden}",
        skill="navigator",
        golden_id=golden,
        skill_sha="frozen-skill",
        scenario_sha=f"sha-{golden}",
        rubric_version="v2",
        metrics=[
            MetricRecord(name, score, True, threshold, "reviewed", version)
            for name, score, threshold, version in (
                (QUALITY_METRICS["tool_use"], 0.8, 0.6, "conversation-v2"),
                (QUALITY_METRICS["completion"], 0.9, 0.75, "conversation-v2"),
                (QUALITY_METRICS["fitness"], 0.85, 0.8, "v2"),
            )
        ],
        turns=5,
        skills_loaded=["navigator"],
        passed=True,
        agent_seconds=20.0,
        duration_seconds=8.0,
        checks=[
            CheckRecord("routing_loaded_expected_skill", "pass"),
            CheckRecord("table_cells_complete", "pass"),
        ],
    )


def usage(input_tokens: int, output_tokens: int, reasoning: int, cached: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_reasoning_tokens": reasoning,
        "cache_read_tokens": cached,
    }


def artifact(root: Path, r: EvalRecord) -> Path:
    path = root / r.run_id / r.attempt_id
    path.mkdir(parents=True)
    r.artifact_path = str(path)
    metadata = {
        "id": r.environment.agent_model,
        "model_name": f"actual-{r.environment.agent_model}",
        "base_url": "https://agent.invalid/v1",
        "api": "responses",
        "temperature": None,
        "sampling": "provider-default",
        "extra_settings": {},
        "agent_backend": "pydantic-ai",
        "skill_sha": r.skill_sha,
        "scenario_sha": r.scenario_sha,
        "rubric_version": r.rubric_version,
        "elapsed_s": 21.5,
    }
    (path / "artifact.json").write_text(
        json.dumps(
            {
                "run_id": r.run_id,
                "attempt_id": r.attempt_id,
                "metadata": metadata,
                "tool_schemas": [{"name": "recommend_model"}, {"name": "load_capability"}],
            }
        )
    )
    (path / "messages.json").write_text(
        json.dumps(
            [
                {
                    "kind": "request",
                    "parts": [{"part_kind": "user-prompt", "content": "choose a model"}],
                },
                {
                    "kind": "response",
                    "usage": usage(100, 10, 3, 20),
                    "parts": [
                        {
                            "part_kind": "tool-call",
                            "tool_kind": "capability-load",
                            "tool_name": "load_capability",
                            "tool_call_id": "load",
                        },
                        {
                            "part_kind": "tool-call",
                            "tool_name": "mcp__server__recommend_model",
                            "tool_call_id": "rec",
                        },
                    ],
                },
                {
                    "kind": "request",
                    "parts": [
                        {
                            "part_kind": "tool-return",
                            "tool_name": "recommend_model",
                            "tool_call_id": "rec",
                            "outcome": "success",
                            "content": {"error": "planner error"},
                        },
                        {
                            "part_kind": "tool-return",
                            "tool_kind": "capability-load",
                            "tool_name": "load_capability",
                            "tool_call_id": "load",
                            "outcome": "success",
                            "content": "loaded",
                        },
                    ],
                },
                {
                    "kind": "response",
                    "usage": usage(150, 30, 7, 100),
                    "parts": [{"part_kind": "text", "content": "a recommendation"}],
                },
            ]
        )
    )
    return path


@pytest.fixture
def records(tmp_path: Path) -> list[EvalRecord]:
    rows = [record(model, golden) for model in ("a", "b") for golden in ("happy", "vague")]
    for r in rows:
        artifact(tmp_path, r)
    return rows


def build(records: list[EvalRecord], **kwargs: Any) -> Any:
    return build_skill_comparison(records, run_ids=sorted({r.run_id for r in records}), **kwargs)


def cohort(records: list[EvalRecord]) -> dict[str, Any]:
    return {
        "records": [asdict(r) for r in records],
        "frozen_configuration": {
            "source_commit": "abc1234full",
            "skill_sha": {"navigator": "frozen-skill"},
            "scenario_sha": {r.golden_id: r.scenario_sha for r in records},
            "judge": {
                "provider": "enmaas",
                "model": "judge",
                "base_url": "https://judge.invalid/v1",
            },
            "thresholds": {"use": 0.6, "completion": 0.75, "fidelity": 0.8},
            "versions": {"mcp": "conversation-v2", "fidelity": "v2"},
        },
        "execution_overlaps": [{"agents": ["a", "b"], "seconds": 12.5}],
    }


def test_quality_usage_turns_and_total_time_are_separate(records: list[EvalRecord]) -> None:
    records[1].passed = False
    records[1].checks[1].status = "fail"
    result = build(records)
    a = result.models[0]
    assert (a.samples, a.passed, a.hard_checks_passed, a.routing_passed) == (2, 1, 1, 2)
    assert a.metrics["fitness"].mean == 0.85
    assert a.efficiency["output_tokens"].mean == 40
    assert a.efficiency["reasoning_tokens"].mean == 10  # Included in 40, not added to it.
    assert a.efficiency["input_tokens"].mean == 250
    assert a.efficiency["cache_read_tokens"].mean == 120
    assert a.efficiency["turns"].mean == 5 and a.efficiency["model_responses"].mean == 2
    assert a.efficiency["total_seconds"].mean == 28
    assert a.efficiency["total_seconds"].observed_total == 56
    assert a.efficiency["mcp_calls"].mean == 1 and a.efficiency["mcp_errors"].mean == 1
    assert a.efficiency["capability_loads"].mean == 1


def test_report_explains_fractional_means_in_markdown_and_json(records: list[EvalRecord]) -> None:
    path = Path(records[0].artifact_path) / "messages.json"
    messages = json.loads(path.read_text())
    messages.extend(
        [
            {
                "kind": "response",
                "usage": usage(10, 2, 0, 0),
                "parts": [
                    {
                        "part_kind": "tool-call",
                        "tool_name": "recommend_model",
                        "tool_call_id": "second",
                    }
                ],
            },
            {
                "kind": "request",
                "parts": [{"part_kind": "tool-return", "tool_call_id": "second", "content": {}}],
            },
        ]
    )
    path.write_text(json.dumps(messages))
    report = build(records)
    assert report.models[0].efficiency["mcp_calls"].mean == 1.5
    markdown = render_skill_comparison(report, "markdown")
    payload = json.loads(render_skill_comparison(report, "json"))
    assert "average 5.5 calls" in markdown
    assert "not a combined quality score" in markdown
    assert "average work and time per conversation" in markdown
    assert "average 5.5 calls" in payload["explanations"]["usage_and_tools"]
    assert payload["models"][0]["efficiency"]["mcp_calls"]["mean"] == 1.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("skill_sha", "new"),
        ("scenario_sha", "new"),
        ("rubric_version", "new"),
        ("git.commit", "new"),
        ("environment.eval_model", "other"),
        ("environment.eval_provider", "other"),
        ("environment.eval_model_base_url", "https://other.invalid"),
        ("environment.cluster_mode", "live"),
        ("environment.mcp_use_threshold", 0.7),
        ("metric.version", "old"),
        ("metric.threshold", 0.9),
    ],
)
def test_incompatible_configurations_are_rejected(
    records: list[EvalRecord], field: str, value: Any
) -> None:
    # Avoid artifact mismatch taking precedence over cross-record compatibility.
    for r in records:
        r.artifact_path = ""
    target: Any = records[-1]
    if field.startswith("metric."):
        target = target.metrics[0]
        field = field.split(".")[1]
    elif "." in field:
        parent, field = field.split(".")
        target = getattr(target, parent)
    setattr(target, field, value)
    with pytest.raises(ValueError, match="Incompatible"):
        build(records)


def test_unequal_golden_sets_refuse_aggregate_but_common_scenario_works(
    records: list[EvalRecord],
) -> None:
    with pytest.raises(ValueError, match="different golden sets"):
        build(records[:-1])
    assert [m.samples for m in build(records[:-1], scenario="happy").models] == [1, 1]


def test_unequal_sampling_mix_is_not_silently_averaged(records: list[EvalRecord]) -> None:
    extra = replace(records[0], attempt_id="extra", artifact_path="")
    with pytest.raises(ValueError, match="sampling proportions"):
        build([*records, extra])


def test_dry_runs_duplicates_and_rescores_do_not_inflate_samples(records: list[EvalRecord]) -> None:
    duplicate = deepcopy(records[0])
    rescore = replace(
        records[0],
        run_id="rescoring-run",
        timestamp="2026-11-03T12:00:00+00:00",
        scoring_id="new-score",
        skill_sha="new-skill",
    )
    dry = replace(records[0], run_id="dry", attempt_id="dry-attempt", dry_run=True)
    report = build([*records, duplicate, rescore, dry])
    assert sum(m.samples for m in report.models) == 4
    assert (
        report.excluded["duplicate"]
        == report.excluded["rescore"]
        == report.excluded["dry_run"]
        == 1
    )


def test_latest_scoring_reuses_original_attempt_and_execution_time(
    records: list[EvalRecord],
) -> None:
    original = records[0]
    rescored = deepcopy(original)
    rescored.timestamp = "2026-11-03T12:00:00+00:00"
    rescored.scoring_id = "new-score"
    rescored.agent_seconds = 0.0  # Replay did not run the agent.
    rescored.duration_seconds = 12.0
    rescored.metrics[0].score = 0.95
    report = build_skill_comparison(
        [*records, rescored], run_ids=["run-a", "run-b"], scoring="latest", concurrent=True
    )
    selected = next(a for a in report.attempts if a.record.attempt_id == original.attempt_id)
    assert len(report.attempts) == 4
    assert selected.record.scoring_id == "new-score"
    assert selected.record.metrics[0].score == 0.95
    assert selected.agent_seconds == 20.0
    assert selected.total_seconds == 32.0
    assert report.excluded["rescore"] == 1
    assert report.configuration["concurrent"] is True
    assert "Attempts ran concurrently" in render_skill_comparison(report, "markdown")


def test_latest_scoring_can_select_replay_by_original_run(records: list[EvalRecord]) -> None:
    rescored = replace(
        records[0], run_id="rescored", scoring_id="new-score", timestamp="2026-11-03T12:00:00+00:00"
    )
    report = build_skill_comparison([*records, rescored], run_ids=["run-a"], scoring="latest")
    assert len(report.attempts) == 2
    assert (
        next(a for a in report.attempts if a.record.attempt_id == rescored.attempt_id).record
        is rescored
    )


def test_latest_scoring_uses_original_checks_only_execution_time(records: list[EvalRecord]) -> None:
    original = records[0]
    original.dry_run = True
    rescored = replace(
        original,
        dry_run=False,
        scoring_id="new-score",
        timestamp="2026-11-03T12:00:00+00:00",
        agent_seconds=0.0,
    )
    report = build_skill_comparison([*records, rescored], run_ids=["run-a"], scoring="latest")
    selected = next(a for a in report.attempts if a.record.attempt_id == original.attempt_id)
    assert selected.record.scoring_id == "new-score"
    assert selected.agent_seconds == 20.0


def test_latest_replay_can_use_a_new_rubric(records: list[EvalRecord]) -> None:
    original = records[0]
    rescored = deepcopy(original)
    rescored.timestamp = "2026-11-03T12:00:00+00:00"
    rescored.scoring_id = "new-score"
    rescored.is_rescore = True
    rescored.agent_seconds = 0
    rescored.rubric_version = "v3"
    rescored.metrics[2].version = "v3"
    report = build_skill_comparison([original, rescored], run_ids=["run-a"], scoring="latest")
    assert report.configuration["rubric_version"] == "v3"
    assert report.attempts[0].record.scoring_id == "new-score"


@pytest.mark.parametrize("marker", [True, None])
def test_replay_only_uses_artifact_time_and_is_not_original(
    records: list[EvalRecord], marker: bool | None
) -> None:
    rescored = replace(records[0], agent_seconds=0, is_rescore=marker)
    report = build_skill_comparison([rescored], run_ids=["run-a"], scoring="latest")
    assert report.attempts[0].agent_seconds == 21.5
    assert report.attempts[0].total_seconds == 29.5
    with pytest.raises(ValueError, match="No original judged"):
        build_skill_comparison([rescored], run_ids=["run-a"])


def test_marked_replay_without_artifact_has_unknown_execution_time(records: list[EvalRecord]) -> None:
    rescored = replace(records[0], agent_seconds=0, is_rescore=True, artifact_path="")
    report = build_skill_comparison([rescored], run_ids=["run-a"], scoring="latest")
    assert report.attempts[0].agent_seconds is None
    assert report.attempts[0].total_seconds is None
    with pytest.raises(ValueError, match="No original judged"):
        build_skill_comparison([rescored], run_ids=["run-a"])


def test_legacy_zero_without_artifact_has_unverified_origin(records: list[EvalRecord]) -> None:
    unknown = replace(records[0], agent_seconds=0, is_rescore=None, artifact_path="")
    report = build_skill_comparison([unknown], run_ids=["run-a"], scoring="latest")
    assert report.attempts[0].agent_seconds is None
    with pytest.raises(ValueError, match="No original judged"):
        build_skill_comparison([unknown], run_ids=["run-a"])


def test_latest_scoring_rejects_mixed_check_contracts(records: list[EvalRecord]) -> None:
    rescored = replace(
        records[0],
        scoring_id="new-score",
        timestamp="2026-11-03T12:00:00+00:00",
        checks_version="different",
    )
    with pytest.raises(ValueError, match="checks version"):
        build_skill_comparison([*records, rescored], run_ids=["run-a", "run-b"], scoring="latest")


def test_reused_tool_call_ids_count_occurrences_and_match_returns(
    records: list[EvalRecord],
) -> None:
    path = Path(records[0].artifact_path) / "messages.json"
    messages = json.loads(path.read_text())
    messages.extend(
        [
            {
                "kind": "response",
                "parts": [
                    {
                        "part_kind": "tool-call",
                        "tool_name": "recommend_model",
                        "tool_call_id": "reused",
                    },
                    {
                        "part_kind": "tool-call",
                        "tool_name": "recommend_model",
                        "tool_call_id": "reused",
                    },
                ],
            },
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "tool-return",
                        "tool_call_id": "reused",
                        "content": {"success": True},
                    },
                    {
                        "part_kind": "tool-return",
                        "tool_call_id": "reused",
                        "content": {"error": "bad"},
                    },
                ],
            },
            {
                "kind": "response",
                "parts": [
                    {
                        "part_kind": "tool-call",
                        "tool_name": "recommend_model",
                        "tool_call_id": "reused",
                    },
                ],
            },
        ]
    )
    path.write_text(json.dumps(messages))
    obs = read_agent_observations(records[0])
    assert (obs.mcp_calls, obs.mcp_errors, obs.unreturned_mcp_calls) == (4, 2, 1)


def test_rescore_is_detected_without_artifacts_even_when_original_run_not_selected(
    records: list[EvalRecord],
) -> None:
    for r in records:
        r.artifact_path = ""
    rescored = replace(records[0], run_id="rescored", timestamp="2026-11-03T12:00:00+00:00")
    result = build_skill_comparison([*records, rescored], run_ids=["run-b", "rescored"])
    assert result.excluded["rescore"] == 1
    assert [m.model for m in result.models] == ["b"]


@pytest.mark.parametrize("score", [-1.0, float("nan"), float("inf")])
def test_missing_or_errored_scores_are_not_zero(records: list[EvalRecord], score: float) -> None:
    records[0].metrics[0].score = score
    records[1].metrics = records[1].metrics[1:]
    result = build(records)
    stat = result.models[0].metrics["tool_use"]
    assert stat.mean is None and stat.measured == 0 and stat.expected == 2
    assert "n/a [0/2]" in render_skill_comparison(result, "markdown")
    payload = json.loads(render_skill_comparison(result, "json"))
    assert payload["models"][0]["metrics"]["tool_use"]["mean"] is None


def test_genuine_zero_score_is_preserved_with_partial_coverage(records: list[EvalRecord]) -> None:
    records[0].metrics[0].score = 0
    records[1].metrics = records[1].metrics[1:]
    stat = build(records).models[0].metrics["tool_use"]
    assert (stat.mean, stat.measured, stat.expected) == (0, 1, 2)


def test_missing_artifacts_leave_tokens_unavailable_without_losing_quality(
    records: list[EvalRecord],
) -> None:
    for r in records:
        r.artifact_path = ""
    report = build(records)
    assert report.models[0].metrics["tool_use"].mean == 0.8
    assert report.models[0].efficiency["output_tokens"].mean is None
    assert report.models[0].efficiency["mcp_calls"].mean is None
    assert any("No artifact path" in note for note in report.notes)


@pytest.mark.parametrize("change", ["missing-output", "unmeasured", "missing-cache"])
def test_partial_response_usage_is_not_an_undercounted_total(
    records: list[EvalRecord], change: str
) -> None:
    path = Path(records[0].artifact_path) / "messages.json"
    messages = json.loads(path.read_text())
    if change == "missing-output":
        del messages[-1]["usage"]["output_tokens"]
    elif change == "unmeasured":
        messages[-1]["usage"]["unmeasured_requests"] = 1
    else:
        del messages[-1]["usage"]["cache_read_tokens"]
    path.write_text(json.dumps(messages))
    obs = read_agent_observations(records[0])
    assert (obs.cache_read_tokens if change == "missing-cache" else obs.output_tokens) is None
    assert obs.model_responses == 2


def test_identical_cumulative_messages_are_not_counted_twice(records: list[EvalRecord]) -> None:
    path = Path(records[0].artifact_path) / "messages.json"
    messages = json.loads(path.read_text())
    messages += deepcopy(messages)
    path.write_text(json.dumps(messages))
    obs = read_agent_observations(records[0])
    assert obs.output_tokens == 40 and obs.model_responses == 2 and obs.mcp_calls == 1
    assert any("deduplicated" in warning for warning in obs.warnings)


def test_portable_artifact_root_finds_original_run_for_rescore(
    records: list[EvalRecord], tmp_path: Path
) -> None:
    original = records[0]
    moved = replace(
        original,
        run_id="rescore-run",
        artifact_path=f"/old/checkout/{original.run_id}/{original.attempt_id}",
    )
    obs = read_agent_observations(moved, tmp_path)
    assert obs.artifact_run_id == original.run_id and obs.output_tokens == 40


def test_missing_timing_is_not_zero_and_explicit_zero_survives(records: list[EvalRecord]) -> None:
    records[0].agent_seconds = records[0].duration_seconds = None
    records[0].artifact_path = ""
    records[1].agent_seconds = records[1].duration_seconds = 0
    records[1].is_rescore = False
    result = build(records)
    stat = result.models[0].efficiency["total_seconds"]
    assert (stat.mean, stat.measured, stat.expected) == (0, 1, 2)
    old = asdict(records[0])
    del old["agent_seconds"]
    del old["duration_seconds"]
    loaded = record_from_dict(old)
    assert loaded.agent_seconds is None and loaded.duration_seconds is None
    assert "n/a" in format_summary([loaded])


def test_item_fractions_remain_visible_in_summary_and_attempt_tables(
    records: list[EvalRecord],
) -> None:
    for r in records:
        r.metrics[0].score = 21 / 22
        r.metrics[1].score = 33 / 34
        for metric in r.metrics[:2]:
            metric.version = "conversation-v3-items"
    summary = format_summary([records[0]], fmt="markdown")
    comparison = render_skill_comparison(build(records), "markdown")
    for rendered in (summary, comparison):
        assert "0.9545" in rendered
        assert "0.9706" in rendered
    assert records[0].metrics[0].score == 21 / 22


def test_current_skill_files_are_not_consulted_and_overlap_is_disclosed(
    records: list[EvalRecord],
) -> None:
    result = build_skill_comparison([], cohort=cohort(records))
    assert result.configuration["skill_sha"] == "frozen-skill"
    assert any("12.50s" in note for note in result.notes)
    assert "not cohort elapsed time" in render_skill_comparison(result)


@pytest.mark.parametrize("change", ["judge", "threshold", "version"])
def test_cohort_configuration_is_checked_against_embedded_records(
    records: list[EvalRecord], change: str
) -> None:
    data = cohort(records)
    if change == "judge":
        data["frozen_configuration"]["judge"]["model"] = "different"
    elif change == "threshold":
        data["frozen_configuration"]["thresholds"]["fidelity"] = 0.9
    else:
        data["frozen_configuration"]["versions"]["mcp"] = "old"
    with pytest.raises(ValueError, match="Cohort"):
        build_skill_comparison([], cohort=data)


def test_agent_settings_cannot_change_within_model(records: list[EvalRecord]) -> None:
    p = Path(records[1].artifact_path) / "artifact.json"
    data = json.loads(p.read_text())
    data["metadata"]["extra_settings"] = {"temperature": 1}
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="settings changed"):
        build(records)


def test_native_usage_hash_must_match_frozen_evidence(records: list[EvalRecord]) -> None:
    data = cohort(records)
    data["case_details"] = [
        {
            "attempt_id": records[0].attempt_id,
            "native_sha256": {"original/messages.json": "wrong-hash"},
        }
    ]
    with pytest.raises(ValueError, match="hash differs"):
        build_skill_comparison([], cohort=data)


def test_missing_artifacts_do_not_make_a_rescore_an_original(records: list[EvalRecord]) -> None:
    original = records[0]
    rescored = replace(
        original,
        run_id="rescored",
        artifact_path=f"/missing/{original.run_id}/{original.attempt_id}",
    )
    with pytest.raises(ValueError, match="No original judged"):
        build_skill_comparison([rescored], run_ids=["rescored"])


def test_batch_order_cannot_silently_drop_selected_models(records: list[EvalRecord]) -> None:
    data = cohort(records)
    data["frozen_configuration"]["batch_order"] = ["a"]
    assert [m.model for m in build_skill_comparison([], cohort=data).models] == ["a", "b"]


def test_cli_writes_offline_json_and_preserves_input(
    records: list[EvalRecord], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(cohort(records)))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "output" / "comparison.json"

    def no_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Reporting must remain offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    main(["compare", "--cohort", str(path), "--format", "json", "--output", str(output)])
    result = json.loads(output.read_text())
    assert (
        len(result["attempts"]) == 4
        and result["models"][0]["efficiency"]["output_tokens"]["mean"] == 40
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_cli_refuses_to_overwrite_cohort(records: list[EvalRecord], tmp_path: Path) -> None:
    path = tmp_path / "cohort.json"
    contents = json.dumps(cohort(records))
    path.write_text(contents)
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--cohort", str(path), "--output", str(path)])
    assert exc.value.code == 2 and path.read_text() == contents


def test_cli_handles_incompatible_runs_with_actionable_error(
    records: list[EvalRecord], tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "history.jsonl"
    records[-1].skill_sha = "different"
    records[-1].artifact_path = ""
    path.write_text("\n".join(json.dumps(asdict(r)) for r in records))
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--file", str(path), "--run-id", "run-a", "--run-id", "run-b"])
    assert exc.value.code == 2 and "Incompatible skill hash" in capsys.readouterr().err


def test_skill_history_needs_selection_and_legacy_lcs_missing_scores_stay_na(
    records: list[EvalRecord],
) -> None:
    with pytest.raises(ValueError, match="explicit"):
        provider_comparison_report(records)
    legacy = [
        replace(
            records[0],
            skill="",
            metrics=[],
            environment=replace(records[0].environment, agent_backend="lcs"),
        ),
        replace(
            records[2], skill="", environment=replace(records[2].environment, agent_backend="lcs")
        ),
    ]
    assert "n/a" in provider_comparison_report(legacy)


def test_markdown_has_valid_separators_for_single_character_columns() -> None:
    text = format_table(["N"], [["1"]], ["r"], fmt="markdown")
    assert text.splitlines()[1] == "| ---: |"


def test_reader_skips_malformed_objects_and_keeps_unknown_fields_compatible(tmp_path: Path) -> None:
    r = asdict(record("a", "happy"))
    r["future_field"] = "ignored"
    path = tmp_path / "data.jsonl"
    path.write_text("[]\n" + json.dumps(r) + "\n")
    assert len(load_records(str(path))) == 1
