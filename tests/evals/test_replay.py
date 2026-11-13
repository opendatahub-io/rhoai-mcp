"""The saved execution is the sole input to every scoring pass."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from evals.agent.transcript import AttemptArtifact
from evals.reporting.models import MetricRecord
from evals.reporting.reader import load_records
from evals.reporting.recorder import EvalRecorder
from evals.skills.attempt import score_artifact
from evals.skills.config import SkillEvalConfig
from evals.skills.goldens import NAVIGATOR_HAPPY_PATH
from evals.skills.rescore import main, rescore

from tests.evals.test_skill_agreement import workflow


def artifact(root: Path, *, snapshot: bool = True) -> Path:
    golden = NAVIGATOR_HAPPY_PATH
    metadata = {
        "golden_id": golden.id,
        "skill": golden.skill,
        "scenario_sha": golden.scenario_sha(),
        "skill_sha": "original-skill",
        "generation_git": {"commit": "original", "branch": "old"},
        "outcome": "evaluated",
        "turns_completed": len(golden.script),
        "id": "test-agent",
    }
    if snapshot:
        metadata["scenario_snapshot"] = golden.snapshot()
    return AttemptArtifact.from_messages(
        workflow(), attempt_id="original-attempt", run_id="original-run", metadata=metadata
    ).save(root)


def test_replay_preserves_evidence_identity_and_saved_intent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = artifact(tmp_path)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    config = SkillEvalConfig(_env_file=None)
    recorder = EvalRecorder(config, path=tmp_path / "history.jsonl")
    changed = replace(NAVIGATOR_HAPPY_PATH, description="Different future scenario")
    with patch("evals.agent.runner.run_conversation", side_effect=AssertionError("agent called")):
        first = score_artifact(path, NAVIGATOR_HAPPY_PATH, config=config, recorder=recorder)
        second = score_artifact(path, changed, config=config, recorder=recorder)
    records = load_records(str(recorder.path))
    assert len(records) == 2
    assert first.report.to_list() == second.report.to_list()
    assert records[0].run_id == records[1].run_id == "original-run"
    assert records[0].attempt_id == records[1].attempt_id == "original-attempt"
    assert records[0].scoring_id != records[1].scoring_id
    assert records[1].skill_sha == "original-skill"
    assert records[1].scenario_sha == NAVIGATOR_HAPPY_PATH.scenario_sha()
    assert records[1].generation_git["commit"] == "original"
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}
    assert "CHECKS ONLY" in capsys.readouterr().out
    assert not first.passed and not second.passed


def test_legacy_artifact_requires_matching_hash(tmp_path: Path) -> None:
    path = artifact(tmp_path, snapshot=False)
    config = SkillEvalConfig(_env_file=None)
    recorder = EvalRecorder(config, path=tmp_path / "history.jsonl")
    score_artifact(path, NAVIGATOR_HAPPY_PATH, config=config, recorder=recorder)
    assert load_records(str(recorder.path))[0].environment.cluster_mode == "unknown"
    with pytest.raises(ValueError, match="incompatible artifact"):
        score_artifact(
            path,
            replace(NAVIGATOR_HAPPY_PATH, description="changed"),
            config=config,
            recorder=recorder,
        )


def test_rescore_uses_snapshot_when_scenario_is_unregistered(tmp_path: Path) -> None:
    path = artifact(tmp_path)
    config = SkillEvalConfig(_env_file=None)
    recorder = EvalRecorder(config, path=tmp_path / "history.jsonl")
    with patch("evals.skills.rescore.GOLDENS_BY_ID", {}):
        outcomes = rescore([path], config=config, recorder=recorder)
    assert len(outcomes) == 1
    assert len(load_records(str(recorder.path))) == 1


def test_rescore_legacy_artifact_uses_registry(tmp_path: Path) -> None:
    path = artifact(tmp_path, snapshot=False)
    config = SkillEvalConfig(_env_file=None)
    recorder = EvalRecorder(config, path=tmp_path / "history.jsonl")
    assert len(rescore([path], config=config, recorder=recorder)) == 1
    with (
        patch("evals.skills.rescore.GOLDENS_BY_ID", {}),
        pytest.raises(ValueError, match="unknown scenario"),
    ):
        rescore([path], config=config, recorder=recorder)


def test_missing_required_metric_and_incomplete_execution_cannot_pass(tmp_path: Path) -> None:
    path = artifact(tmp_path)
    config = SkillEvalConfig(_env_file=None)
    recorder = EvalRecorder(config, path=tmp_path / "history.jsonl")
    metric = MetricRecord(
        name="Conversation MCP Use", score=1, success=True, threshold=0.6, reason=""
    )
    with patch("evals.skills.attempt._score", return_value=([metric], 0)):
        result = score_artifact(
            path, NAVIGATOR_HAPPY_PATH, config=config, recorder=recorder, judge=object()
        )
    assert not result.passed

    all_metrics = [
        metric,
        MetricRecord(
            name="Conversation MCP Task Completion",
            score=1,
            success=True,
            threshold=0.75,
            reason="",
        ),
        MetricRecord(
            name="Navigator Skill Fidelity [Conversational GEval]",
            score=1,
            success=True,
            threshold=0.8,
            reason="",
        ),
    ]
    with patch("evals.skills.attempt._score", return_value=(all_metrics, 0)):
        result = score_artifact(
            path, NAVIGATOR_HAPPY_PATH, config=config, recorder=recorder, judge=object()
        )
    assert result.passed
    with patch("evals.skills.attempt._score", return_value=(all_metrics, 0)):
        result = score_artifact(
            path,
            NAVIGATOR_HAPPY_PATH,
            config=config,
            recorder=recorder,
            judge=object(),
            run_error="checkpoint failed",
        )
    assert not result.passed


def test_replay_cli_requires_selector_and_reports_no_match(tmp_path: Path) -> None:
    assert main(["--results-root", str(tmp_path), "--attempt", "missing"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster_mode", ["mock", "live"])
async def test_live_scoring_and_replay_share_the_saved_record_path(
    tmp_path: Path, cluster_mode: str
) -> None:
    from unittest.mock import AsyncMock

    from evals.agent.runner import ConversationResult
    from evals.skills.attempt import run_attempt

    class Spec:
        id = "offline-agent"

        def describe(self) -> dict[str, str]:
            return {"id": self.id, "model_name": "offline"}

    golden = NAVIGATOR_HAPPY_PATH
    config = SkillEvalConfig(_env_file=None, cluster_mode=cluster_mode)
    recorder = EvalRecorder(config, path=tmp_path / "history.jsonl")

    async def fake_run(
        _spec: object, _script: object, *, run_id: str, attempt_id: str, mock_cluster: bool
    ) -> ConversationResult:
        assert mock_cluster is (cluster_mode == "mock")
        return ConversationResult(
            run_id=run_id,
            attempt_id=attempt_id,
            spec_id="offline-agent",
            messages=workflow(),
            turns_completed=len(golden.script),
            task="support chatbot",
        )

    with patch(
        "evals.skills.attempt.run_conversation", new=AsyncMock(side_effect=fake_run)
    ) as agent:
        first = await run_attempt(
            golden, Spec(), recorder=recorder, config=config, results_root=tmp_path
        )
        replay = score_artifact(
            first.artifact_path,
            golden,
            config=SkillEvalConfig(
                _env_file=None, cluster_mode="live" if cluster_mode == "mock" else "mock"
            ),
            recorder=recorder,
        )
        assert agent.await_count == 1
    records = load_records(str(recorder.path))
    assert first.report.to_list() == replay.report.to_list()
    assert records[0].run_id == records[1].run_id == recorder.run_id
    assert records[0].attempt_id == records[1].attempt_id
    assert records[0].scoring_id != records[1].scoring_id
    assert records[0].checks == records[1].checks
    assert AttemptArtifact.load(first.artifact_path).metadata["cluster_mode"] == cluster_mode
    assert [record.environment.cluster_mode for record in records] == [cluster_mode, cluster_mode]
