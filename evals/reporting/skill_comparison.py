"""Offline comparisons of explicitly selected, compatible Navigator attempts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from statistics import mean
from typing import Any

from evals.reporting.formatting import format_metric_score, format_table
from evals.reporting.models import EvalRecord
from evals.reporting.observations import AgentObservations, read_agent_observations
from evals.reporting.reader import record_from_dict

QUALITY_METRICS = {
    "tool_use": "Conversation MCP Use",
    "completion": "Conversation MCP Task Completion",
    "fitness": "Navigator Skill Fidelity [Conversational GEval]",
}
REPORT_EXPLANATIONS = {
    "quality": (
        "Quality summarizes how the agent performed. Tool use is the judge's score for MCP "
        "tool choices and arguments; Completion scores the requested outcome and deliverables; "
        "Fitness (fidelity) scores adherence to the Navigator skill. These are separate scores "
        "from 0 to 1, averaged over available measurements, not a combined quality score. "
        "Overall, Routing and Hard checks show passed/selected or passed/measured counts. "
        "N is the number of selected conversations; [measured/selected] shows score coverage."
    ),
    "efficiency": (
        "Efficiency shows average work and time per conversation: completed customer turns, "
        "model responses, output tokens, agent execution time, judge scoring time and their "
        "combined time. [measured/selected] shows how many conversations supplied each "
        "measurement; missing values are excluded from the mean."
    ),
    "usage_and_tools": (
        "Usage and tool figures are averages per conversation, not totals. Individual MCP "
        "call counts are integers, but their mean can be fractional: 5 calls in one conversation "
        "and 6 in another average 5.5 calls. The same applies to MCP errors and skill loads. "
        "[measured/selected] shows how many conversations contributed to each average."
    ),
}
PROFILE_KEYS = (
    "id",
    "model_name",
    "base_url",
    "api",
    "temperature",
    "sampling",
    "extra_settings",
    "agent_backend",
)


@dataclass
class Statistic:
    mean: float | None
    measured: int
    expected: int
    observed_total: float | None = None


@dataclass
class AttemptComparison:
    record: EvalRecord
    observations: AgentObservations
    routing_ok: bool | None
    hard_checks_ok: bool | None
    agent_seconds: float | None
    scoring_seconds: float | None
    total_seconds: float | None


@dataclass
class ModelComparison:
    model: str
    samples: int
    passed: int
    routing_passed: int
    routing_measured: int
    hard_checks_passed: int
    hard_checks_measured: int
    errors: dict[str, int]
    metrics: dict[str, Statistic]
    efficiency: dict[str, Statistic]
    profile: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillComparison:
    configuration: dict[str, Any]
    models: list[ModelComparison]
    attempts: list[AttemptComparison]
    excluded: dict[str, int]
    notes: list[str]
    execution_overlaps: list[dict[str, Any]] = field(default_factory=list)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _is_rescore(record: EvalRecord, observations: AgentObservations) -> bool | None:
    if record.is_rescore is not None:
        return record.is_rescore
    # Older records have no marker. A replay reports zero execution time, while
    # the original artifact can show that the agent actually ran for longer.
    elapsed = _number(observations.metadata.get("elapsed_s"))
    if record.agent_seconds == 0:
        return True if elapsed is not None and elapsed > 0 else None
    return False


def _stat(values: list[float | int | None]) -> Statistic:
    measured = [v for v in values if v is not None]
    return Statistic(
        mean(measured) if measured else None,
        len(measured),
        len(values),
        sum(measured) if measured else None,
    )


def _metric(record: EvalRecord, name: str) -> float | None:
    matches = [m for m in record.metrics if m.name == name]
    if len(matches) > 1:
        raise ValueError(f"Duplicate metric {name!r} in attempt {record.attempt_id}")
    value = _number(matches[0].score) if matches else None
    return value if value is not None and value <= 1 else None


def _check(record: EvalRecord, name: str) -> bool | None:
    found = next((c for c in record.checks if c.name == name), None)
    return found.status == "pass" if found and found.status in {"pass", "fail"} else None


def _single(values: set[Any], label: str, *, allow_empty: bool = False) -> Any:
    if len(values) != 1:
        raise ValueError(f"Incompatible {label}; select one cohort or a matching set of runs")
    value = next(iter(values))
    if value in (None, "unknown") or (value == "" and not allow_empty):
        raise ValueError(f"Missing {label}; comparison configuration cannot be established")
    return value


def _configuration(records: list[EvalRecord]) -> dict[str, Any]:
    """Missing metrics are allowed; conflicting versions/cutoffs are not."""
    result = {
        "skill": _single({r.skill for r in records}, "skill"),
        "skill_sha": _single({r.skill_sha for r in records}, "skill hash"),
        "source_commit": _single(
            {r.git.commit for r in records}, "source commit / host instructions"
        ),
        "rubric_version": _single({r.rubric_version for r in records}, "rubric version"),
        "judge_provider": _single({r.environment.eval_provider for r in records}, "judge provider"),
        "judge_model": _single({r.environment.eval_model for r in records}, "judge model"),
        "judge_endpoint": _single(
            {r.environment.eval_model_base_url for r in records}, "judge endpoint", allow_empty=True
        ),
        "cluster_mode": _single({r.environment.cluster_mode for r in records}, "cluster mode"),
        "agent_backend": _single({r.environment.agent_backend for r in records}, "agent backend"),
    }
    checks_versions = {r.checks_version for r in records}
    if checks_versions != {""}:
        result["checks_version"] = _single(checks_versions, "checks version")
    result["generation_commit"] = _single(
        {r.generation_git.get("commit") or r.git.commit for r in records}, "generation commit"
    )
    cutoffs = {
        (r.environment.mcp_use_threshold, r.environment.task_completion_threshold) for r in records
    }
    result["mcp_thresholds"] = _single(cutoffs, "MCP thresholds")
    scenarios: dict[str, str] = {}
    for golden in sorted({r.golden_id for r in records}):
        if not golden:
            raise ValueError("A skill comparison requires golden identities")
        scenarios[golden] = _single(
            {r.scenario_sha for r in records if r.golden_id == golden},
            f"scenario hash for {golden}",
        )
    result["scenario_sha"] = scenarios
    metric_config = {}
    for name in sorted({m.name for r in records for m in r.metrics}):
        configs = {(m.version, m.threshold) for r in records for m in r.metrics if m.name == name}
        version, threshold = _single(configs, f"metric version/threshold for {name}")
        if not version or _number(threshold) is None:
            raise ValueError(f"Missing metric version/threshold for {name}")
        metric_config[name] = {"version": version, "threshold": threshold}
    result["metrics"] = metric_config
    by_model: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_model[record.environment.agent_model].add(record.golden_id)
    if len({tuple(sorted(goldens)) for goldens in by_model.values()}) != 1:
        raise ValueError(
            "Models have different golden sets; filter to a common --scenario before comparing"
        )
    mixes = set()
    for model in by_model:
        group = [r for r in records if r.environment.agent_model == model]
        mixes.add(
            tuple(
                Fraction(sum(r.golden_id == golden for r in group), len(group))
                for golden in scenarios
            )
        )
    if len(mixes) != 1:
        raise ValueError(
            "Models have different golden sampling proportions; use balanced runs or a common --scenario"
        )
    return result


def load_cohort(path: Path) -> dict[str, Any]:
    """A frozen sweep report explicitly selects records and scheduling evidence."""
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("A cohort must be a JSON object containing a records list")
    if not isinstance(data.get("frozen_configuration"), dict):
        raise ValueError("A cohort must include its frozen_configuration")
    data["cohort_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def build_skill_comparison(
    records: list[EvalRecord],
    *,
    cohort: dict[str, Any] | None = None,
    run_ids: list[str] | None = None,
    scenario: str | None = None,
    last_n: int = 10,
    artifacts_root: Path | None = None,
    scoring: str = "original",
    concurrent: bool = False,
) -> SkillComparison:
    """Compare judged Navigator attempts; never use current skill files."""
    if not cohort and not run_ids:
        raise ValueError("Select skill records explicitly with --cohort or --run-id")
    if cohort and run_ids:
        raise ValueError("Choose --cohort or --run-id, not both")
    if last_n < 1:
        raise ValueError("--last must be positive")
    if scoring not in {"original", "latest"}:
        raise ValueError("scoring must be 'original' or 'latest'")
    excluded = {
        "dry_run": 0,
        "rescore": 0,
        "unverified_origin": 0,
        "duplicate": 0,
        "outside_selection": 0,
        "last_limit": 0,
    }
    source = [record_from_dict(r) for r in cohort["records"]] if cohort else records
    origins: dict[str, str] = {}
    initial_records: dict[str, EvalRecord] = {}
    for record in sorted(source, key=lambda r: r.timestamp):
        if record.attempt_id:
            origins.setdefault(record.attempt_id, record.run_id)
            initial_records.setdefault(record.attempt_id, record)
    selected: list[tuple[EvalRecord, AgentObservations]] = []
    known_runs = {r.run_id for r in source}
    seen: set[str] = set()
    selected_positions: dict[str, int] = {}
    for record in sorted(source, key=lambda r: r.timestamp):
        recorded_path = Path(record.artifact_path) if record.artifact_path else None
        path_run = (
            recorded_path.parent.name
            if recorded_path and recorded_path.name == record.attempt_id
            else ""
        )
        selection_run = path_run or origins.get(record.attempt_id, record.run_id)
        if (
            run_ids and (selection_run if scoring == "latest" else record.run_id) not in run_ids
        ) or (scenario and record.golden_id != scenario):
            excluded["outside_selection"] += 1
            continue
        if record.skill != "navigator":
            raise ValueError(
                "This comparison supports Navigator goldens only; deployment and LCS records must stay separate"
            )
        if record.dry_run:
            excluded["dry_run"] += 1
            continue
        if not record.attempt_id:
            raise ValueError("Skill comparisons require original attempt identities")
        obs = read_agent_observations(record, artifacts_root)
        origin = obs.artifact_run_id or path_run or origins[record.attempt_id]
        if scoring == "latest":
            known_runs.add(origin)
        if scoring == "latest" and run_ids and origin not in run_ids:
            excluded["outside_selection"] += 1
            continue
        if scoring == "original":
            replay = _is_rescore(record, obs)
            if record.run_id != origin or replay is True:
                excluded["rescore"] += 1
                continue
            if replay is None:
                excluded["unverified_origin"] += 1
                continue
        if record.attempt_id in seen:
            index = selected_positions[record.attempt_id]
            if scoring == "original" or selected[index][0].scoring_id == record.scoring_id:
                excluded["duplicate"] += 1
                continue
            excluded["rescore"] += 1
            selected[index] = (record, obs)
        else:
            seen.add(record.attempt_id)
            selected_positions[record.attempt_id] = len(selected)
            selected.append((record, obs))
        for key in ("skill_sha", "scenario_sha"):
            if key in obs.metadata and obs.metadata[key] != getattr(record, key):
                raise ValueError(
                    f"Attempt {record.attempt_id} {key} differs from its original artifact"
                )
        if cohort:
            detail: dict[str, Any] = next(
                (d for d in cohort.get("case_details", []) if d["attempt_id"] == record.attempt_id),
                {},
            )
            for name, expected_hash in detail.get("native_sha256", {}).items():
                actual = (
                    obs.artifact_sha256
                    if Path(name).name == "artifact.json"
                    else obs.messages_sha256
                    if Path(name).name == "messages.json"
                    else ""
                )
                if actual and actual != expected_hash:
                    raise ValueError(
                        f"Native artifact hash differs from frozen cohort: {record.attempt_id}/{Path(name).name}"
                    )
    if run_ids:
        missing = set(run_ids) - known_runs
        if missing:
            raise ValueError(f"Unknown run IDs: {', '.join(sorted(missing))}")
    recent: dict[tuple[str, str], list[tuple[EvalRecord, AgentObservations]]] = defaultdict(list)
    for pair in selected:
        recent[(pair[0].environment.agent_model, pair[0].golden_id)].append(pair)
    selected = []
    for recent_group in recent.values():
        excluded["last_limit"] += max(0, len(recent_group) - last_n)
        selected.extend(recent_group[-last_n:])
    if not selected:
        raise ValueError("No original judged Navigator attempts match this selection")
    configuration = _configuration([r for r, _ in selected])
    configuration["scoring_selection"] = scoring
    configuration["concurrent"] = concurrent
    configuration["run_ids"] = sorted(
        {obs.artifact_run_id or origins[r.attempt_id] for r, obs in selected}
    )
    configuration["scoring_run_ids"] = sorted({r.run_id for r, _ in selected})
    configuration["cohort_file_sha256"] = cohort.get("cohort_file_sha256") if cohort else None
    frozen = cohort.get("frozen_configuration", {}) if cohort else {}
    if frozen:
        if not frozen["source_commit"].startswith(configuration["source_commit"]):
            raise ValueError("Cohort source commit differs from its selected records")
        if frozen["skill_sha"].get("navigator") != configuration["skill_sha"]:
            raise ValueError("Cohort skill hash differs from its selected records")
        for golden, sha in configuration["scenario_sha"].items():
            if frozen["scenario_sha"].get(golden) != sha:
                raise ValueError(f"Cohort scenario hash differs for {golden}")
        judge = frozen.get("judge", {})
        for name, field_name in (
            ("provider", "judge_provider"),
            ("model", "judge_model"),
            ("base_url", "judge_endpoint"),
        ):
            if judge.get(name) != configuration[field_name]:
                raise ValueError(f"Cohort judge {name} differs from selected records")
        if frozen.get("versions", {}).get("fidelity") != configuration["rubric_version"]:
            raise ValueError("Cohort rubric version differs from selected records")
        thresholds = frozen.get("thresholds", {})
        if configuration["mcp_thresholds"] != (thresholds.get("use"), thresholds.get("completion")):
            raise ValueError("Cohort MCP thresholds differ from selected records")
        for key, name in QUALITY_METRICS.items():
            metric = configuration["metrics"].get(name)
            expected_version = frozen.get("versions", {}).get(
                "fidelity" if key == "fitness" else "mcp"
            )
            expected_threshold = thresholds.get(
                {"tool_use": "use", "completion": "completion", "fitness": "fidelity"}[key]
            )
            if metric and metric != {"version": expected_version, "threshold": expected_threshold}:
                raise ValueError(f"Cohort metric configuration differs for {name}")
    notes = [
        "Fitness means the existing skill-fidelity score; no composite ranking is calculated.",
        "Hard checks pass when all applicable measured checks pass. Conditional n/a or unexercised checks do not establish branch coverage.",
        "All averages are per selected conversation. Each value shows measured/selected coverage; missing or errored scores stay n/a, never zero. Partial coverage means are not directly comparable to full coverage.",
        "Turns are completed scripted user turns; model responses count saved responses, including tool-use steps.",
        "Token counts sum the retained per-response agent usage across the conversation, including repeated context. Output includes reasoning and tool-call tokens; reasoning is a subset, not an extra charge to add. Cached input is a subset of input. Failed requests without saved usage cannot be counted.",
        "Tokenizers and provider accounting differ. These are recorded token counts, not billed EnMaaS cost. SDK price estimates are not reported as billing.",
        "Agent time excludes judging. Total time = agent + scoring per conversation; it is not cohort elapsed time. Timing includes provider latency, mock MCP work, and startup.",
        "One attempt per cell is a smoke comparison, not a reliability estimate. Failures remain in quality and efficiency denominators.",
    ]
    if concurrent:
        notes.append(
            "Attempts ran concurrently; timing is descriptive, not a controlled latency comparison."
        )
    elif not cohort:
        notes.append(
            "Run-ID selection has no scheduling manifest; concurrency is unknown, so timing is descriptive."
        )
    if scoring == "latest":
        notes.append(
            "Latest scoring is selected per original attempt; earlier scorings are excluded. "
            "Agent time comes from the original record or artifact, not the rescore; total time "
            "excludes earlier scoring attempts."
        )
    overlaps = cohort.get("execution_overlaps", []) if cohort else []
    if overlaps:
        notes.append(
            "Batches overlapped: "
            + "; ".join(f"{' / '.join(o['agents'])}: {o['seconds']:.2f}s" for o in overlaps)
            + ". Concurrent load prevents a controlled latency ranking."
        )
    rows: list[AttemptComparison] = []
    for record, obs in selected:
        original = initial_records[record.attempt_id]
        agent_time = (
            _number(original.agent_seconds) if _is_rescore(original, obs) is False else None
        )
        if agent_time is None:
            agent_time = _number(obs.metadata.get("elapsed_s"))
        scoring_time = _number(record.duration_seconds)
        routing = _check(record, "routing_loaded_expected_skill")
        if routing is None and (record.skills_loaded or record.outcome == "routing_failure"):
            routing = "navigator" in record.skills_loaded
        known_checks = [c for c in record.checks if c.status in {"pass", "fail"}]
        rows.append(
            AttemptComparison(
                record,
                obs,
                routing,
                all(c.status == "pass" for c in known_checks) if known_checks else None,
                agent_time,
                scoring_time,
                agent_time + scoring_time
                if agent_time is not None and scoring_time is not None
                else None,
            )
        )
    groups: dict[str, list[AttemptComparison]] = defaultdict(list)
    for row in rows:
        groups[row.record.environment.agent_model].append(row)
    summaries = []
    order = list(dict.fromkeys([*frozen.get("batch_order", []), *sorted(groups)]))
    for model in order:
        if model not in groups:
            continue
        group = groups[model]
        profiles = [
            {
                k: row.observations.metadata[k]
                for k in PROFILE_KEYS
                if k in row.observations.metadata
            }
            for row in group
        ]
        known_profiles = {json.dumps(p, sort_keys=True) for p in profiles if p}
        if len(known_profiles) > 1:
            raise ValueError(f"Agent API/settings changed within model {model}")
        profile = json.loads(next(iter(known_profiles))) if known_profiles else {}
        expected: dict[str, Any] = next(
            (a for a in frozen.get("agents", []) if a["id"] == model), {}
        )
        if frozen.get("agents") and not expected:
            raise ValueError(f"Agent {model} is absent from the frozen cohort roster")
        if expected and profile and any(profile.get(k) != expected.get(k) for k in PROFILE_KEYS):
            raise ValueError(f"Agent profile for {model} differs from frozen cohort")
        if expected and not profile:
            profile = {k: expected[k] for k in PROFILE_KEYS if k in expected}
            notes.append(
                f"{model}: profile comes from frozen cohort; local artifact metadata is unavailable."
            )
        efficiency = {
            name: _stat([getattr(row.observations, name) for row in group])
            for name in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "model_responses",
                "mcp_calls",
                "mcp_errors",
                "unreturned_mcp_calls",
                "capability_loads",
            )
        }
        efficiency.update(
            {
                name: _stat([getattr(row, name) for row in group])
                for name in ("agent_seconds", "scoring_seconds", "total_seconds")
            }
        )
        efficiency["turns"] = _stat([row.record.turns for row in group])
        summaries.append(
            ModelComparison(
                model=model,
                samples=len(group),
                passed=sum(row.record.passed for row in group),
                routing_passed=sum(row.routing_ok is True for row in group),
                routing_measured=sum(row.routing_ok is not None for row in group),
                hard_checks_passed=sum(row.hard_checks_ok is True for row in group),
                hard_checks_measured=sum(row.hard_checks_ok is not None for row in group),
                errors={
                    "harness": sum(row.record.outcome == "harness_error" for row in group),
                    "model": sum(row.record.outcome == "model_failure" for row in group),
                    "routing": sum(row.record.outcome == "routing_failure" for row in group),
                    "judge_metrics": sum(
                        _number(m.score) is None or m.score > 1
                        for row in group
                        for m in row.record.metrics
                    ),
                },
                metrics={
                    key: _stat([_metric(row.record, name) for row in group])
                    for key, name in QUALITY_METRICS.items()
                },
                efficiency=efficiency,
                profile=profile,
            )
        )
    for row in rows:
        notes.extend(f"{row.record.attempt_id}: {warning}" for warning in row.observations.warnings)
    return SkillComparison(
        configuration, summaries, rows, excluded, list(dict.fromkeys(notes)), overlaps
    )


def render_skill_comparison(report: SkillComparison, fmt: str = "terminal") -> str:
    if fmt == "json":
        # Preserve invalid metric slots as null in JSON, as n/a in tables.
        payload = json.loads(json.dumps(asdict(report)), parse_constant=lambda _value: None)
        payload["explanations"] = REPORT_EXPLANATIONS
        return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)

    def value(stat: Statistic, digits: int = 2) -> str:
        number = f"{stat.mean:,.{digits}f}" if stat.mean is not None else "n/a"
        return f"{number} [{stat.measured}/{stat.expected}]"

    def table(headers: list[str], rows: list[list[str]]) -> str:
        return format_table(headers, rows, ["l"] + ["r"] * (len(headers) - 1), fmt=fmt)

    quality = [
        [
            m.model,
            str(m.samples),
            f"{m.passed}/{m.samples}",
            f"{m.routing_passed}/{m.routing_measured}",
            f"{m.hard_checks_passed}/{m.hard_checks_measured}",
            *[value(m.metrics[k], 3) for k in QUALITY_METRICS],
        ]
        for m in report.models
    ]
    efficiency = [
        [
            m.model,
            *[
                value(m.efficiency[k], digits)
                for k, digits in (
                    ("turns", 1),
                    ("model_responses", 1),
                    ("output_tokens", 1),
                    ("agent_seconds", 2),
                    ("scoring_seconds", 2),
                    ("total_seconds", 2),
                )
            ],
        ]
        for m in report.models
    ]
    usage = [
        [
            m.model,
            *[
                value(m.efficiency[k], 1)
                for k in (
                    "input_tokens",
                    "reasoning_tokens",
                    "cache_read_tokens",
                    "mcp_calls",
                    "mcp_errors",
                    "capability_loads",
                )
            ],
        ]
        for m in report.models
    ]
    diagnostics = [
        [m.model, *[str(m.errors[k]) for k in ("harness", "model", "routing", "judge_metrics")]]
        for m in report.models
    ]
    detail = []
    for a in sorted(
        report.attempts,
        key=lambda a: (a.record.environment.agent_model, a.record.golden_id, a.record.timestamp),
    ):
        r = a.record
        scores = [
            format_metric_score(next((m for m in r.metrics if m.name == name), None))
            for name in QUALITY_METRICS.values()
        ]
        detail.append(
            [
                r.environment.agent_model,
                r.golden_id,
                r.attempt_id,
                "Pass" if r.passed else "Fail",
                *scores,
                str(r.turns),
                str(a.observations.output_tokens)
                if a.observations.output_tokens is not None
                else "n/a",
                f"{a.total_seconds:.2f}" if a.total_seconds is not None else "n/a",
            ]
        )
    prefix = "## " if fmt == "markdown" else ""
    config = report.configuration
    cutoffs = "; ".join(
        f"{key}: {metric['threshold']:.2f} ({metric['version']})"
        for key, name in QUALITY_METRICS.items()
        if (metric := config["metrics"].get(name))
    )
    sections = [
        f"{prefix}Navigator model comparison",
        "",
        f"Skill {config['skill_sha']} | source {config['source_commit']} | judge {config['judge_provider']}/{config['judge_model']} | rubric {config['rubric_version']}",
        "",
        f"{len(report.attempts)} conversations; {sum(m.passed for m in report.models)}/{len(report.attempts)} overall passes. Cutoffs: {cutoffs or 'unavailable'}.",
        "",
        *(
            [
                "Timing is descriptive: overlapping batches prevent a controlled latency ranking.",
                "",
            ]
            if report.execution_overlaps
            else []
        ),
        "Quality — means [measured/selected]",
        REPORT_EXPLANATIONS["quality"],
        "",
        table(
            [
                "Model",
                "N",
                "Overall",
                "Routing",
                "Hard checks",
                "Tool use",
                "Completion",
                "Fitness (fidelity)",
            ],
            quality,
        ),
        "",
        "Efficiency — means per conversation [measured/selected]",
        REPORT_EXPLANATIONS["efficiency"],
        "",
        table(
            [
                "Model",
                "Turns",
                "Model responses",
                "Output tokens",
                "Agent s",
                "Scoring s",
                "Total s",
            ],
            efficiency,
        ),
        "",
        "Usage and tools — means per conversation [measured/selected]",
        REPORT_EXPLANATIONS["usage_and_tools"],
        "",
        table(
            [
                "Model",
                "Input tokens",
                "Reasoning tokens",
                "Cached input",
                "MCP calls",
                "MCP errors",
                "Skill loads",
            ],
            usage,
        ),
        "",
        "Errors — counts",
        table(
            ["Model", "Harness", "Model failures", "Routing failures", "Errored metrics"],
            diagnostics,
        ),
        "",
        "Per attempt",
        table(
            [
                "Model",
                "Golden",
                "Attempt",
                "Overall",
                "Tool use",
                "Completion",
                "Fitness",
                "Turns",
                "Output tokens",
                "Total s",
            ],
            detail,
        ),
        "",
        "Failures and model settings",
    ]
    for m in report.models:
        sections.append(
            f"- {m.model}: {json.dumps(m.profile, sort_keys=True)}"
            if m.profile
            else f"- {m.model}: agent settings unavailable."
        )
    for a in report.attempts:
        r = a.record
        failures = [f"{c.name}: {c.detail}" for c in r.checks if c.status == "fail"]
        failures += [f"{m.name}: {m.reason}" for m in r.metrics if not m.success]
        if r.error_detail:
            failures.append(f"{r.error_stage}: {r.error_detail}")
        sections.extend(f"- {r.attempt_id}: {failure}" for failure in failures)
    sections.extend(
        ["", "Excluded records: " + json.dumps(report.excluded, sort_keys=True), "", "Notes", ""]
    )
    sections.extend(f"- {note}" for note in report.notes)
    if fmt == "markdown":
        # Give each report section a heading and a blank line before its table/list.
        titles = {
            "Quality — means [measured/selected]",
            "Efficiency — means per conversation [measured/selected]",
            "Usage and tools — means per conversation [measured/selected]",
            "Errors — counts",
            "Per attempt",
            "Failures and model settings",
            "Notes",
        }
        sections = [
            part
            for section in sections
            for part in (["### " + section, ""] if section in titles else [section])
        ]
    return "\n".join(sections)
