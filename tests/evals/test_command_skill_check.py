"""Synthetic workflow traces and malformed-output tests; no Claude or judge calls."""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from evals import command_skill_check as check

_IDS = itertools.count()
DESCRIPTION = "Guide the customer through a navigator workflow"
TOOLS = ["mcp__rhoai-mcp__recommend_model", "mcp__rhoai-mcp__get_expected_rps"]
DEFAULTS = {
    "use_case": check.USE_CASE,
    "description": "Support chatbot",
    "slo_targets": {
        "ttft_ms": {"default": 300},
        "itl_ms": {"default": 30},
        "e2e_ms": {"default": 8620},
    },
    "workload": {"prompt_tokens": 512, "output_tokens": 256},
}
TRAFFIC = {"use_case": check.USE_CASE, "user_count": 50, "expected_rps": 0.07, "peak_rps": 0.13}
INVENTORY = {"gpu_info": {"products": ["NVIDIA-A100-SXM4-80GB"]}}


def call(tool_name: str, payload: Any, **arguments: Any) -> dict[str, Any]:
    return {
        "name": tool_name,
        "arguments": arguments,
        "result": [{"type": "text", "text": json.dumps(payload)}],
        "call_id": f"tool-{next(_IDS)}",
        "is_error": False,
    }


def recommendations(model_id: str = "example/model-a") -> dict[str, Any]:
    return {
        "recommendations": {
            f"top_{category}": {
                "model_id": model_id if category == "balanced" else f"example/model-{category}",
                "model": f"Model {category.title()}",
                "gpu": "1x A100-80",
            }
            for category in check.CATEGORIES
        }
    }


def comparison_table(payload: dict[str, Any] | None = None) -> str:
    payload = payload or recommendations()
    lines = ["| | Balanced | Cost | Performance | Quality |", "|---|---|---|---|---|"]
    for label in check.TABLE_ROWS:
        values = (
            [
                payload["recommendations"][f"top_{category}"]["model_id"]
                for category in check.CATEGORIES
            ]
            if label == "Model"
            else ["Available ✓" if label == "Cluster fit" else "42"] * 4
        )
        lines.append("| " + " | ".join([label, *values]) + " |")
    return "\n".join(lines)


def handoff_values(model_id: str = "example/model-a") -> dict[str, Any]:
    return {
        "model_id": model_id,
        "use_case": check.USE_CASE,
        "user_count": 50,
        "chosen_category": "balanced",
        "prompt_tokens": 512,
        "output_tokens": 256,
        "expected_qps": 0.07,
        "ttft_target_ms": 300,
        "itl_target_ms": 30,
        "e2e_target_ms": 8620,
        "cluster_gpu_types": ["A100-80"],
    }


def handoff_block(values: dict[str, Any] | None = None) -> str:
    values = values or handoff_values()
    return (
        "<!-- navigator-handoff\n"
        + "\n".join(
            f"{key}: {', '.join(value) if isinstance(value, list) else value}"
            for key, value in values.items()
        )
        + "\n-->"
    )


@dataclass
class FakeResult:
    task: str = ""
    final_output: str = "Please confirm the next phase."
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 1
    conversation_id: str = ""


class SyntheticFailure(RuntimeError):
    def __init__(self, result: FakeResult) -> None:
        super().__init__("synthetic SDK execution failure")
        self.result = result


def tool_messages(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for tool in calls:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool["call_id"],
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "arguments": json.dumps(tool["arguments"]),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tool["call_id"], "content": tool["result"]},
            ]
        )
    return messages


class FakeClient:
    def __init__(self, responses: list[Any], session_id: str) -> None:
        self.responses = responses
        self.session_id = session_id
        self.last_result: FakeResult | None = None
        self.initialization = {"commands": [{"name": name} for name in check.SKILLS]}
        self.diagnostics: list[dict[str, Any]] = []
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        self.closed = True

    async def query(self, task: str, *, skill: str) -> FakeResult:
        return self._respond(f"/{skill} {task}")

    async def follow_up(self, message: str, conversation_id: str) -> FakeResult:
        assert conversation_id == self.session_id
        return self._respond(message)

    def _respond(self, prompt: str) -> FakeResult:
        self.sent.append(prompt)
        response = self.responses.pop(0)
        self.last_result = response.result if isinstance(response, SyntheticFailure) else response
        self.last_result.task = prompt
        self.last_result.conversation_id = self.session_id
        self.last_result.messages = [
            {"role": "user", "content": prompt},
            *tool_messages(self.last_result.tool_calls),
            {"role": "assistant", "content": self.last_result.final_output},
        ]
        self.diagnostics = [{"type": "synthetic", "exchange": len(self.sent)}]
        if isinstance(response, SyntheticFailure):
            raise response
        return response


class FakeFactory:
    def __init__(self, responses: list[list[Any]]) -> None:
        self.responses = responses
        self.clients: list[FakeClient] = []

    def __call__(self, **_kwargs: Any) -> FakeClient:
        client = FakeClient(self.responses[len(self.clients)], f"scenario-{len(self.clients)}")
        self.clients.append(client)
        return client


def scripted_responses(
    *, ready: bool = False, storage: bool = True, model_id: str = "example/model-a"
) -> list[list[FakeResult]]:
    payload = recommendations(model_id)
    handoff = handoff_values(model_id)
    plan_arguments = {
        key: handoff[key]
        for key in check.HANDOFF_KEYS
        if key not in {"model_id", "chosen_category", "cluster_gpu_types"}
    }
    plan_arguments.update(
        category="balanced", namespace=check.NAMESPACE, preferred_gpu_types=["A100-80"]
    )
    plan = {
        "ready": True,
        "issues": None,
        "model_id": model_id,
        "namespace": check.NAMESPACE,
        "runtime": "vllm-runtime",
        "storage_uri": None if storage else "oci://registry.example.com/resolved:latest",
        "gpu_count": 1,
        "gpu_type": "A100-80",
        "tensor_parallel": 1,
        "replicas": 1,
        "runtime_needs_creation": False,
    }
    execute_arguments = {
        key: plan[key]
        for key in (
            "model_id",
            "namespace",
            "runtime",
            "gpu_count",
            "gpu_type",
            "tensor_parallel",
            "replicas",
        )
    }
    execute_arguments["storage_uri"] = plan["storage_uri"] or check.STORAGE_URI
    state = "Ready" if ready else "Pending"
    status_calls = [
        call(
            "get_inference_service",
            {"name": "new-model", "namespace": check.NAMESPACE, "status": state},
            name="new-model",
            namespace=check.NAMESPACE,
        )
    ]
    if ready:
        status_calls.append(
            call(
                "get_model_endpoint",
                {"name": "new-model", "status": "Ready", "url": "https://model.example.com"},
                name="new-model",
                namespace=check.NAMESPACE,
            )
        )
    deployment = [FakeResult(tool_calls=[call("plan_deployment", plan, **plan_arguments)])]
    if storage:
        deployment.append(FakeResult(final_output="Storage resolved. Type deploy to proceed."))
    deployment.extend(
        [
            FakeResult(
                tool_calls=[
                    call(
                        "execute_deployment",
                        {
                            "deployed": True,
                            "name": "new-model",
                            "namespace": check.NAMESPACE,
                            "status": state,
                        },
                        **execute_arguments,
                    )
                ]
            ),
            FakeResult(tool_calls=status_calls),
        ]
    )
    return [
        [
            FakeResult(),
            FakeResult(
                tool_calls=[
                    call("get_use_case_defaults", DEFAULTS, use_case=check.USE_CASE),
                    call("get_expected_rps", TRAFFIC, use_case=check.USE_CASE, user_count=50),
                ]
            ),
            FakeResult(
                final_output=comparison_table(payload),
                tool_calls=[
                    call("get_cluster_resources", INVENTORY),
                    call(
                        "recommend_model",
                        payload,
                        text="support chatbot",
                        use_case=check.USE_CASE,
                        user_count=50,
                        preferred_gpu_types=["A100-80"],
                    ),
                ],
            ),
            FakeResult(final_output=handoff_block(handoff)),
        ],
        deployment,
    ]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    commands = tmp_path / ".claude/commands"
    commands.mkdir(parents=True)
    for name in check.SKILLS:
        (commands / f"{name}.md").write_text(
            f"---\ndescription: {DESCRIPTION}\nallowed-tools: {', '.join(TOOLS)}\n---\n\nBody for {name}.\n",
            encoding="utf-8",
        )
    (commands / "scaffold-plugin.md").write_text(
        "# Scaffold\nUnchanged instructions.\n", encoding="utf-8"
    )
    return tmp_path


def migrate(repo: Path) -> None:
    for name in check.SKILLS:
        command = repo / ".claude/commands" / f"{name}.md"
        body = command.read_bytes().split(b"---\n", 2)[2]
        skill = repo / ".claude/skills" / name / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_bytes(
            f"---\nname: {name}\ndescription: {DESCRIPTION}\nallowed-tools: {' '.join(TOOLS)}\n---\n".encode()
            + body
        )
        command.unlink()


def test_valid_table_among_other_tables() -> None:
    output = "| Workload | Value |\n|---|---|\n| Users | 50 |\n\n" + comparison_table()
    table = check.validate_comparison_table(output)
    assert len(table) == 8 and all(len(row) == 5 for row in table)
    assert check.validate_comparison_table(comparison_table().replace("42", "—"))
    assert check.validate_comparison_table(
        comparison_table().replace("example/model-a", "[Model A](https://example.com/a)")
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("| Quality |", "| Other |", 1),
        lambda text: text.replace(
            "| | Balanced | Cost | Performance | Quality |",
            "| Balanced | Cost | Performance | Quality |",
        ),
        lambda text: text.replace("|---|---|---|---|---|", "|---|---|---|---|"),
        lambda text: text.replace("| 42 | 42 | 42 | 42 |", "| 42 | 42 | 42 |", 1),
        lambda text: text.replace("| 42 | 42 | 42 | 42 |", "| 42 | 42 | 42 | 42 | 42 |", 1),
        lambda text: text.replace("| TTFT p95", "| Latency"),
        lambda text: text + "\n| Extra | 1 | 2 | 3 | 4 |",
        lambda text: "\n".join(line for line in text.splitlines() if not line.startswith("| GPU")),
        lambda text: text.replace("| 42 |", "| |", 1),
        lambda text: text.replace("42", "[value]", 1),
        lambda text: text.replace("42", "<value>", 1),
        lambda text: text.replace("42", "TBD", 1),
        lambda text: text.replace("42", "...", 1),
        lambda text: "```markdown\n" + text + "\n```",
        lambda text: text + "\n\n" + text,
    ],
)
def test_malformed_tables_fail(change: Any) -> None:
    with pytest.raises(AssertionError):
        check.validate_comparison_table(change(comparison_table()))


def test_parse_handoff_and_validate_workload() -> None:
    block, values = check.parse_handoff(handoff_block())
    assert block == handoff_block()
    assert values == handoff_values()
    check.validate_handoff(
        values,
        category="balanced",
        recommendation={"model_id": "example/model-a"},
        defaults=DEFAULTS,
        traffic=TRAFFIC,
        gpus=["A100-80"],
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda _text: "No handoff",
        lambda text: text + text,
        lambda text: text.replace("user_count: 50\n", ""),
        lambda text: text.replace("user_count: 50", "user_count: 1.5"),
        lambda text: text.replace("user_count: 50", "user_count: NaN"),
        lambda text: text.replace("expected_qps: 0.07", "expected_qps: inf"),
        lambda text: text.replace("user_count: 50", "user_count: 50\nuser_count: 50"),
        lambda text: text.replace("chosen_category: balanced", "chosen_category: cheapest"),
    ],
)
def test_malformed_handoffs_fail(change: Any) -> None:
    with pytest.raises(AssertionError):
        check.parse_handoff(change(handoff_block()))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_id", "invented"),
        ("user_count", 51),
        ("prompt_tokens", 1024),
        ("expected_qps", 0.13),
        ("ttft_target_ms", 301),
        ("cluster_gpu_types", ["H100"]),
    ],
)
def test_handoff_values_must_match_observed_results(key: str, value: Any) -> None:
    values = handoff_values() | {key: value}
    with pytest.raises(AssertionError):
        check.validate_handoff(
            values,
            category="balanced",
            recommendation={"model_id": "example/model-a"},
            defaults=DEFAULTS,
            traffic=TRAFFIC,
            gpus=["A100-80"],
        )


def test_static_metadata_and_byte_preservation(repo: Path) -> None:
    before = check.validate_sources(repo, "commands")
    migrate(repo)
    after = check.validate_sources(repo, "skills")
    for name in check.SKILLS:
        for key in ("body_sha256", "description", "tools"):
            assert before[name][key] == after[name][key]
        assert after[name]["name"] == name
    assert before["scaffold_sha256"] == after["scaffold_sha256"]
    path = repo / ".claude/skills/navigator/SKILL.md"
    path.write_bytes(path.read_bytes() + b"\r\n")
    assert (
        check.validate_sources(repo, "skills")["navigator"]["body_sha256"]
        != before["navigator"]["body_sha256"]
    )


@pytest.mark.parametrize(
    "fault", ["missing-command", "coexisting-skill", "missing-scaffold", "scaffold-skill"]
)
def test_static_layout_failures(repo: Path, fault: str) -> None:
    if fault == "missing-command":
        (repo / ".claude/commands/navigator.md").unlink()
    elif fault == "coexisting-skill":
        path = repo / ".claude/skills/navigator/SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text("replacement")
    elif fault == "missing-scaffold":
        (repo / ".claude/commands/scaffold-plugin.md").unlink()
    else:
        (repo / ".claude/skills/scaffold-plugin").mkdir(parents=True)
    with pytest.raises(AssertionError):
        check.validate_sources(repo, "commands")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("name: navigator\n", "name: wrong\n"),
        ("name: navigator\n", "name: navigator\nname: navigator\n"),
        (f"description: {DESCRIPTION}", "description: ''"),
        (f"description: {DESCRIPTION}", "description: [broken"),
        (" ".join(TOOLS), ", ".join(TOOLS)),
    ],
)
def test_invalid_skill_frontmatter(repo: Path, old: str, new: str) -> None:
    migrate(repo)
    path = repo / ".claude/skills/navigator/SKILL.md"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises((AssertionError, check.yaml.YAMLError)):
        check.validate_sources(repo, "skills")


@pytest.mark.parametrize(
    "result",
    [
        {"answer": 42},
        '{"answer":42}',
        [{"type": "text", "text": '{"answer":42}'}],
        {"structuredContent": {"answer": 42}},
        {"content": [{"type": "text", "text": '{"answer":42}'}]},
    ],
)
def test_tool_content_decoding(result: Any) -> None:
    assert check.tool_payload({"name": "test", "result": result}) == {"answer": 42}


@pytest.mark.parametrize("result", [{"error": "unavailable"}, {"isError": True}, None, "not JSON"])
def test_tool_errors_are_not_accepted(result: Any) -> None:
    with pytest.raises((AssertionError, ValueError)):
        check.tool_payload({"name": "test", "result": result})


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("storage", [False, True])
async def test_complete_synthetic_workflow(repo: Path, ready: bool, storage: bool) -> None:
    factory = FakeFactory(scripted_responses(ready=ready, storage=storage))
    output = repo / "results/explicit.json"
    artifact = await check.run_checks(
        phase="commands", output=output, repo_root=repo, client_factory=factory
    )
    assert artifact["status"] == "passed", artifact["failure"]
    assert all(outcome["status"] == "passed" for outcome in artifact["checks"].values())
    assert json.loads(output.read_text()) == artifact
    assert len(factory.clients) == 2 and all(client.closed for client in factory.clients)
    assert len(factory.clients[0].sent) == 4
    assert len(factory.clients[1].sent) == (4 if storage else 3)
    assert factory.clients[1].sent[0].startswith("/navigator-deploy")
    assert factory.clients[1].sent[-2] == "deploy"
    assert (
        "oci://registry.example.com/mock/model:latest" in factory.clients[1].sent[1]
        if storage
        else True
    )
    nav = artifact["scenarios"]["navigator"]["exchanges"]
    assert all(exchange["diagnostics"] for exchange in nav)
    assert nav[1]["tool_calls"][1]["arguments"]["user_count"] == 50
    assert nav[1]["tool_calls"][1]["result"]
    assert (
        len(
            [
                message
                for exchange in nav
                for message in exchange["messages"]
                if message["role"] == "user"
            ]
        )
        == 4
    )


@pytest.mark.parametrize("gate", ["workload", "planning", "storage"])
async def test_confirmation_gates_reject_early_actions(repo: Path, gate: str) -> None:
    responses = scripted_responses()
    if gate == "workload":
        responses[0][0].tool_calls.append(responses[0][2].tool_calls[-1])
        failure_name = "navigator.workload_gate"
    else:
        index = 0 if gate == "planning" else 1
        responses[1][index].tool_calls.append(responses[1][2].tool_calls[0])
        failure_name = "deployment.approval_gate"
    factory = FakeFactory(responses)
    artifact = await check.run_checks(
        phase="commands", output=repo / "failed.json", repo_root=repo, client_factory=factory
    )
    assert artifact["status"] == "failed"
    assert artifact["checks"][failure_name]["status"] == "failed"
    assert all(client.closed for client in factory.clients)
    assert len(factory.clients) == (1 if gate == "workload" else 2)


@pytest.mark.parametrize(
    ("stage", "key", "value", "check_name"),
    [
        ("recommend", "preferred_gpu_types", ["H100"], "navigator.recommendations"),
        ("recommend", "user_count", 999, "navigator.recommendations"),
        ("plan", "expected_qps", 999, "deployment.plan_matches_handoff"),
        ("execute", "storage_uri", "s3://other/model", "deployment.execution"),
        ("execute", "namespace", "other", "deployment.execution"),
        ("status", "name", "existing-model", "deployment.session_state"),
    ],
)
async def test_wrong_tool_arguments_fail_named_check(
    repo: Path, stage: str, key: str, value: Any, check_name: str
) -> None:
    responses = scripted_responses()
    targets = {
        "recommend": responses[0][2].tool_calls[-1],
        "plan": responses[1][0].tool_calls[0],
        "execute": responses[1][2].tool_calls[0],
        "status": responses[1][3].tool_calls[0],
    }
    targets[stage]["arguments"][key] = value
    artifact = await check.run_checks(
        phase="commands",
        output=repo / "failed.json",
        repo_root=repo,
        client_factory=FakeFactory(responses),
    )
    assert artifact["status"] == "failed"
    assert artifact["checks"][check_name]["status"] == "failed", artifact["failure"]


async def test_missing_inventory_and_confirmed_overrides() -> None:
    calls = scripted_responses()[0][2].tool_calls
    trace = {"exchanges": [{"tool_calls": calls, "messages": tool_messages(calls)}]}
    assert check.validate_recommendations(trace) == ["A100-80"]
    with pytest.raises(AssertionError, match="override"):
        check.validate_recommendations(trace, {"ttft_max_ms": 250})
    trace["exchanges"][0]["tool_calls"][-1]["arguments"]["ttft_max_ms"] = 250
    assert check.validate_recommendations(trace, {"ttft_max_ms": 250})
    trace["exchanges"][0]["tool_calls"].reverse()
    with pytest.raises(AssertionError, match="inventory"):
        check.validate_recommendations(trace)


async def test_sdk_failure_preserves_partial_exchange_and_stops(repo: Path) -> None:
    responses = scripted_responses()
    responses[0][2] = SyntheticFailure(responses[0][2])
    factory = FakeFactory(responses)
    output = repo / "partial.json"
    artifact = await check.run_checks(
        phase="commands", output=output, repo_root=repo, client_factory=factory
    )
    assert artifact["status"] == "failed"
    assert "synthetic SDK execution failure" in artifact["failure"]
    assert len(artifact["scenarios"]["navigator"]["exchanges"]) == 3
    partial = json.loads(output.read_text())["scenarios"]["navigator"]["exchanges"][-1]
    assert partial["final_output"] == comparison_table()
    assert partial["tool_calls"][-1]["name"] == "recommend_model"
    assert artifact["scenarios"]["deployment"]["status"] == "not_run"
    assert len(factory.clients) == 1 and factory.clients[0].closed


async def test_static_failure_saves_artifact_without_starting_client(repo: Path) -> None:
    migrate(repo)
    factory = FakeFactory([])
    output = repo / "failed.json"
    artifact = await check.run_checks(
        phase="commands", output=output, repo_root=repo, client_factory=factory
    )
    assert artifact["checks"]["static.sources"]["status"] == "failed"
    assert factory.clients == []
    assert json.loads(output.read_text())["status"] == "failed"


async def paired_artifacts(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = await check.run_checks(
        phase="commands",
        output=repo / "before.json",
        repo_root=repo,
        client_factory=FakeFactory(scripted_responses()),
    )
    migrate(repo)
    after = await check.run_checks(
        phase="skills",
        output=repo / "after.json",
        repo_root=repo,
        client_factory=FakeFactory(scripted_responses(model_id="other/model-a")),
    )
    assert before["status"] == after["status"] == "passed"
    return before, after


async def test_compare_allows_different_actual_recommended_models(repo: Path) -> None:
    before, after = await paired_artifacts(repo)
    assert all(passed for _name, passed, _detail in check.compare_artifacts(before, after))


@pytest.mark.parametrize(
    "fault",
    [
        "check",
        "scenario",
        "body",
        "description",
        "tools",
        "scaffold",
        "phase",
        "script",
        "failure",
        "empty-trace",
    ],
)
async def test_compare_rejects_missing_or_changed_evidence(repo: Path, fault: str) -> None:
    before, original = await paired_artifacts(repo)
    after = copy.deepcopy(original)
    if fault == "check":
        after["checks"].pop("deployment.execution")
    elif fault == "scenario":
        after["scenarios"].pop("deployment")
    elif fault in {"body", "description", "tools"}:
        after["static"]["navigator"]["body_sha256" if fault == "body" else fault] = "changed"
    elif fault == "scaffold":
        after["static"]["scaffold_sha256"] = "changed"
    elif fault == "phase":
        after["phase"] = "commands"
    elif fault == "script":
        after["script"]["approval"] = "yes"
    elif fault == "failure":
        after["checks"]["deployment.execution"]["status"] = "failed"
    else:
        after["scenarios"]["deployment"]["exchanges"] = []
    assert not all(passed for _name, passed, _detail in check.compare_artifacts(before, after))


async def test_compare_cli_and_clear_failure_output(repo: Path, capsys: Any) -> None:
    _before, after = await paired_artifacts(repo)
    assert check.main(["compare", str(repo / "before.json"), str(repo / "after.json")]) == 0
    assert "Commands" in capsys.readouterr().out
    after["checks"]["navigator.handoff"]["status"] = "failed"
    check.write_artifact(repo / "after.json", after)
    assert check.main(["compare", str(repo / "before.json"), str(repo / "after.json")]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_run_cli_saves_static_failure_without_claude(tmp_path: Path, capsys: Any) -> None:
    output = tmp_path / "explicit.json"
    assert (
        check.main(
            ["run", "--phase", "commands", "--repo-root", str(tmp_path), "--output", str(output)]
        )
        == 1
    )
    assert json.loads(output.read_text())["checks"]["static.sources"]["status"] == "failed"
    assert "Required source missing" in capsys.readouterr().err


async def test_runtime_creation_requires_and_obeys_approval(repo: Path) -> None:
    responses = scripted_responses()
    plan_call = responses[1][0].tool_calls[0]
    plan = check.tool_payload(plan_call)
    plan.update(runtime_needs_creation=True, template_to_instantiate="vllm-template")
    plan_call["result"][0]["text"] = json.dumps(plan)
    creation = call(
        "create_serving_runtime",
        {"name": "vllm-runtime"},
        namespace=check.NAMESPACE,
        template_name="vllm-template",
    )
    responses[1][2].tool_calls.insert(0, creation)
    artifact = await check.run_checks(
        phase="commands",
        output=repo / "runtime.json",
        repo_root=repo,
        client_factory=FakeFactory(copy.deepcopy(responses)),
    )
    assert artifact["status"] == "passed", artifact["failure"]
    responses[1][1].tool_calls.append(creation)
    failed = await check.run_checks(
        phase="commands",
        output=repo / "early-runtime.json",
        repo_root=repo,
        client_factory=FakeFactory(responses),
    )
    assert failed["checks"]["deployment.approval_gate"]["status"] == "failed"


@pytest.mark.parametrize("ready", [False, True])
async def test_endpoint_lookups_follow_readiness(repo: Path, ready: bool) -> None:
    responses = scripted_responses(ready=ready)
    if ready:
        responses[1][-1].tool_calls.pop()  # Ready, but the endpoint was never retrieved.
    else:
        responses[1][-1].tool_calls.extend(scripted_responses(ready=True)[1][-1].tool_calls[-1:])
    artifact = await check.run_checks(
        phase="commands",
        output=repo / "endpoint.json",
        repo_root=repo,
        client_factory=FakeFactory(responses),
    )
    assert artifact["checks"]["deployment.session_state"]["status"] == "failed"


def test_unconstrained_fallback_requires_missing_primary_slots() -> None:
    primary = scripted_responses()[0][2].tool_calls
    fallback = call(
        "recommend_model",
        recommendations(),
        text="support chatbot",
        use_case=check.USE_CASE,
        user_count=50,
        preferred_gpu_types=[],
    )
    calls = [*primary, fallback]
    trace = {"exchanges": [{"tool_calls": calls, "messages": tool_messages(calls)}]}
    with pytest.raises(AssertionError, match="GPU filter"):
        check.validate_recommendations(trace)
    incomplete = recommendations()
    incomplete["recommendations"].pop("top_cost")
    primary[-1]["result"][0]["text"] = json.dumps(incomplete)
    assert check.validate_recommendations(trace) == ["A100-80"]


async def test_compare_rejects_shared_scenario_session(repo: Path) -> None:
    before, after = await paired_artifacts(repo)
    for exchange in after["scenarios"]["deployment"]["exchanges"]:
        exchange["conversation_id"] = after["scenarios"]["navigator"]["exchanges"][0][
            "conversation_id"
        ]
    assert not all(passed for _name, passed, _detail in check.compare_artifacts(before, after))


@pytest.mark.parametrize("field", ["messages", "tool_calls"])
async def test_compare_rejects_truncated_trace_evidence(repo: Path, field: str) -> None:
    before, after = await paired_artifacts(repo)
    after["scenarios"]["navigator"]["exchanges"][0].pop(field)
    assert not all(passed for _name, passed, _detail in check.compare_artifacts(before, after))


def test_inventory_result_must_arrive_before_recommendation_request() -> None:
    calls = scripted_responses()[0][2].tool_calls
    messages = tool_messages(calls)
    # Both tools requested together: call order alone must not satisfy discovery.
    messages[0]["tool_calls"].extend(messages[2]["tool_calls"])
    messages.pop(2)
    trace = {"exchanges": [{"tool_calls": calls, "messages": messages}]}
    with pytest.raises(AssertionError, match="inventory result"):
        check.validate_recommendations(trace)


def test_status_result_must_prove_ready_before_endpoint_request() -> None:
    responses = scripted_responses(ready=True)
    prior = {
        "tool_calls": responses[1][-2].tool_calls,
        "messages": tool_messages(responses[1][-2].tool_calls),
    }
    current = {
        "tool_calls": responses[1][-1].tool_calls,
        "messages": tool_messages(responses[1][-1].tool_calls),
    }
    deployed = {"name": "new-model", "namespace": check.NAMESPACE, "status": "Pending"}
    check.validate_status(current, deployed, prior)
    current["messages"][0]["tool_calls"].extend(current["messages"][2]["tool_calls"])
    current["messages"].pop(2)
    with pytest.raises(AssertionError, match="before Ready"):
        check.validate_status(current, deployed, prior)
