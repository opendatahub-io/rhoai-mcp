"""Before/after navigator workflow checks, without DeepEval or a judge.

Run with ``uv run --group claude-check python -m evals.command_skill_check``.
``run`` uses authenticated Claude Code sessions and the local mock-cluster MCP
server. ``compare`` and the validators read saved data without starting Claude.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS = ("navigator", "navigator-deploy")
CATEGORIES = ("balanced", "cost", "performance", "quality")
TABLE_COLUMNS = ("", "Balanced", "Cost", "Performance", "Quality")
TABLE_ROWS = (
    "Model",
    "GPU",
    "TTFT p95",
    "E2E p95",
    "Quality score",
    "Cost/month",
    "Meets SLO",
    "Cluster fit",
)
HANDOFF_KEYS = (
    "model_id",
    "use_case",
    "user_count",
    "chosen_category",
    "prompt_tokens",
    "output_tokens",
    "expected_qps",
    "ttft_target_ms",
    "itl_target_ms",
    "e2e_target_ms",
    "cluster_gpu_types",
)
USE_CASE = "chatbot_conversational"
USER_COUNT = 50
NAMESPACE = "production-models"
STORAGE_URI = "oci://registry.example.com/mock/model:latest"
SCRIPT = {
    "opening": (
        "I'm building a customer support chatbot for 50 concurrent users. Cost is my top priority. "
        "Please confirm the requirements and show the workload profile for review. "
        "I have not confirmed the workload or selected a model yet."
    ),
    "requirements": (
        "I confirm the requirements: a customer support chatbot, 50 concurrent users, cost priority. "
        "This confirms requirements only. Show the workload profile and default SLO targets, "
        "and wait for my workload confirmation before fetching recommendations."
    ),
    "workload": "I confirm the workload and all default SLO targets. Please fetch the model recommendations.",
    "selection": "I select the {category} recommendation, model_id {model_id}. Please provide the navigator-handoff block.",
    "planning": (
        "Prepare the deployment plan from this navigator handoff in namespace production-models. "
        "Show the plan and wait for my explicit deployment approval.\n\n{handoff}"
    ),
    "storage": (
        f"Use {STORAGE_URI} as the model storage URI. This supplies storage only; "
        "I have not approved deployment. Show the resolved plan and wait for 'deploy'."
    ),
    "approval": "deploy",
    "status": (
        "Check the current status of the new service {name} in namespace production-models. "
        "Look up its endpoint only if it is Ready."
    ),
}
CHECKS = (
    "static.sources",
    "navigator.workload_gate",
    "navigator.recommendations",
    "navigator.comparison_table",
    "navigator.handoff",
    "navigator.selection_only",
    "deployment.fresh_session",
    "deployment.approval_gate",
    "deployment.plan_matches_handoff",
    "deployment.execution",
    "deployment.session_state",
)
SCENARIOS = ("navigator", "deployment")
CONTRACT = {
    "version": 1,
    "script": SCRIPT,
    "columns": TABLE_COLUMNS,
    "rows": TABLE_ROWS,
    "handoff_keys": HANDOFF_KEYS,
    "checks": CHECKS,
}
CONTRACT_HASH = hashlib.sha256(json.dumps(CONTRACT, sort_keys=True).encode()).hexdigest()


def require(condition: Any, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def _plain(value: str) -> str:
    return value.strip().replace("**", "").replace("`", "").strip()


def _cells(line: str) -> list[str]:
    row = line.strip().removeprefix("|").removesuffix("|")
    return [_plain(cell) for cell in re.split(r"(?<!\\)\|", row)]


def validate_comparison_table(output: str) -> list[list[str]]:
    """Require the exact table shape, ignoring prose and other Markdown tables."""
    tables: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in [*output.splitlines(), ""]:
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence and line.lstrip().startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    candidates = [
        table for table in tables if any(col in _cells(table[0]) for col in TABLE_COLUMNS[1:])
    ]
    require(len(candidates) == 1, f"Expected one recommendation table; found {len(candidates)}")
    table = [_cells(line) for line in candidates[0]]
    require(tuple(table[0]) == TABLE_COLUMNS, f"Wrong comparison columns: {table[0]}")
    require(
        len(table) == 10, f"Expected header, separator, and eight rows; found {len(table)} lines"
    )
    require(
        all(len(row) == 5 for row in table), "Comparison table must have five cells in every row"
    )
    require(all(re.fullmatch(r":?-{3,}:?", cell) for cell in table[1]), "Invalid table separator")
    require(tuple(row[0] for row in table[2:]) == TABLE_ROWS, "Wrong comparison row names or order")
    placeholders = re.compile(
        r"\[[^]]+\](?!\()|<(?!br\s*/?>)[^>]+>|\{\{.*?\}\}|\b(?:TBD|TODO|placeholder|N/?A|null|None)\b",
        re.I,
    )
    for row in table[2:]:
        for cell in row[1:]:
            require(bool(cell), f"Empty recommendation cell in {row[0]}")
            require(
                not placeholders.search(cell) and cell not in {"...", "…", "-", "--"},
                f"Unresolved recommendation cell in {row[0]}: {cell}",
            )
    return table[2:]


def parse_handoff(output: str) -> tuple[str, dict[str, Any]]:
    blocks = list(re.finditer(r"<!--\s*navigator-handoff\s*\n(.*?)-->", output, re.S))
    require(len(blocks) == 1, f"Expected one navigator-handoff block; found {len(blocks)}")
    values: dict[str, Any] = {}
    for line in blocks[0][1].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        key = key.strip()
        require(separator and key not in values, f"Invalid or duplicate handoff line: {line}")
        values[key] = value.strip().strip("\"'")
    require(set(values) == set(HANDOFF_KEYS), f"Wrong handoff keys: {sorted(values)}")
    for key in HANDOFF_KEYS[2:]:
        if key in {"chosen_category", "cluster_gpu_types"}:
            continue
        try:
            number = float(values[key])
        except (ValueError, TypeError):
            raise AssertionError(f"Handoff {key} is not numeric: {values[key]}") from None
        require(math.isfinite(number) and number > 0, f"Handoff {key} must be positive")
        if key != "expected_qps":
            require(number.is_integer(), f"Handoff {key} must be an integer")
            number = int(number)
        values[key] = number
    values["cluster_gpu_types"] = [
        part.strip() for part in values["cluster_gpu_types"].split(",") if part.strip()
    ]
    require(values["chosen_category"] in CATEGORIES, "Invalid handoff category")
    require(bool(values["model_id"]), "Empty handoff model_id")
    return blocks[0][0], values


def _source_metadata(path: Path, *, skill_name: str | None) -> dict[str, Any]:
    raw = path.read_bytes()
    match = re.match(rb"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", raw, re.M | re.S)
    require(match is not None, f"Missing YAML frontmatter: {path}")
    assert match is not None
    node = yaml.compose(match[1], Loader=yaml.SafeLoader)
    require(isinstance(node, yaml.MappingNode), f"Frontmatter must be a mapping: {path}")
    require(
        all(
            isinstance(key, yaml.ScalarNode) and key.tag == "tag:yaml.org,2002:str"
            for key, _value in node.value
        ),
        f"Frontmatter keys must be strings: {path}",
    )
    keys = [key.value for key, _value in node.value]
    require(len(keys) == len(set(keys)), f"Duplicate frontmatter keys: {path}")
    metadata = yaml.safe_load(match[1])
    description = metadata.get("description")
    require(isinstance(description, str) and description.strip(), f"Missing description: {path}")
    require(len(description) <= 1024, f"Description too long: {path}")
    tools = metadata.get("allowed-tools")
    require(isinstance(tools, str) and tools.strip(), f"Missing allowed-tools: {path}")
    if skill_name:
        require(
            metadata.get("name") == skill_name == path.parent.name, f"Invalid skill name: {path}"
        )
        require(
            len(skill_name) <= 64 and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", skill_name),
            "Invalid skill name syntax",
        )
        require(
            "," not in tools and tools == " ".join(tools.split()),
            f"allowed-tools must use spaces: {path}",
        )
    ordered_tools = re.split(r"[,\s]+", tools.strip())
    require(
        all(ordered_tools) and len(ordered_tools) == len(set(ordered_tools)),
        f"Empty or duplicate tools: {path}",
    )
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "body_sha256": hashlib.sha256(raw[match.end() :]).hexdigest(),
        "description": description,
        "tools": ordered_tools,
        "name": metadata.get("name"),
    }


def validate_sources(repo_root: Path, phase: str) -> dict[str, Any]:
    require(phase in {"commands", "skills"}, f"Invalid phase: {phase}")
    metadata: dict[str, Any] = {}
    for name in SKILLS:
        command = repo_root / ".claude/commands" / f"{name}.md"
        skill = repo_root / ".claude/skills" / name / "SKILL.md"
        source, absent = (command, skill) if phase == "commands" else (skill, command)
        require(source.is_file(), f"Required source missing: {source}")
        require(not absent.exists(), f"Old and replacement sources must not coexist: {absent}")
        metadata[name] = _source_metadata(source, skill_name=name if phase == "skills" else None)
        metadata[name]["path"] = str(source.relative_to(repo_root))
    scaffold = repo_root / ".claude/commands/scaffold-plugin.md"
    require(scaffold.is_file(), "scaffold-plugin.md must remain a command")
    require(
        not (repo_root / ".claude/skills/scaffold-plugin").exists(),
        "scaffold-plugin must not acquire a skill replacement",
    )
    metadata["scaffold_sha256"] = hashlib.sha256(scaffold.read_bytes()).hexdigest()
    return metadata


def tool_payload(call: dict[str, Any]) -> dict[str, Any]:
    """Decode SDK text blocks or structured MCP content, preserving errors as failures."""
    require(not call.get("is_error"), f"MCP tool failed: {call['name']}: {call.get('result')}")
    value = call.get("result")
    for _ in range(5):
        if isinstance(value, str):
            value = json.loads(value)
        elif isinstance(value, list):
            require(
                all(block.get("type") == "text" for block in value),
                "Expected JSON text tool result",
            )
            value = "\n".join(block["text"] for block in value)
        elif isinstance(value, dict):
            require(
                not (value.get("error") or value.get("isError")),
                f"MCP tool {call['name']} returned an error: {value}",
            )
            if "structuredContent" in value:
                value = value["structuredContent"]
            elif "content" in value and isinstance(value["content"], list):
                value = value["content"]
            else:
                return value
        else:
            break
    raise AssertionError(f"Cannot decode JSON result for {call['name']}")


def _calls(trace: dict[str, Any], name: str | None = None) -> list[dict[str, Any]]:
    calls = [call for exchange in trace["exchanges"] for call in exchange["tool_calls"]]
    return [call for call in calls if call["name"] == name] if name else calls


def _last_call(trace: dict[str, Any], name: str) -> dict[str, Any]:
    calls = _calls(trace, name)
    require(calls, f"Required MCP call missing: {name}")
    return calls[-1]


def _gpu_types(inventory: dict[str, Any]) -> list[str]:
    products = (inventory.get("gpu_info") or {}).get("products", [])
    found = set()
    for product in products:
        product = product.upper()
        if "A100" in product and "80" in product:
            found.add("A100-80")
        elif "A100" in product and "40" in product:
            found.add("A100-40")
        else:
            found.update(gpu for gpu in ("H100", "H200", "B200", "L4") if gpu in product)
    require(found, f"Mock cluster GPU inventory could not be mapped: {products}")
    return sorted(found)


def _result_precedes_call(trace: dict[str, Any], result_id: str, call_id: str) -> bool:
    """Check completion order, including tools requested in one assistant message."""
    seen_result = False
    for exchange in trace["exchanges"]:
        for message in exchange["messages"]:
            if message["role"] == "tool" and message.get("tool_call_id") == result_id:
                seen_result = True
            if any(call.get("id") == call_id for call in message.get("tool_calls", [])):
                return seen_result
    return False


def validate_recommendations(
    trace: dict[str, Any], overrides: dict[str, int] | None = None
) -> list[str]:
    calls = _calls(trace)
    recommendations = _calls(trace, "recommend_model")
    require(recommendations, "No recommendations after workload confirmation")
    first_index = calls.index(recommendations[0])
    inventories = [call for call in calls[:first_index] if call["name"] == "get_cluster_resources"]
    require(inventories, "Cluster inventory must precede the first recommendation")
    inventories = [
        call
        for call in inventories
        if _result_precedes_call(trace, call["call_id"], recommendations[0]["call_id"])
    ]
    require(
        inventories,
        "Cluster inventory result must be available before recommendations are requested",
    )
    gpus = _gpu_types(tool_payload(inventories[-1]))
    allow_fallback = False
    for index, call in enumerate(recommendations):
        arguments = call["arguments"]
        require(
            arguments.get("use_case") == USE_CASE and arguments.get("user_count") == USER_COUNT,
            "Recommendation use_case/user_count do not match confirmed requirements",
        )
        gpu_filter = arguments.get("preferred_gpu_types")
        require(isinstance(gpu_filter, list), "Recommendation omitted explicit GPU filter")
        require(
            sorted(gpu_filter) == gpus or (index > 0 and allow_fallback and gpu_filter == []),
            f"Recommendation GPU filter does not match inventory/fallback: {gpu_filter}",
        )
        for key, expected in (overrides or {}).items():
            value = arguments.get(key)
            require(
                value is not None and (value == expected if index == 0 else 0 < value <= expected),
                f"Confirmed SLO override {key} was not respected",
            )
        payload = tool_payload(call)
        slots = payload.get("recommendations", {})
        require(isinstance(slots, dict), "Recommendation result has invalid slots")
        allow_fallback |= bool(gpu_filter) and any(
            not slots.get(f"top_{category}") for category in CATEGORIES
        )
    return gpus


def _select_recommendation(
    trace: dict[str, Any], table: list[list[str]]
) -> tuple[str, dict[str, Any]]:
    # Balanced remains the primary slot when other slots receive duplicate-removal constraints.
    for category in CATEGORIES:
        column = CATEGORIES.index(category) + 1
        label = table[0][column]
        if label == "—":
            continue
        for call in reversed(_calls(trace, "recommend_model")):
            recommendation = tool_payload(call).get("recommendations", {}).get(f"top_{category}")
            if recommendation and (
                recommendation["model_id"] in label
                or (recommendation.get("model") and recommendation["model"] in label)
            ):
                return category, recommendation
    raise AssertionError("No displayed recommendation could be selected from actual MCP results")


def validate_handoff(
    values: dict[str, Any],
    *,
    category: str,
    recommendation: dict[str, Any],
    defaults: dict[str, Any],
    traffic: dict[str, Any],
    gpus: list[str],
) -> None:
    expected = {
        "model_id": recommendation["model_id"],
        "chosen_category": category,
        "use_case": USE_CASE,
        "user_count": USER_COUNT,
        "prompt_tokens": defaults["workload"]["prompt_tokens"],
        "output_tokens": defaults["workload"]["output_tokens"],
        "expected_qps": traffic["expected_rps"],
    }
    expected.update(
        {
            f"{metric}_target_ms": defaults["slo_targets"][f"{metric}_ms"]["default"]
            for metric in ("ttft", "itl", "e2e")
        }
    )
    for key, value in expected.items():
        require(values[key] == value, f"Handoff {key}: expected {value!r}, got {values[key]!r}")
    require(sorted(values["cluster_gpu_types"]) == gpus, "Handoff GPU types do not match inventory")


class CheckFailure(AssertionError):
    pass


@contextmanager
def checking(artifact: dict[str, Any], name: str) -> Iterator[None]:
    try:
        yield
    except CheckFailure:
        raise
    except Exception as exc:
        artifact["checks"][name] = {"status": "failed", "detail": str(exc)}
        raise CheckFailure(f"{name}: {exc}") from exc
    else:
        artifact["checks"][name] = {"status": "passed", "detail": ""}


class _Session:
    def __init__(self, client: Any, trace: dict[str, Any]) -> None:
        self.client = client
        self.trace = trace
        self.conversation_id = ""

    def _capture(self, result: Any, label: str) -> dict[str, Any]:
        exchange = {"label": label, **asdict(result), "diagnostics": self.client.diagnostics}
        self.trace["exchanges"].append(exchange)
        self.trace["initialization"] = self.client.initialization.copy()
        self.conversation_id = result.conversation_id
        return exchange

    async def send(self, message: str, label: str, *, skill: str | None = None) -> dict[str, Any]:
        previous = self.client.last_result
        try:
            if skill:
                result = await self.client.query(message, skill=skill)
            else:
                result = await self.client.follow_up(message, self.conversation_id)
        except BaseException:
            partial = self.client.last_result
            if partial is not None and partial is not previous:
                self._capture(partial, label)
            raise
        exchange = self._capture(result, label)
        for call in exchange["tool_calls"]:
            tool_payload(call)
        return exchange


def _selection_only(artifact: dict[str, Any], trace: dict[str, Any]) -> None:
    with checking(artifact, "navigator.selection_only"):
        forbidden = {
            "plan_deployment",
            "execute_deployment",
            "deploy_model",
            "create_serving_runtime",
        }
        require(
            not any(call["name"] in forbidden for call in _calls(trace)),
            "Navigator attempted deployment",
        )


async def _navigator(artifact: dict[str, Any], session: _Session) -> tuple[str, dict[str, Any]]:
    trace = session.trace
    with checking(artifact, "navigator.workload_gate"):
        await session.send(SCRIPT["opening"], "opening", skill="navigator")
        _selection_only(artifact, trace)
        require(
            not _calls(trace, "recommend_model"),
            "Recommendations occurred before workload confirmation",
        )
        await session.send(SCRIPT["requirements"], "requirements")
        _selection_only(artifact, trace)
        require(
            not _calls(trace, "recommend_model"),
            "Requirements confirmation bypassed the workload gate",
        )
        for name in ("get_use_case_defaults", "get_expected_rps"):
            for call in _calls(trace, name):
                require(
                    call["arguments"].get("use_case") == USE_CASE,
                    f"Wrong use_case passed to {name}",
                )
                if name == "get_expected_rps":
                    require(
                        call["arguments"].get("user_count") == USER_COUNT, "Wrong RPS user count"
                    )
        defaults = tool_payload(_last_call(trace, "get_use_case_defaults"))
        traffic = tool_payload(_last_call(trace, "get_expected_rps"))
    with checking(artifact, "navigator.recommendations"):
        recommendations = await session.send(SCRIPT["workload"], "workload")
        _selection_only(artifact, trace)
        gpus = validate_recommendations(trace)
    with checking(artifact, "navigator.comparison_table"):
        table = validate_comparison_table(recommendations["final_output"])
    with checking(artifact, "navigator.handoff"):
        category, chosen = _select_recommendation(trace, table)
        selection = SCRIPT["selection"].format(category=category, model_id=chosen["model_id"])
        result = await session.send(selection, "selection")
        _selection_only(artifact, trace)
        block, handoff = parse_handoff(result["final_output"])
        validate_handoff(
            handoff,
            category=category,
            recommendation=chosen,
            defaults=defaults,
            traffic=traffic,
            gpus=gpus,
        )
    return block, handoff


def _no_deployment(trace: dict[str, Any]) -> None:
    forbidden = {"execute_deployment", "create_serving_runtime", "deploy_model"}
    require(
        not any(call["name"] in forbidden for call in _calls(trace)),
        "Deployment/runtime creation preceded approval",
    )


def validate_plan(trace: dict[str, Any], handoff: dict[str, Any]) -> dict[str, Any]:
    for call in _calls(trace, "plan_deployment"):
        expected = {
            key: handoff[key]
            for key in HANDOFF_KEYS
            if key not in {"model_id", "chosen_category", "cluster_gpu_types"}
        }
        expected.update(
            category=handoff["chosen_category"],
            namespace=NAMESPACE,
            preferred_gpu_types=handoff["cluster_gpu_types"],
        )
        for key, value in expected.items():
            actual = call["arguments"].get(key)
            require(
                sorted(actual or []) == sorted(value)
                if key == "preferred_gpu_types"
                else actual == value,
                f"Deployment plan argument {key} does not match handoff",
            )
        payload = tool_payload(call)
        require(
            payload.get("model_id") == handoff["model_id"], "Planner changed the selected model"
        )
        require(payload.get("namespace") == NAMESPACE, "Planner changed the namespace")
    plan = tool_payload(_last_call(trace, "plan_deployment"))
    require(
        plan.get("ready") is True and not plan.get("issues"),
        f"Deployment plan has unresolved issues: {plan.get('issues')}",
    )
    return plan


def validate_execution(
    trace: dict[str, Any], plan: dict[str, Any], storage_uri: str
) -> dict[str, Any]:
    required = (
        "model_id",
        "namespace",
        "runtime",
        "gpu_count",
        "gpu_type",
        "tensor_parallel",
        "replicas",
    )
    expected = {key: plan[key] for key in required}
    expected["storage_uri"] = storage_uri
    executions = _calls(trace, "execute_deployment")
    require(executions, "No execute_deployment after explicit approval")
    for call in executions:
        for key, value in expected.items():
            actual = call["arguments"].get(
                key, 1 if key in {"tensor_parallel", "replicas"} else None
            )
            require(actual == value, f"Deployment changed approved {key}: {actual!r} != {value!r}")
        result = tool_payload(call)
        require(result.get("deployed") is True, f"Deployment did not succeed: {result}")
        require(
            result.get("namespace") == NAMESPACE and result.get("name"),
            "Deployment returned no matching service identity",
        )
    if plan.get("runtime_needs_creation"):
        creation = _last_call(trace, "create_serving_runtime")
        require(
            creation["arguments"].get("namespace") == NAMESPACE,
            "Runtime created in wrong namespace",
        )
        require(
            creation["arguments"].get("template_name") == plan["template_to_instantiate"],
            "Wrong runtime template",
        )
        tool_payload(creation)
        calls = _calls(trace)
        require(
            calls.index(creation) < calls.index(executions[0]),
            "Deployment preceded required runtime creation",
        )
        require(
            _result_precedes_call(trace, creation["call_id"], executions[0]["call_id"]),
            "Runtime creation must finish before deployment is requested",
        )
    return tool_payload(executions[-1])


def validate_status(
    exchange: dict[str, Any], deployment: dict[str, Any], prior_exchange: dict[str, Any]
) -> None:
    calls = exchange["tool_calls"]
    require(
        not any(
            call["name"] in {"execute_deployment", "deploy_model", "create_serving_runtime"}
            for call in calls
        ),
        "Status follow-up unexpectedly created a deployment/runtime",
    )
    status_calls = [call for call in calls if call["name"] == "get_inference_service"]
    require(status_calls, "Follow-up did not inspect the new service")
    all_calls = [*prior_exchange["tool_calls"], *calls]
    for call in all_calls:
        if call["name"] not in {"get_inference_service", "get_model_endpoint"}:
            continue
        require(
            call["arguments"].get("name") == deployment["name"]
            and call["arguments"].get("namespace") == NAMESPACE,
            "Status/endpoint lookup targeted a different service",
        )
        payload = tool_payload(call)
        require(
            payload.get("name") == deployment["name"],
            "Status result does not identify the new service",
        )
        require(payload.get("status"), "Status result has no provisioning state")
        if call["name"] == "get_inference_service":
            require(payload.get("namespace") == NAMESPACE, "Status result namespace changed")
        else:
            require(
                payload.get("url") or payload.get("internal_url"),
                "Ready service has no endpoint URL",
            )
    ready = deployment.get("status") == "Ready"
    by_id = {call["call_id"]: call for call in all_calls}
    for turn in (prior_exchange, exchange):
        for message in turn["messages"]:
            for request in message.get("tool_calls", []):
                call = by_id.get(request["id"], {})
                if call.get("name") == "get_model_endpoint":
                    require(ready, "Endpoint lookup occurred before Ready")
            if message["role"] == "tool":
                call = by_id.get(message.get("tool_call_id"), {})
                if call.get("name") == "get_inference_service":
                    ready = tool_payload(call)["status"] == "Ready"
    if ready:
        require(
            any(call["name"] == "get_model_endpoint" for call in all_calls),
            "Ready service endpoint was not retrieved",
        )


async def _deployment(
    artifact: dict[str, Any], session: _Session, block: str, handoff: dict[str, Any]
) -> None:
    trace = session.trace
    with checking(artifact, "deployment.approval_gate"):
        await session.send(
            SCRIPT["planning"].format(handoff=block), "planning", skill="navigator-deploy"
        )
        with checking(artifact, "deployment.fresh_session"):
            navigator_id = artifact["scenarios"]["navigator"]["exchanges"][0]["conversation_id"]
            require(
                session.conversation_id and session.conversation_id != navigator_id,
                "Deployment must start in a fresh SDK session",
            )
        _no_deployment(trace)
        plan = tool_payload(_last_call(trace, "plan_deployment"))
        if not plan.get("storage_uri"):
            await session.send(SCRIPT["storage"], "storage")
            _no_deployment(trace)
    with checking(artifact, "deployment.plan_matches_handoff"):
        plan = validate_plan(trace, handoff)
    with checking(artifact, "deployment.execution"):
        execution_exchange = await session.send(SCRIPT["approval"], "approval")
        deployed = validate_execution(trace, plan, plan.get("storage_uri") or STORAGE_URI)
    with checking(artifact, "deployment.session_state"):
        status = await session.send(SCRIPT["status"].format(name=deployed["name"]), "status")
        validate_status(status, deployed, execution_exchange)


def new_artifact(phase: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase,
        "backend": "claude-code",
        "created_at": datetime.now(UTC).isoformat(),
        "contract_hash": CONTRACT_HASH,
        "script": SCRIPT.copy(),
        "status": "running",
        "failure": None,
        "static": {},
        "checks": {name: {"status": "not_run", "detail": ""} for name in CHECKS},
        "scenarios": {
            name: {"status": "not_run", "failure": None, "initialization": {}, "exchanges": []}
            for name in SCENARIOS
        },
    }


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


async def run_checks(
    *,
    phase: str,
    output: Path,
    repo_root: Path = REPO_ROOT,
    timeout: float = 300,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    artifact = new_artifact(phase)
    active: dict[str, Any] | None = None
    try:
        with checking(artifact, "static.sources"):
            artifact["static"] = validate_sources(repo_root, phase)
        if client_factory is None:
            from evals.claude_code_client import ClaudeCodeClient

            client_factory = ClaudeCodeClient
        handoff: tuple[str, dict[str, Any]] | None = None
        for name in SCENARIOS:
            active = artifact["scenarios"][name]
            active["status"] = "running"
            async with client_factory(repo_root=repo_root, timeout=timeout) as client:
                session = _Session(client, active)
                active["initialization"] = client.initialization.copy()
                if name == "navigator":
                    handoff = await _navigator(artifact, session)
                else:
                    assert handoff is not None
                    await _deployment(artifact, session, *handoff)
            active["status"] = "passed"
        artifact["status"] = "passed"
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["failure"] = f"{type(exc).__name__}: {exc}"
        if active is not None:
            active["status"] = "failed"
            active["failure"] = artifact["failure"]
    finally:
        write_artifact(output, artifact)
    return artifact


def compare_artifacts(before: dict[str, Any], after: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Compare named contracts and packaging invariants, never recommendation identities."""
    rows: list[tuple[str, bool, str]] = []

    def add(name: str, condition: Any, detail: str = "") -> None:
        rows.append((name, bool(condition), "" if condition else detail))

    add(
        "artifact.phases",
        before.get("phase") == "commands" and after.get("phase") == "skills",
        "Expected commands then skills",
    )
    add(
        "artifact.schema",
        before.get("schema_version") == after.get("schema_version") == 1,
        "Unknown artifact schema",
    )
    add(
        "artifact.contract",
        before.get("contract_hash") == after.get("contract_hash") == CONTRACT_HASH
        and before.get("script") == after.get("script") == SCRIPT,
        "Workflow scripts/contracts differ",
    )
    for phase, artifact in (("commands", before), ("skills", after)):
        add(
            f"{phase}.completed",
            artifact.get("status") == "passed" and not artifact.get("failure"),
            str(artifact.get("failure")),
        )
        for name in SCENARIOS:
            scenario = artifact.get("scenarios", {}).get(name, {})
            exchanges = scenario.get("exchanges", [])
            ids = {exchange.get("conversation_id") for exchange in exchanges}
            evidence = all(
                exchange.get("messages")
                and isinstance(exchange.get("tool_calls"), list)
                and exchange.get("task")
                and exchange.get("final_output")
                and exchange.get("turns", 0) > 0
                for exchange in exchanges
            )
            complete = (
                scenario.get("status") == "passed"
                and not scenario.get("failure")
                and exchanges
                and evidence
                and len(ids) == 1
                and bool(next(iter(ids)))
            )
            add(
                f"{phase}.{name}.scenario",
                complete,
                "Scenario missing, incomplete, or session IDs inconsistent",
            )
        first_ids = [
            artifact.get("scenarios", {})
            .get(name, {})
            .get("exchanges", [{}])[0]
            .get("conversation_id")
            for name in SCENARIOS
            if artifact.get("scenarios", {}).get(name, {}).get("exchanges")
        ]
        add(
            f"{phase}.fresh_sessions",
            len(first_ids) == 2 and len(set(first_ids)) == 2 and all(first_ids),
            "Navigator and deployment must use distinct sessions",
        )
    for name in CHECKS:
        left = before.get("checks", {}).get(name, {})
        right = after.get("checks", {}).get(name, {})
        add(
            name,
            left.get("status") == right.get("status") == "passed",
            f"commands={left.get('status', 'missing')}; skills={right.get('status', 'missing')}; {left.get('detail', '')} {right.get('detail', '')}",
        )
    for name in SKILLS:
        left = before.get("static", {}).get(name, {})
        right = after.get("static", {}).get(name, {})
        for field in ("body_sha256", "description", "tools"):
            add(
                f"invariant.{name}.{field}",
                bool(left.get(field)) and left.get(field) == right.get(field),
                "Source invariant changed or missing",
            )
        add(f"invariant.{name}.name", right.get("name") == name, "Skill name mismatch")
    left_hash = before.get("static", {}).get("scaffold_sha256")
    add(
        "invariant.scaffold",
        left_hash and left_hash == after.get("static", {}).get("scaffold_sha256"),
        "Scaffold changed or missing",
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    run = commands.add_parser(
        "run", help="Run workflows through Claude Code and save JSON evidence"
    )
    run.add_argument("--phase", choices=("commands", "skills"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    run.add_argument(
        "--timeout", type=float, default=300, help="Seconds allowed per Claude exchange"
    )
    compare = commands.add_parser("compare", help="Compare saved before/after assertion artifacts")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.operation == "run":
            if not math.isfinite(args.timeout) or args.timeout <= 0:
                parser.error("--timeout must be a finite positive number")
            artifact = asyncio.run(
                run_checks(
                    phase=args.phase,
                    output=args.output,
                    repo_root=args.repo_root.resolve(),
                    timeout=args.timeout,
                )
            )
            for name, outcome in artifact["checks"].items():
                print(f"{outcome['status'].upper():7} {name} {outcome['detail']}")
            print(f"Artifact: {args.output}")
            if artifact["failure"]:
                print(artifact["failure"], file=sys.stderr)
            return 0 if artifact["status"] == "passed" else 1
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("Artifacts must be JSON objects")
        rows = compare_artifacts(before, after)
        print(f"{'Check':48} {'Commands':9} {'Skills':9} {'Result':6} Detail")
        for name, passed, detail in rows:
            left = before.get("checks", {}).get(name, {}).get("status", "—")
            right = after.get("checks", {}).get(name, {}).get("status", "—")
            print(f"{name:48} {left:9} {right:9} {'PASS' if passed else 'FAIL':6} {detail}")
        return 0 if all(passed for _name, passed, _detail in rows) else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
