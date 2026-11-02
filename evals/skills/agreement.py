"""Compare structured selections with scenario intent and transcript evidence.

Scenario intent is explicit: it must never be inferred from the handoff or
execution arguments being graded. Derived workload values come from successful
tool results before confirmation. Recommendation identity comes from the slot
the customer actually saw, not whichever rerun happened to finish last.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from evals.agent.transcript import AgentTranscript, ToolCallRecord

SLOTS = {
    "balanced": "top_balanced",
    "cost": "top_cost",
    "performance": "top_performance",
    "quality": "top_quality",
}
EXECUTION_FIELDS = (
    "model_id",
    "namespace",
    "runtime",
    "storage_uri",
    "gpu_count",
    "gpu_type",
    "tensor_parallel",
    "replicas",
)
NUMERIC_FIELDS = {
    "user_count",
    "prompt_tokens",
    "output_tokens",
    "expected_qps",
    "ttft_target_ms",
    "itl_target_ms",
    "e2e_target_ms",
    "gpu_count",
    "tensor_parallel",
    "replicas",
}


@dataclass(frozen=True)
class SelectionExpectation:
    """Facts supplied or selected by the scripted customer, independent of the agent."""

    use_case: str
    user_count: int
    category: str
    cluster_gpu_types: tuple[str, ...]
    model_id: str | None = None
    namespace: str | None = None
    storage_uri: str | None = None
    slo_overrides: dict[str, int] = field(default_factory=dict)


def clean(value: Any) -> str:
    return str(value).strip().strip("`*").strip()


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def gpu_type(value: Any) -> str:
    return re.sub(r"^NVIDIA[- ]", "", clean(value).upper())


def gpu_selection(value: Any) -> tuple[int, str] | None:
    matches = list(
        re.finditer(
            r"\b(\d+)\s*[x×]\s*((?:NVIDIA[- ])?(?:A100-(?:40|80)|H100|H200|B200|L4))\b",
            clean(value),
            re.I,
        )
    )
    return (int(matches[0][1]), gpu_type(matches[0][2])) if len(matches) == 1 else None


def same(field: str, actual: Any, expected: Any) -> bool:
    if actual is None or expected is None or isinstance(actual, bool) or isinstance(expected, bool):
        return False
    if field == "model_id":
        return clean(actual).casefold() == clean(expected).casefold()
    if field == "gpu_type":
        return gpu_type(actual) == gpu_type(expected)
    if field in NUMERIC_FIELDS:
        try:
            number = float(actual)
            target = float(expected)
        except (ValueError, TypeError):
            return False
        # Handoffs commonly round QPS to three decimals; all other numbers
        # must agree exactly. Non-finite values never establish agreement.
        tolerance = 0.0005 if field == "expected_qps" else 0.0
        return (
            math.isfinite(number)
            and math.isfinite(target)
            and number > 0
            and target > 0
            and math.isclose(number, target, rel_tol=0, abs_tol=tolerance)
        )
    return clean(actual) == clean(expected)


def differences(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    return [
        f"{key}: expected {value!r}, got {actual.get(key)!r}"
        for key, value in expected.items()
        if not same(key, actual.get(key), value)
    ]


def valid_gate(transcript: AgentTranscript, index: int | None) -> bool:
    return (
        index is not None
        and 0 <= index < len(transcript.turns)
        and transcript.turns[index].role == "user"
    )


def confirmed_workload(
    transcript: AgentTranscript, intent: SelectionExpectation, confirmation: int | None
) -> tuple[dict[str, Any], list[str]]:
    """Read defaults and expected traffic only from evidence the customer could confirm."""
    if confirmation is None or not valid_gate(transcript, confirmation):
        return {}, ["workload confirmation turn is missing"]
    results: dict[str, dict[str, Any]] = {}
    for name in ("get_use_case_defaults", "get_expected_rps"):
        candidates = [
            c
            for c in transcript.mcp_tool_calls
            if c.name == name
            and c.turn_index < confirmation
            and not c.is_error
            and isinstance(c.structured_result, dict)
            and not c.structured_result.get("error")
            and c.arguments.get("use_case") == intent.use_case
            and (name != "get_expected_rps" or c.arguments.get("user_count") == intent.user_count)
        ]
        if not candidates:
            return {}, [f"no successful {name} for the scripted workload before confirmation"]
        results[name] = mapping(candidates[-1].structured_result)
    defaults, traffic = results["get_use_case_defaults"], results["get_expected_rps"]
    workload = {
        "use_case": intent.use_case,
        "user_count": intent.user_count,
        "prompt_tokens": mapping(defaults.get("workload")).get("prompt_tokens"),
        "output_tokens": mapping(defaults.get("workload")).get("output_tokens"),
        "expected_qps": traffic.get("expected_rps"),
        **{
            f"{metric}_target_ms": intent.slo_overrides.get(
                f"{metric}_target_ms",
                mapping(mapping(defaults.get("slo_targets")).get(f"{metric}_ms")).get("default"),
            )
            for metric in ("ttft", "itl", "e2e")
        },
    }
    errors = differences(traffic, {"use_case": intent.use_case, "user_count": intent.user_count})
    errors += differences(defaults, {"use_case": intent.use_case})
    errors += [
        f"confirmed source omitted {key}" for key, value in workload.items() if value is None
    ]
    return workload, errors


def selected_recommendation(
    transcript: AgentTranscript,
    intent: SelectionExpectation,
    selection: int | None,
    model_cell: str,
    gpu_cell: str,
    *,
    after_turn: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Find the selected slot by its displayed model, excluding subsequent tool calls."""
    if selection is None or not valid_gate(transcript, selection):
        return {}, {}, ["configuration selection turn is missing"]
    slot = SLOTS.get(intent.category)
    if slot is None:
        return {}, {}, [f"unknown recommendation category {intent.category!r}"]
    cell = re.sub(r"[*`]", "", model_cell).casefold()
    matches = []
    for call in transcript.mcp_tool_calls:
        data = call.structured_result
        if (
            call.name != "recommend_model"
            or call.turn_index >= selection
            or (after_turn is not None and call.turn_index <= after_turn)
            or call.is_error
            or not isinstance(data, dict)
        ):
            continue
        recommendation = mapping(data.get("recommendations")).get(slot)
        if not isinstance(recommendation, dict) or not recommendation.get("model_id"):
            continue
        if gpu_selection(gpu_cell) is None or gpu_selection(gpu_cell) != gpu_selection(
            recommendation.get("gpu", "")
        ):
            continue
        identities = [recommendation.get("model_id"), recommendation.get("model")]
        if any(
            name
            and re.search(r"(?<![\w/-])" + re.escape(clean(name).casefold()) + r"(?![\w/-])", cell)
            for name in identities
        ):
            matches.append((call, recommendation, mapping(data.get("specification"))))
    if not matches:
        return (
            {},
            {},
            ["displayed selection has no matching successful recommendation in the chosen slot"],
        )
    if len({clean(rec["model_id"]).casefold() for _, rec, _ in matches}) != 1:
        return {}, {}, ["displayed model name matches multiple model IDs"]
    # Only matching model/GPU offers in this slot can update its evidence;
    # a later rerun offering a different model cannot replace the selection.
    call, recommendation, specification = matches[-1]
    errors = differences(
        call.arguments, {"use_case": intent.use_case, "user_count": intent.user_count}
    )
    errors += differences(
        specification, {"use_case": intent.use_case, "user_count": intent.user_count}
    )
    requested_gpus = call.arguments.get("preferred_gpu_types")
    inventory = {gpu_type(value) for value in intent.cluster_gpu_types}
    if not isinstance(requested_gpus, list):
        errors.append("recommendation omitted the hardware filter")
    elif requested_gpus:
        if {gpu_type(value) for value in requested_gpus} != inventory:
            errors.append("recommendation hardware filter differs from the customer inventory")
    elif inventory:
        # The skills permit an unconstrained fallback only when a preceding
        # cluster-constrained call returned no candidate for this slot.
        fallback_allowed = any(
            previous.name == "recommend_model"
            and previous.turn_index < call.turn_index
            and (after_turn is None or previous.turn_index > after_turn)
            and not previous.is_error
            and not differences(
                previous.arguments, {"use_case": intent.use_case, "user_count": intent.user_count}
            )
            and isinstance(previous.arguments.get("preferred_gpu_types"), list)
            and {gpu_type(g) for g in previous.arguments["preferred_gpu_types"]} == inventory
            and SLOTS[intent.category]
            in mapping(mapping(previous.structured_result).get("recommendations"))
            and mapping(mapping(previous.structured_result).get("recommendations"))[
                SLOTS[intent.category]
            ]
            is None
            for previous in transcript.mcp_tool_calls
        )
        if not fallback_allowed:
            errors.append(
                "unconstrained recommendation has no preceding empty cluster-constrained slot"
            )
    if intent.model_id:
        errors += differences(recommendation, {"model_id": intent.model_id})
    return recommendation, specification, errors


def handoff_errors(
    handoff: dict[str, str], workload: dict[str, Any], intent: SelectionExpectation
) -> list[str]:
    errors = differences(handoff, workload)
    actual_gpus = {
        gpu_type(value)
        for value in handoff.get("cluster_gpu_types", "").split(",")
        if value.strip()
    }
    expected_gpus = {gpu_type(value) for value in intent.cluster_gpu_types}
    if "cluster_gpu_types" not in handoff or actual_gpus != expected_gpus:
        errors.append(
            f"cluster_gpu_types: expected {sorted(expected_gpus)}, got {sorted(actual_gpus)}"
        )
    return errors


def plan_selection_errors(
    plan: ToolCallRecord,
    recommendation: dict[str, Any],
    specification: dict[str, Any],
    workload: dict[str, Any],
    intent: SelectionExpectation,
) -> list[str]:
    expected = {
        **workload,
        # Deployment's skill contract takes traffic from the recommendation
        # specification; navigator's handoff instead uses get_expected_rps.
        **{
            k: mapping(specification.get("traffic_profile")).get(k)
            for k in ("prompt_tokens", "output_tokens", "expected_qps")
        },
        "category": intent.category,
        "namespace": intent.namespace,
    }
    errors = differences(plan.arguments, expected)
    errors += differences(
        mapping(specification.get("traffic_profile")),
        {key: workload.get(key) for key in ("prompt_tokens", "output_tokens")},
    )
    gpus = plan.arguments.get("preferred_gpu_types")
    if not isinstance(gpus, list) or {gpu_type(g) for g in gpus} != {
        gpu_type(g) for g in intent.cluster_gpu_types
    }:
        errors.append("plan preferred_gpu_types differ from the customer-supplied inventory")
    data = plan.structured_result if isinstance(plan.structured_result, dict) else {}
    errors += differences(
        data, {"model_id": recommendation.get("model_id"), "namespace": intent.namespace}
    )
    selected_gpu = gpu_selection(recommendation.get("gpu", ""))
    if selected_gpu is None:
        errors.append("selected recommendation has no readable GPU configuration")
    else:
        errors += differences(data, {"gpu_count": selected_gpu[0], "gpu_type": selected_gpu[1]})
    return errors


def execution_errors(
    call: ToolCallRecord, plan: ToolCallRecord, review: dict[str, str], intent: SelectionExpectation
) -> list[str]:
    """Compare every execution with both the structured plan and the visible review."""
    data = mapping(plan.structured_result)
    expected = {key: data.get(key) for key in EXECUTION_FIELDS}
    expected["tensor_parallel"] = data.get("tensor_parallel", 1)
    if expected["storage_uri"] is None:
        expected["storage_uri"] = intent.storage_uri
    effective = {"tensor_parallel": 1, "replicas": 1, **call.arguments}
    errors = differences(effective, expected)
    review_expected = {
        "model": expected["model_id"],
        "namespace": expected["namespace"],
        "serving runtime": expected["runtime"],
        "storage": expected["storage_uri"],
    }
    for label, value in review_expected.items():
        text = clean(review.get(label, ""))
        flags = re.I if label == "model" else 0
        if label == "storage":
            uri = re.search(r"(?:oci|s3|pvc)://[^\s`*|<>]+", text)
            agrees = uri is not None and same("storage_uri", uri[0], value)
        else:
            agrees = value is not None and bool(
                re.search(r"(?<![\w/.-])" + re.escape(str(value)) + r"(?![\w/.-])", text, flags)
            )
        if not agrees:
            errors.append(f"review {label} does not contain the planned value {value!r}")
    if gpu_selection(review.get("gpus", "")) != (
        data.get("gpu_count"),
        gpu_type(data.get("gpu_type", "")),
    ):
        errors.append("review GPU count/type differs from the structured plan")
    for label, key in (("replicas", "replicas"), ("tensor parallelism", "tensor_parallel")):
        text = clean(review.get(label, ""))
        number = re.match(r"(\d+)\b", text)
        if (label == "replicas" or text) and (
            not number or not same(key, number[1], expected[key])
        ):
            errors.append(f"review {label} differs from the structured plan")
    if intent.storage_uri and not same(
        "storage_uri", call.arguments.get("storage_uri"), intent.storage_uri
    ):
        errors.append("execution storage_uri differs from the customer-supplied URI")
    return errors
